import os
import time
import threading
import subprocess
import cv2
import torch
import numpy as np
from collections import OrderedDict
from flask import Flask, request, jsonify, render_template, send_from_directory, Response, send_file
from werkzeug.utils import secure_filename
import imageio_ffmpeg

from image_conditioning import preprocess_image, postprocess_image
from models import *
from utils import chw_to_hwc, hwc_to_chw

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Global state for tracking pre-recorded background task progress
progress_lock = threading.Lock()
progress_state = {
    "status": "idle",
    "current_frame": 0,
    "total_frames": 0,
    "fps": 0.0,
    "eta": 0.0,
    "error": None,
    "output_video": None,
    "comparison_video": None
}

def update_progress(status=None, current_frame=None, total_frames=None, fps=None, eta=None, error=None, output_video=None, comparison_video=None):
    with progress_lock:
        if status is not None: progress_state["status"] = status
        if current_frame is not None: progress_state["current_frame"] = current_frame
        if total_frames is not None: progress_state["total_frames"] = total_frames
        if fps is not None: progress_state["fps"] = fps
        if eta is not None: progress_state["eta"] = eta
        if error is not None: progress_state["error"] = error
        if output_video is not None: progress_state["output_video"] = output_video
        if comparison_video is not None: progress_state["comparison_video"] = comparison_video

def load_state_dict(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned[key] = value
    return cleaned

# ==========================================
# LIVE CAMERA MODE BACKEND SYSTEM
# ==========================================

class LiveFrameGrabber(threading.Thread):
    def __init__(self, stream_url):
        super().__init__()
        self.stream_url = stream_url
        self.capture = cv2.VideoCapture(stream_url)
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.connected = False
        self.daemon = True

    def run(self):
        while self.running:
            if not self.capture.isOpened():
                self.connected = False
                time.sleep(1.0)
                self.capture.open(self.stream_url)
                continue
            
            ok, frame = self.capture.read()
            if not ok:
                self.connected = False
                time.sleep(0.01)
                continue
                
            self.connected = True
            t = time.perf_counter()
            with self.lock:
                self.latest_frame = (frame.copy(), t)

    def get_latest(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False
        if self.capture.isOpened():
            self.capture.release()


class LiveStreamManager:
    def __init__(self):
        self.grabber = None
        self.processor_thread = None
        self.active = False
        
        # Current configurations
        self.model_name = 'dehazeformer-t'
        self.resolution = 640
        self.fp16 = True
        self.preprocess_mode = 'video'
        self.postprocess_mode = 'video'
        
        # In-Memory Stream Buffer & Signals
        self.latest_mjpeg = None
        self.latest_dehazed = None
        self.frame_ready_event = threading.Event()
        
        # Performance statistics
        self.stats_lock = threading.Lock()
        self.fps = 0.0
        self.latency = 0.0
        
        # Recording state
        self.recording_lock = threading.Lock()
        self.recording_process = None
        self.recording_path = None
        self.recording_session = None
        self.recording_w = 0
        self.recording_h = 0

    def start(self, url, model_name, resolution, fp16, preprocess_mode, postprocess_mode):
        self.stop()
        
        self.model_name = model_name
        self.resolution = resolution
        self.fp16 = fp16
        self.preprocess_mode = preprocess_mode
        self.postprocess_mode = postprocess_mode
        
        self.grabber = LiveFrameGrabber(url)
        self.grabber.start()
        
        self.active = True
        self.processor_thread = threading.Thread(target=self._processing_loop)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def stop(self):
        self.active = False
        self.stop_recording()
        
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
            self.processor_thread = None
            
        if self.grabber:
            self.grabber.stop()
            self.grabber = None
            
        self.latest_mjpeg = None
        self.latest_dehazed = None
        self.frame_ready_event.clear()
        
        with self.stats_lock:
            self.fps = 0.0
            self.latency = 0.0

    def _processing_loop(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load weights on GPU
        try:
            model_path = os.path.join('./save_models/outdoor', self.model_name + '.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model weights not found at {model_path}")
                
            network = eval(self.model_name.replace('-', '_'))()
            network.load_state_dict(load_state_dict(model_path, device))
            network.to(device)
            if self.fp16 and device.type == 'cuda':
                network.half()
            network.eval()
        except Exception as e:
            print(f"Error loading model in live stream loop: {e}")
            self.active = False
            return

        last_timestamp = 0.0
        frame_timestamps = []
        
        while self.active:
            if not self.grabber or not self.grabber.connected:
                time.sleep(0.05)
                continue
                
            latest = self.grabber.get_latest()
            if not latest:
                time.sleep(0.01)
                continue
                
            frame, t_captured = latest
            if t_captured == last_timestamp:
                time.sleep(0.002)
                continue
                
            last_timestamp = t_captured
            t_proc_start = time.perf_counter()
            
            # 1. Scaling Frame (Max side constraint)
            h_orig, w_orig = frame.shape[:2]
            scale = self.resolution / max(h_orig, w_orig)
            if scale < 1:
                new_w = max(2, int(round(w_orig * scale)))
                new_h = max(2, int(round(h_orig * scale)))
                new_w -= new_w % 2
                new_h -= new_h % 2
                img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                new_w, new_h = w_orig, h_orig
                img = frame.copy()
                
            # 2. Dehaze inference
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                
                with torch.no_grad():
                    tensor = torch.from_numpy(hwc_to_chw(img_rgb * 2 - 1)).unsqueeze(0).to(device)
                    if self.fp16 and device.type == 'cuda':
                        tensor = tensor.half()
                    output = network(tensor).clamp_(-1, 1)
                    output = output * 0.5 + 0.5
                    
                out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy()), self.postprocess_mode)
                out_uint8 = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
                
                # Keep latest dehazed BGR frame for snapshots
                self.latest_dehazed = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
                
                # 3. Compile Side-by-Side Comparison image
                orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                combined = np.concatenate([orig_rgb, out_uint8], axis=1)
                
                # Convert back to BGR for jpeg encoding
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                ok_enc, jpeg_bytes = cv2.imencode('.jpg', combined_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_enc:
                    self.latest_mjpeg = jpeg_bytes.tobytes()
                    self.frame_ready_event.set()
                    self.frame_ready_event.clear()
                    
                # 4. If recording is active, pipe frame
                with self.recording_lock:
                    if self.recording_process:
                        if new_w == self.recording_w and new_h == self.recording_h:
                            try:
                                self.recording_process.stdin.write(out_uint8.tobytes())
                            except Exception as rec_err:
                                print(f"Error piping frame to live recording: {rec_err}")
                                # Stop recording safely outside block
                                
            except Exception as proc_err:
                print(f"Error processing live stream frame: {proc_err}")
                
            # Compute Statistics
            t_proc_end = time.perf_counter()
            cur_latency = (t_proc_end - t_captured) * 1000.0
            
            frame_timestamps.append(t_proc_end)
            now = time.perf_counter()
            frame_timestamps = [ft for ft in frame_timestamps if now - ft < 1.0]
            
            with self.stats_lock:
                self.latency = cur_latency
                self.fps = len(frame_timestamps)

        # Free GPU
        if device.type == 'cuda':
            del network
            torch.cuda.empty_cache()

    def start_recording(self):
        with self.recording_lock:
            if not self.active or self.latest_dehazed is None:
                return False, "Stream must be active to record."
            if self.recording_process:
                return True, "Recording is already active."
                
            h, w = self.latest_dehazed.shape[:2]
            self.recording_w = w
            self.recording_h = h
            
            session_id = str(int(time.time()))
            self.recording_session = session_id
            
            output_dir = os.path.join(app.config['RESULTS_FOLDER'], f'live_record_{session_id}')
            os.makedirs(output_dir, exist_ok=True)
            self.recording_path = os.path.join(output_dir, 'recorded.mp4')
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{w}x{h}',
                '-r', '12.0',
                '-i', '-',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                self.recording_path
            ]
            
            try:
                self.recording_process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, "Recording successfully initialized."
            except Exception as e:
                return False, f"FFmpeg connection failed: {e}"

    def stop_recording(self):
        with self.recording_lock:
            if not self.recording_process:
                return None
                
            try:
                self.recording_process.stdin.close()
                self.recording_process.wait()
            except:
                pass
                
            self.recording_process = None
            url = f"/results/live_record_{self.recording_session}/recorded.mp4"
            self.recording_session = None
            return url


# Initialize single global stream manager
live_manager = LiveStreamManager()

# ==========================================
# PRE-RECORDED VIDEO DEHAZING SYSTEM
# ==========================================

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise FileNotFoundError(f'Could not open video: {video_path}')

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    # Calculate duration and target frame count at 12 fps
    duration = frame_count / fps
    target_frame_count = int(round(duration * 12.0))
    if target_frame_count < 1:
        target_frame_count = 1
        
    # Precompute the set of frame indices to keep
    keep_indices = set(int(round(k * (fps / 12.0))) for k in range(target_frame_count))
    
    update_progress(status="extracting", current_frame=0, total_frames=len(keep_indices))
    
    written = 0
    current_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
            
        if current_idx in keep_indices:
            out_path = os.path.join(output_dir, f'{written:06d}.jpg')
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
            
            if written % 5 == 0 or written == len(keep_indices):
                update_progress(current_frame=written, total_frames=len(keep_indices))
                
        current_idx += 1
        
    capture.release()
    return 12.0, written

def run_dehaze_thread(video_path, model_name, resolution, fp16, preprocess_mode, postprocess_mode, comparison):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    session_id = str(int(time.time()))
    output_session_dir = os.path.join(app.config['RESULTS_FOLDER'], f'web_dehaze_{session_id}')
    os.makedirs(output_session_dir, exist_ok=True)
    
    capture = None
    process_out = None
    process_comp = None
    
    try:
        # 1. Initialize Video Capture
        update_progress(status="extracting", current_frame=0, total_frames=0, error=None)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f'Could not open video: {video_path}')
            
        fps_in = capture.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        
        # Calculate duration and target frame count at 12 fps
        duration = frame_count / fps_in
        target_frame_count = int(round(duration * 12.0))
        if target_frame_count < 1:
            target_frame_count = 1
            
        # Precompute the set of frame indices to keep
        keep_indices = set(int(round(k * (fps_in / 12.0))) for k in range(target_frame_count))
        total_targets = len(keep_indices)
        
        # 2. Read first frame to determine shape
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Could not read any frames from the input video.")
            
        h_orig, w_orig = frame.shape[:2]
        scale = resolution / max(h_orig, w_orig)
        if scale < 1:
            new_w = max(2, int(round(w_orig * scale)))
            new_h = max(2, int(round(h_orig * scale)))
            new_w -= new_w % 2
            new_h -= new_h % 2
        else:
            new_w, new_h = w_orig, h_orig
            
        # 3. Setup Model
        update_progress(status="loading_model")
        model_path = os.path.join('./save_models/outdoor', model_name + '.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        network = eval(model_name.replace('-', '_'))()
        network.load_state_dict(load_state_dict(model_path, device))
        network.to(device)
        if fp16 and device.type == 'cuda':
            network.half()
        network.eval()
        
        # 4. Open FFmpeg subprocesses for H.264 writing
        out_video_name = 'dehazed.mp4'
        out_video_path = os.path.join(output_session_dir, out_video_name)
        
        cmd_out = [
            ffmpeg_exe,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{new_w}x{new_h}',
            '-r', '12.0',
            '-i', '-',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            out_video_path
        ]
        
        process_out = subprocess.Popen(cmd_out, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        process_comp = None
        comp_video_name = None
        if comparison:
            comp_video_name = 'comparison.mp4'
            comp_video_path = os.path.join(output_session_dir, comp_video_name)
            
            cmd_comp = [
                ffmpeg_exe,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{new_w * 2}x{new_h}',
                '-r', '12.0',
                '-i', '-',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                comp_video_path
            ]
            process_comp = subprocess.Popen(cmd_comp, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        update_progress(status="dehazing", current_frame=0, total_frames=total_targets)
        
        # 5. Process loop
        current_idx = 0
        written = 0
        start_time = time.perf_counter()
        
        with torch.no_grad():
            while ok:
                if current_idx in keep_indices:
                    # Resize
                    if scale < 1:
                        img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        img = frame.copy()
                        
                    # Convert to float RGB in [0, 1]
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    
                    # Tensor casting
                    tensor = torch.from_numpy(hwc_to_chw(img_rgb * 2 - 1)).unsqueeze(0).to(device)
                    if fp16 and device.type == 'cuda':
                        tensor = tensor.half()
                        
                    # Run network
                    output = network(tensor).clamp_(-1, 1)
                    output = output * 0.5 + 0.5
                    
                    # Postprocess
                    out_img = postprocess_image(chw_to_hwc(output.detach().cpu().squeeze(0).float().numpy()), postprocess_mode)
                    out_uint8 = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
                    
                    # Write frames to FFmpeg
                    process_out.stdin.write(out_uint8.tobytes())
                    
                    if process_comp:
                        orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        comp_frame = np.concatenate([orig_rgb, out_uint8], axis=1)
                        process_comp.stdin.write(comp_frame.tobytes())
                        
                    written += 1
                    
                    # Update progress
                    elapsed = time.perf_counter() - start_time
                    cur_fps = written / max(elapsed, 0.001)
                    eta = (total_targets - written) / max(cur_fps, 0.001)
                    
                    update_progress(current_frame=written, total_frames=total_targets, fps=cur_fps, eta=eta)
                    
                current_idx += 1
                ok, frame = capture.read()
                
        # 6. Finalize videos
        update_progress(status="compiling")
        
        # Close standard inputs
        if process_out:
            process_out.stdin.close()
            process_out.wait()
            process_out = None
            
        if process_comp:
            process_comp.stdin.close()
            process_comp.wait()
            process_comp = None
            
        # Free CUDA cache
        if device.type == 'cuda':
            del network
            torch.cuda.empty_cache()
            
        update_progress(
            status="completed",
            output_video=f"/results/web_dehaze_{session_id}/{out_video_name}",
            comparison_video=f"/results/web_dehaze_{session_id}/{comp_video_name}" if comparison else None
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # Terminate subprocesses
        if process_out:
            try:
                process_out.kill()
            except:
                pass
        if process_comp:
            try:
                process_comp.kill()
            except:
                pass
                
        # Free VRAM
        if device.type == 'cuda':
            try:
                del network
            except:
                pass
            torch.cuda.empty_cache()
            
        update_progress(status="error", error=str(e))
        
    finally:
        if capture:
            capture.release()

@app.route('/')
def index():
    return render_template('index.html')

# ==========================================
# Flask routes for Video Mode
# ==========================================

@app.route('/process', methods=['POST'])
def process():
    # Check if a task is already running
    with progress_lock:
        if progress_state["status"] not in ["idle", "completed", "error"]:
            return jsonify({"success": False, "error": "A video is already being processed."}), 400
            
    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided."}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename."}), 400
        
    # Get parameters
    model_name = request.form.get('model', 'dehazeformer-t')
    resolution = int(request.form.get('resolution', '1280'))
    fp16 = request.form.get('fp16', 'false').lower() == 'true'
    preprocess_mode = request.form.get('preprocess', 'video')
    postprocess_mode = request.form.get('postprocess', 'video')
    comparison = request.form.get('comparison', 'false').lower() == 'true'
    
    # Save file
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    
    # Start thread
    thread = threading.Thread(
        target=run_dehaze_thread, 
        args=(video_path, model_name, resolution, fp16, preprocess_mode, postprocess_mode, comparison)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({"success": True})

@app.route('/progress')
def get_progress():
    with progress_lock:
        return jsonify(progress_state)

# ==========================================
# Flask routes for Live Camera Mode
# ==========================================

@app.route('/live/start', methods=['POST'])
def live_start():
    url = request.form.get('url')
    if not url:
        return jsonify({"success": False, "error": "IP Camera URL is required."}), 400
        
    model_name = request.form.get('model', 'dehazeformer-t')
    resolution = int(request.form.get('resolution', '640'))
    fp16 = request.form.get('fp16', 'false').lower() == 'true'
    preprocess_mode = request.form.get('preprocess', 'video')
    postprocess_mode = request.form.get('postprocess', 'video')
    
    try:
        live_manager.start(url, model_name, resolution, fp16, preprocess_mode, postprocess_mode)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/live/stop', methods=['POST'])
def live_stop():
    live_manager.stop()
    return jsonify({"success": True})

def live_stream_generator():
    while live_manager.active:
        if live_manager.frame_ready_event.wait(timeout=0.1):
            frame = live_manager.latest_mjpeg
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/live/stream')
def live_stream():
    return Response(live_stream_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live/stats')
def live_stats():
    with live_manager.stats_lock:
        connected = live_manager.grabber.connected if live_manager.grabber else False
        return jsonify({
            "active": live_manager.active,
            "connected": connected,
            "fps": round(live_manager.fps, 1),
            "latency": round(live_manager.latency, 1),
            "is_recording": live_manager.recording_process is not None
        })

@app.route('/live/snapshot')
def live_snapshot():
    frame = live_manager.latest_dehazed
    if frame is None:
        return "No active live frame available for snapshot.", 404
        
    ok, png_bytes = cv2.imencode('.png', frame)
    if not ok:
        return "Failed to encode snapshot image.", 500
        
    from io import BytesIO
    return send_file(
        BytesIO(png_bytes.tobytes()), 
        mimetype='image/png', 
        as_attachment=True, 
        download_name='dehazed_snapshot.png'
    )

@app.route('/live/record/start', methods=['POST'])
def live_record_start():
    success, msg = live_manager.start_recording()
    return jsonify({"success": success, "message": msg})

@app.route('/live/record/stop', methods=['POST'])
def live_record_stop():
    url = live_manager.stop_recording()
    if url:
        return jsonify({"success": True, "video_url": url})
    else:
        return jsonify({"success": False, "error": "No active live camera recording found."}), 400

# ==========================================
# Static results routes
# ==========================================

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
