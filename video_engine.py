"""
video_engine.py
---------------
Frame acquisition wrapper tuned for ADAS pipeline.

Two-phase workflow for uploaded videos
--------------------------------------
Phase 1 — DEHAZE:  Read every frame, apply DehazeModel, write to a
                   temporary dehazed video file.  A progress callback
                   lets the Dashboard show a progress bar.
Phase 2 — DETECT:  Open the dehazed file and deliver frames at the
                   configured `target_fps` (default 10) for real-time
                   object detection.

For live webcam feeds the dehazing phase is skipped and frames are
delivered in real-time (controlled by `live_interval`).
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Callable

import cv2
import numpy as np

VideoSource = Union[int, str]


@dataclass
class VideoFrame:
    frame: "cv2.Mat"
    timestamp: float
    frame_index: int = 0           # absolute frame counter inside the file
    second_index: int = 0          # which second this frame belongs to


class VideoEngine:
    """
    Lightweight frame acquisition wrapper — Streamlit-friendly.

    Create once → keep in st.session_state → call read() on every rerun.
    For uploaded videos the engine delivers frames at `target_fps` (10 fps
    by default).  For webcams, delivery is gated by `live_interval`.
    """

    # Fallback assumed fps when the codec cannot report it
    _DEFAULT_FPS: float = 30.0

    def __init__(
        self,
        source: Optional[VideoSource] = None,
        prefer_file: str = "POV_driving_video_from_inside.mp4",
        target_size: Optional[Tuple[int, int]] = (960, 540),
        target_fps: float = 10.0,
    ) -> None:

        if source is None:
            source = prefer_file if os.path.exists(prefer_file) else 0

        self.source: VideoSource = source
        self.target_size = target_size
        self.target_fps = target_fps

        self._cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(source)

        # How many raw frames make up one second in this source
        reported = self._cap.get(cv2.CAP_PROP_FPS) if self._cap else 0.0
        self._source_fps: float = reported if reported > 1.0 else self._DEFAULT_FPS

        # Frames to skip so we process at target_fps
        # e.g. source=30fps, target=12fps → skip every 2.5 → skip 1 (deliver ~15fps, close enough)
        self._skip_frames: int = max(0, round(self._source_fps / self.target_fps) - 1)

        # Internal counters
        self._raw_frame_index: int = 0      # counts every raw frame read
        self._second_index: int = 0         # counts delivered frames

        # Wall-clock guard for webcam / real-time sources
        self._last_deliver_ts: float = 0.0

        # Always hold a copy of the most recently *delivered* frame so the
        # dashboard can show something before the first real second elapses.
        self._last_delivered: Optional[VideoFrame] = None

        # Live feed processing interval in seconds (default: 1.0s)
        self.live_interval: float = 1.0

        # Whether this engine's video has been pre-dehazed
        self.is_dehazed: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return bool(self._cap) and self._cap.isOpened()

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def skip_frames(self) -> int:
        return self._skip_frames

    # ------------------------------------------------------------------
    # Pre-dehaze the entire video (Phase 1)
    # ------------------------------------------------------------------

    def dehaze_video(
        self,
        output_path: str = "temp_dehazed_video.mp4",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> "VideoEngine":
        """
        Read every frame from the source, dehaze it, and write to a new
        video file.  Returns a NEW VideoEngine that reads from the
        dehazed file.

        Parameters
        ----------
        output_path : str
            Path for the dehazed output video.
        progress_callback : callable(float)
            Called with a 0.0→1.0 progress value after each frame.

        Returns
        -------
        A fresh VideoEngine pointing at the dehazed video, with
        `is_dehazed = True`.
        """
        from dehaze import DehazeModel
        dehaze_model = DehazeModel()

        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1  # prevent div-by-zero

        # Seek to beginning
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get source properties for the output writer
        src_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = self._source_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (src_w, src_h))

        frame_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            
            # Fast DCP fog density check: only dehaze if > 35%
            img_float = frame.astype(np.float64) / 255.0
            dark_channel = np.min(img_float, axis=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dark_channel = cv2.erode(dark_channel, kernel)
            fog_density = float(np.mean(dark_channel) * 100.0)

            if fog_density > 35.0:
                dehazed = dehaze_model.process(frame)
            else:
                dehazed = frame.copy()

            writer.write(dehazed)
            frame_idx += 1

            if progress_callback is not None:
                progress_callback(min(frame_idx / total_frames, 1.0))

        writer.release()
        self.release()

        # Return a new engine pointing at the dehazed video
        engine = VideoEngine(
            source=output_path,
            target_size=self.target_size,
            target_fps=self.target_fps,
        )
        engine.is_dehazed = True
        return engine

    # ------------------------------------------------------------------
    # Public API — frame delivery (Phase 2)
    # ------------------------------------------------------------------

    def read(self) -> Optional[VideoFrame]:
        """
        Return the next frame at the configured delivery rate, or None
        if not yet due / EOF.

        For video files: skips (_skip_frames) raw frames then returns
        the next one — delivering at approximately target_fps.

        For webcams: additionally enforces a wall-clock guard so we
        never deliver faster than live_interval of real time.
        """
        is_live_stream = isinstance(self.source, int) or (
            isinstance(self.source, str) and (
                self.source.startswith("http://") or self.source.startswith("https://")
            )
        )
        if not is_live_stream and self._last_deliver_ts > 0:
            target_interval = 1.0 / self.target_fps
            elapsed = time.time() - self._last_deliver_ts
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        vf = self._read_with_retry(0)
        if not is_live_stream:
            self._last_deliver_ts = time.time()
        return vf

    def _read_with_retry(self, depth: int) -> Optional[VideoFrame]:
        if not self.is_opened or depth > 1:
            return None

        # --- Skip the in-between frames ---------------------------------
        for _ in range(self._skip_frames):
            ok, _ = self._cap.read()
            if not ok:
                if self._try_loop():
                    return self._read_with_retry(depth + 1)
                return None
            self._raw_frame_index += 1

        # --- Read the representative frame ------------------------------
        ok, frame = self._cap.read()
        if not ok:
            if self._try_loop():
                return self._read_with_retry(depth + 1)
            return None

        self._raw_frame_index += 1

        # Webcam/IPcam real-time guard: don't deliver faster than live_interval
        is_live_stream = isinstance(self.source, int) or (
            isinstance(self.source, str) and (
                self.source.startswith("http://") or self.source.startswith("https://")
            )
        )
        if is_live_stream:
            now = time.time()
            if now - self._last_deliver_ts < self.live_interval:
                return self._last_delivered      # not yet a new second
            self._last_deliver_ts = now

        # Resize
        if self.target_size is not None:
            try:
                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    gpu_frame = cv2.cuda_GpuMat()
                    gpu_frame.upload(frame)
                    gpu_resized = cv2.cuda.resize(gpu_frame, self.target_size)
                    frame = gpu_resized.download()
                else:
                    frame = cv2.resize(frame, self.target_size)
            except Exception:
                frame = cv2.resize(frame, self.target_size)

        vf = VideoFrame(
            frame=frame,
            timestamp=time.time(),
            frame_index=self._raw_frame_index,
            second_index=self._second_index,
        )
        self._second_index += 1
        self._last_delivered = vf
        return vf

    def peek_last(self) -> Optional[VideoFrame]:
        """Return the last delivered frame without advancing the stream."""
        return self._last_delivered

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_loop(self) -> bool:
        """Loop video files back to frame 0; no-op for webcams/IPcams. Returns True if looped."""
        is_live_stream = isinstance(self.source, int) or (
            isinstance(self.source, str) and (
                self.source.startswith("http://") or self.source.startswith("https://")
            )
        )
        if not is_live_stream and isinstance(self.source, str):
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._raw_frame_index = 0
                self._second_index = 0
                return True
            except Exception:
                pass
        return False