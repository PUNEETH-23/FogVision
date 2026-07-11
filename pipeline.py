"""
pipeline.py
-----------
Core ADAS processing pipeline.

Frame sampling contract
-----------------------
The VideoEngine already delivers ONE frame per source-second
(skipping the other 29 frames of each 30-fps second).
This module therefore runs the FULL inference chain on every
frame it receives.  Interval-gating is kept for the heavy
network calls (GPS / LLM) but object detection, fog estimation,
red-glow run every delivered frame
(i.e. once per second of video).

LLD modules wired in
--------------------
1.  Frame Acquisition  — VideoEngine (caller side)
2.  Fog Density        — fog_density.estimate_fog_density()
3.  Image Enhancement  — dehaze.DehazeModel()
4.  Object Detection   — object_detect.process_frame()
5.  Red Glow           — red_glow.detect_red_glow()
6.  Sensor Fusion      — context dict built here
7.  Rule Engine        — rule_engine() below
8.  Risk Score         — risk_score.compute_risk()
9.  LLM                — llm.get_llm_decision()
10. Output / Alerts    — voice_alert + return dict
"""

import cv2
import time
import numpy as np
import threading

# ── Module imports ────────────────────────────────────────────────────────────
from fog_density         import estimate_fog_density
from object_detect       import process_frame
from road_context        import get_road_context
from voice_alert         import speak_alert
from llm                 import get_llm_decision, DrivingContext
from dehaze              import DehazeModel
from risk_score          import compute_risk              # NEW – LLD module 9

# NEW: Enhanced modules
from temporal_fog_predictor import TemporalFogPredictor
from confidence_gated_alerts import ConfidenceGatedAlerts
from alert_hysteresis import AlertHysteresis
from sensor_fusion import SensorFusion
from evaluation_metrics import LatencyProfiler

# from lane_detection       import LaneDetector
from motion_ttc           import MotionAnalyzer
from improved_distance    import EnhancedDistanceEstimator
from risk_alerts          import RiskBasedAlertSystem
from fog_aware            import FogAwarePreprocessor


# ── Singleton dehaze model ─────────────────────────────────────────────────────
dehaze_model = DehazeModel()


# ── Pipeline class ─────────────────────────────────────────────────────────────
class ADASPipeline:

    def __init__(self) -> None:

        # ── Interval gates (seconds) ──────────────────────────────────────────
        # Frame acquisition = 1 fps (handled by VideoEngine).
        # All visual modules run every delivered frame.
        # Only expensive network/LLM calls are time-gated here.
        self.llm_interval_s = 5.0
        self.gps_interval_s = 10.0

        self.last_llm_time  = 0.0
        self.last_gps_time  = 0.0

        # ── Cache stores ──────────────────────────────────────────────────────
        self.cached_road_context   : dict = {}
        self.cached_llm_response   : str  = ""
        self.cached_fog_data       : dict = {
            "dark_channel":      None,
            "transmission_map":  None,
            "fog_density":       0.0,
            "visibility":        "UNKNOWN",
            "recommended_speed": 0,
        }
        self.last_llm_spoken       : str  = ""
        # Preserved full result for Streamlit reruns that happen between frames
        self.cached_full_result    : dict = {}

        # ── NEW: Enhanced components ──────────────────────────────────────────
        self.temporal_fog = TemporalFogPredictor(window_size=10, alpha=0.3)
        self.confidence_gating = ConfidenceGatedAlerts(confidence_threshold=0.5, frames_required=3)
        self.alert_hysteresis = AlertHysteresis(warn_cooldown=5.0, critical_cooldown=10.0, escalation_threshold=3)
        self.sensor_fusion = SensorFusion()
        self.latency_profiler = LatencyProfiler()

        # Fixed Box Boundary (center 50% width, bottom 50% height of 640x480 space)
        self.fixed_box = [160, 240, 480, 432]
        self.motion_analyzer = MotionAnalyzer(fps=30.0)
        self.distance_estimator = EnhancedDistanceEstimator(frame_height=480)
        self.risk_alerts = RiskBasedAlertSystem(
            ttc_warning_threshold=5.0,
            ttc_critical_threshold=2.0,
            min_distance_warning=20.0,
            min_distance_critical=10.0,
        )
        self.fog_preprocessor = FogAwarePreprocessor(frame_width=640, frame_height=480)

        # Threading and caching fields for background processing
        self._llm_in_progress = False
        self._llm_lock = threading.Lock()
        
        self._fog_in_progress = False
        self._fog_lock = threading.Lock()
        self._cached_fog_condition = None
        self._last_fog_time = 0.0

        self._prev_frame_time = 0.0
        self._ego_speed_kmh = 50.0

        # Relative Velocity & Obstacle history
        self.prev_center_dist = None
        self.prev_dist_time = None
        self.relative_velocity = 0.0
        self.alpha_v = 0.3
        self._has_calculated_fog = False
        self.initial_fog = None


    def add_synthetic_haze(self, frame: np.ndarray, intensity_pct: float) -> np.ndarray:
        if intensity_pct <= 0:
            return frame
        alpha = (intensity_pct / 100.0) * 0.7
        haze = np.full_like(frame, 220, dtype=np.uint8)
        return cv2.addWeighted(frame, 1.0 - alpha, haze, alpha, 0)

    # ── Main entry point ───────────────────────────────────────────────────────

    def process(
        self,
        frame:            np.ndarray,
        lat:              float,
        lon:              float,
        is_video:         bool = False,
        speed_kmh:        float = 50.0,
        is_live:          bool = False,
        haze_intensity:   float = 0.0,
        esp32_ip:         str = "localhost",
        control_mode:     str = "Manual Control",
        cruising_speed:   float = 50.0
    ) -> dict:
        """
        Run the full LLD inference chain on one (already-sampled) frame.

        Parameters
        ----------
        frame : BGR ndarray from VideoEngine.read()
        lat   : GPS latitude  (float)
        lon   : GPS longitude (float)

        Returns
        -------
        Large result dict consumed by DashBoard.py.
        """

        t_start = time.time()
        now     = t_start
        self._ego_speed_kmh = speed_kmh

        # Start latency profiling
        self.latency_profiler.start_pipeline()

        # Apply synthetic haze if running in video mode and haze intensity > 0
        # (Disabled for live feed per user request)
        if not is_live and haze_intensity > 0.0:
            frame = self.add_synthetic_haze(frame, haze_intensity)

        # Initialize motor control speed/turn defaults
        speed = 0
        turn = 0

        # ── 1. Resize ─────────────────────────────────────────────────────────
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, (640, 480))
                frame = gpu_resized.download()
            else:
                frame = cv2.resize(frame, (640, 480))
        except Exception:
            frame = cv2.resize(frame, (640, 480))

        # Check if pre-calculated initial_fog exists (for video modes)
        if self.initial_fog is not None:
            if self._cached_fog_condition is None:
                self._cached_fog_condition = self.fog_preprocessor.set_condition_density(self.initial_fog)
                self._has_calculated_fog = True
            should_calc_fog = False
        else:
            if self._cached_fog_condition is None:
                self._cached_fog_condition = self.fog_preprocessor._current_condition
            
            should_calc_fog = False
            if not self._has_calculated_fog:
                should_calc_fog = True

        if should_calc_fog and not self._fog_in_progress:
            self._last_fog_time = now
            if not self._has_calculated_fog:
                # Calculate synchronously on the very first frame of Live Feed to show correct value immediately
                try:
                    cond = self.fog_preprocessor.analyze_fog(frame)
                    self._cached_fog_condition = cond
                    self._has_calculated_fog = True
                except Exception as e:
                    print(f"[Fog Sync Startup] Error: {e}")
            else:
                self._fog_in_progress = True
                def run_fog_async(f):
                    try:
                        cond = self.fog_preprocessor.analyze_fog(f)
                        with self._fog_lock:
                            self._cached_fog_condition = cond
                            self._has_calculated_fog = True
                    except Exception as e:
                        print(f"[Fog Thread] Error: {e}")
                    finally:
                        self._fog_in_progress = False

                t_fog = threading.Thread(target=run_fog_async, args=(frame.copy(),), daemon=True)
                t_fog.start()

        with self._fog_lock:
            fog_condition = self._cached_fog_condition

        fog_density = fog_condition.density
        visibility = fog_condition.visibility
        recommended_speed = fog_condition.recommended_speed

        fog_data = {
            "fog_density": fog_density,
            "visibility": visibility,
            "recommended_speed": recommended_speed,
        }

        # Temporal fog prediction
        fog_prediction = self.temporal_fog.update(fog_density)
        fog_data.update({
            "predicted_next": fog_prediction["predicted_next"],
            "drift_direction": fog_prediction["drift_direction"],
            "fog_trend": self.temporal_fog.get_trend_analysis()
        })

        self.cached_fog_data = fog_data

        # ── 4. Lane detection (DELETED - replaced with fixed box boundary) ────
        # Fixed box boundary coordinates (rectangular region of interest)
        left_x = 160
        right_x = 480
        top_y = 240
        bottom_y = 432

        fixed_box_polygon = np.array([
            [left_x, bottom_y],
            [left_x, top_y],
            [right_x, top_y],
            [right_x, bottom_y],
        ], dtype=np.int32)

        lane_info = {
            "detected": True,
            "lane_polygon": fixed_box_polygon,
            "horizon_y": 264,
        }

        # ── 5. Object detection (Conditional Dehazing based on 35% Fog Threshold) ──────────────────
        # If fog density > 35%, run dehazing first, then run YOLO.
        # Otherwise, run direct object detection on the raw frame.
        is_dehazed = False
        processed_frame = frame.copy()
        if fog_density > 35.0:
            try:
                processed_frame = dehaze_model.process(frame)
                is_dehazed = True
                print(f"[Pipeline] Fog density {fog_density:.1f}% > 35%. Running DehazeTransformer...")
            except Exception as e:
                print(f"[Pipeline] Dehazing error: {e}. Falling back to raw frame.")
                processed_frame = frame.copy()

        t_det_start = time.time()
        annotated_frame, detections = process_frame(processed_frame, is_video=is_video)
        t_det_ms = (time.time() - t_det_start) * 1000
        print(f"[TIMING] Object detection: {t_det_ms:.1f} ms  ({len(detections)} objects) [Dehazed: {is_dehazed}]")


        # ── 8. Distance estimation & Sensor Fusion ───────────────────────────
        # In video mode, filter objects to only keep those inside the fixed box boundary
        if is_video:
            in_lane_detections = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                is_in_box = (left_x <= cx <= right_x) and (top_y <= cy <= bottom_y)
                if is_in_box:
                    in_lane_detections.append(d)
            detections = in_lane_detections

        # --- Plan A ESP32 Sensors & Connectivity check ---
        esp32_connected = False
        esp32_sensors = {"left": 80.0, "center": 80.0, "right": 80.0}
        
        # Check physical ESP32 connectivity if IP is provided and is_live
        if is_live :
            import esp32_module
            real_data = esp32_module.get_sensor_data(esp32_ip, timeout=0.5)

            if real_data is not None:
                esp32_sensors["left"]   = real_data["left"]
                esp32_sensors["center"] = real_data["center"]
                esp32_sensors["right"]  = real_data["right"]
                esp32_connected = True
                
        # Update self.sensor_fusion status
        if hasattr(self, "sensor_fusion") and self.sensor_fusion:
            self.sensor_fusion.sensor_connected = esp32_connected

        # ── Distance estimation ────────────────────────────────────────────────
        # • Live  mode → ESP32 physical sensors are the SOLE distance source.
        #               MiDaS is NOT used. No torch. No depth map.
        # • Video mode → YOLO bounding-box heuristic only (no MiDaS either).
        for d in detections:
            d["distance_bbox"] = float(d.get("distance", 0.0))
            if is_live and esp32_connected:
                # Assign sensor zone distance to each detection
                x1, y1, x2, y2 = d["bbox"]
                cx = (x1 + x2) // 2
                if cx < 213:
                    zone_dist = esp32_sensors["left"]
                elif cx < 426:
                    zone_dist = esp32_sensors["center"]
                else:
                    zone_dist = esp32_sensors["right"]
                d["distance"]        = zone_dist
                d["distance_source"] = "esp32_sensor"
            else:
                # Video mode: use YOLO bounding-box estimate
                d["distance"]        = float(d.get("distance", 20.0))
                d["distance_source"] = "bounding_box"

        esp32_sensors = {k: round(v, 2) for k, v in esp32_sensors.items()}


        # Draw vertical dividers and section labels in live mode
        if is_live:
            cv2.line(annotated_frame, (213, 0), (213, 480), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(annotated_frame, (426, 0), (426, 480), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "LEFT",   (70,  25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "CENTER", (265, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, "RIGHT",  (490, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw bounding boxes and distance comparisons
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) // 2
            
            if cx < 213:
                obj_section = "left"
            elif cx < 426:
                obj_section = "center"
            else:
                obj_section = "right"
                
            sensor_dist = esp32_sensors[obj_section]
            d["distance_sensor"] = sensor_dist
            d["sensor_section"] = obj_section
            
            if is_live and esp32_connected:
                d["distance"]        = sensor_dist
                d["distance_bbox"]   = sensor_dist
                d["distance_source"] = "esp32_sensor"
                
            dist_to_use = sensor_dist if (is_live and esp32_connected) else d["distance"]
            box_color = (0, 255, 0)
            if dist_to_use < 0.30:
                box_color = (0, 0, 255) # Red (Critical follow risk)
            elif dist_to_use < 0.60:
                box_color = (0, 255, 255) # Yellow (Warning range)
            else:
                box_color = (0, 255, 0) # Green
                
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            if is_live and esp32_connected:
                label_text = f"{d['label']}: {dist_to_use:.2f}m (Sens)"
            else:
                label_text = f"{d['label']}: {d['distance']}m"
                
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

        # Estimate Relative Velocity of leading vehicle/obstacle in front
        current_time = time.time()
        current_center_dist = esp32_sensors["center"]
        if is_live:
            if self.prev_center_dist is not None and self.prev_dist_time is not None:
                dt = current_time - self.prev_dist_time
                if dt > 0.05:
                    raw_v_rel = (current_center_dist - self.prev_center_dist) / dt
                    raw_v_rel = max(-10.0, min(raw_v_rel, 10.0))  # filter spikes
                    self.relative_velocity = self.alpha_v * raw_v_rel + (1.0 - self.alpha_v) * self.relative_velocity
                    self.prev_center_dist = current_center_dist
                    self.prev_dist_time = current_time
            else:
                self.relative_velocity = 0.0
                self.prev_center_dist = current_center_dist
                self.prev_dist_time = current_time
        else:
            self.relative_velocity = 0.0

        # Imminent collision warning checks
        automated_braking = False
        
        # Determine critical limits
        limit_dist = 0.30 if is_live else 10.0
        limit_ttc = 0.8 if is_live else 2.0
        
        # Check proximity distances (using all 3 zones from esp32_sensors, which holds either real ESP32 or simulated camera distances)
        left_dist = esp32_sensors.get("left", 80.0)
        mid_dist = esp32_sensors.get("center", 80.0)
        right_dist = esp32_sensors.get("right", 80.0)
        
        # Proximity-based stop
        if (left_dist < limit_dist) or (mid_dist < limit_dist) or (right_dist < limit_dist):
            automated_braking = True
            
        # Also check camera detections directly
        for d in detections:
            dist = d.get("distance", 80.0)
            if 0.0 < dist < limit_dist:
                automated_braking = True
                
        # TTC-based stop
        # Calculate speed in m/s for TTC calculation:
        # For live prototype vehicle, the speed is raw motor speed (0-100), where 100 corresponds to ~1.0 m/s
        # For video mode, speed is in km/h, converted to m/s
        speed_val_kmh = float(self._ego_speed_kmh)
        if is_live:
            speed_m_s = speed_val_kmh / 100.0
        else:
            speed_m_s = speed_val_kmh / 3.6

        if speed_m_s > 0.01:
            # Check sensor closing rate TTC
            if is_live and self.relative_velocity < -0.05:
                closing_speed = -self.relative_velocity
                ttc_sensor = current_center_dist / closing_speed
                if ttc_sensor < limit_ttc:
                    automated_braking = True

            # Check camera detection TTC
            for d in detections:
                dist = float(d.get("distance", d.get("distance_bbox", 0.0)))
                if dist > 0:
                    ttc = dist / speed_m_s
                    if ttc < limit_ttc:
                        automated_braking = True

        if automated_braking:
            speak_alert("automated_braking", "Collision hazard! Automated braking engaged.")

        # Real-time manual keyboard control and Adaptive Cruise Control (ACC)
        if is_live:
            import keyboard
            
            # Read steering inputs manually in both Manual and ACC modes
            manual_turn = 0
            if keyboard.is_pressed("left"):
                manual_turn = -80
            elif keyboard.is_pressed("right"):
                manual_turn = 80
            
            if control_mode == "Adaptive Cruise Control (ACC)":
                # ACC PD-Controller
                v_cruising = float(cruising_speed)
                d_target = 0.50 # 50 cm target follow distance
                
                if current_center_dist > 1.2:
                    # Clear path ahead: travel at cruising speed
                    speed = int(v_cruising)
                else:
                    error_dist = current_center_dist - d_target
                    kp = 80.0
                    kd = 30.0
                    speed_calc = v_cruising + kp * error_dist + kd * self.relative_velocity
                    speed = int(max(0.0, min(speed_calc, v_cruising)))
                
                # Assign steering
                turn = manual_turn
            else:
                # Manual driving controls
                if keyboard.is_pressed("up"):
                    speed = int(cruising_speed) # use set cruising speed as manual speed limit
                elif keyboard.is_pressed("down"):
                    speed = -int(cruising_speed)
                else:
                    speed = 0
                    
                turn = manual_turn
                
            # Apply fog-based safe speed limit
            # Calculate max safe speed limit based on estimated fog density:
            max_safe_speed = max(20.0, 100.0 - fog_density)

            # Map max safe speed (km/h) to motor scale (0-100)
            fog_speed_cap = int((max_safe_speed / 120.0) * 100)
            # Ensure a minimum cap of 15 so the car doesn't completely stall
            fog_speed_cap = max(15, fog_speed_cap)
            
            if speed > 0:
                speed = min(speed, fog_speed_cap)
            elif speed < 0:
                speed = max(speed, -fog_speed_cap)

            # Apply LLM recommended speed limit if available as a safety threshold ceiling
            llm_speed_limit = None
            if self.cached_llm_response:
                resp_dict = None
                if isinstance(self.cached_llm_response, dict):
                    resp_dict = self.cached_llm_response
                elif isinstance(self.cached_llm_response, str):
                    try:
                        import json
                        resp_dict = json.loads(self.cached_llm_response)
                    except Exception:
                        pass
                
                if resp_dict and "recommended_speed" in resp_dict:
                    rec_speed_val = resp_dict["recommended_speed"]
                    import re
                    digits = re.findall(r"\d+", str(rec_speed_val))
                    if digits:
                        llm_speed_limit = float(digits[0])
            
            if llm_speed_limit is not None:
                # Map LLM speed limit (km/h) to motor speed scale (0-100)
                # Map 120 km/h to 100 motor units
                motor_speed_cap = int((llm_speed_limit / 120.0) * 100)
                # Ensure a minimum cap of 15 so the car doesn't completely stall
                motor_speed_cap = max(15, motor_speed_cap)
                
                if speed > 0:
                    speed = min(speed, motor_speed_cap)
                elif speed < 0:
                    speed = max(speed, -motor_speed_cap)
                
            # Emergency Stop override: force speed to zero if obstacle is dangerously close
            if automated_braking:
                speed = 0
                turn = 0
                
            # Force stop on space bar
            if keyboard.is_pressed("space"):
                speed = 0
                turn = 0

            # Send motor signals asynchronously to ESP32 with rate limiting
            if esp32_ip:
                current_sent_time = time.time()
                is_stopped_now = (speed == 0 and turn == 0)
                was_stopped_before = (getattr(self, "_last_sent_speed", 0) == 0 and getattr(self, "_last_sent_turn", 0) == 0)
                should_send = not (is_stopped_now and was_stopped_before)
                
                if should_send:
                    time_elapsed = current_sent_time - getattr(self, "_last_sent_time", 0.0)
                    value_changed = (getattr(self, "_last_sent_speed", None) != speed or 
                                     getattr(self, "_last_sent_turn", None) != turn)
                    
                    if not hasattr(self, "_last_sent_time") or value_changed or time_elapsed > 0.08:
                        self._last_sent_time = current_sent_time
                        self._last_sent_speed = speed
                        self._last_sent_turn = turn
                        
                        def send_esp32_control(ip, s, t):
                            import esp32_module
                            esp32_module.send_motor_speed(ip, s, t, timeout=0.30)
                        
                        threading.Thread(
                            target=send_esp32_control, 
                            args=(esp32_ip, speed, turn), 
                            daemon=True
                        ).start()

        # ── Sensor HUD overlay drawn on the frame (live mode only) ───────────
        if is_live and esp32_connected:
            h_frame, w_frame = annotated_frame.shape[:2]
            bar_h = 44
            bar_y = h_frame - bar_h
            # Semi-transparent dark background strip
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, bar_y), (w_frame, h_frame), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.60, annotated_frame, 0.40, 0, annotated_frame)

            left = esp32_sensors["left"]
            center = esp32_sensors["center"]
            right = esp32_sensors["right"]
            print("left", left)
            print("center", center)
            print("right", right)
            zones = [
                ("LEFT", left, 0, 213),
                ("CENTER", center, 213, 426),
                ("RIGHT", right, 426, w_frame),
            ]

            for label, dist_m_val, x0, x1 in zones:
                cx = (x0 + x1) // 2

                if dist_m_val < 0.30:
                    col = (50, 50, 255)      # Red
                elif dist_m_val < 0.70:
                    col = (0, 165, 255)      # Orange
                else:
                    col = (50, 220, 50)      # Green

                lbl_sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (cx - lbl_sz[0] // 2, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )

                dist_str = f"{dist_m_val:.2f} m"

                val_sz, _ = cv2.getTextSize(dist_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.putText(
                    annotated_frame,
                    dist_str,
                    (cx - val_sz[0] // 2, bar_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    col,
                    2,
                    cv2.LINE_AA,
                )

            # Vertical separators inside the bar
            cv2.line(annotated_frame, (213, bar_y), (213, h_frame), (60, 60, 60), 1)
            cv2.line(annotated_frame, (426, bar_y), (426, h_frame), (60, 60, 60), 1)

        # Live comparison data

        sensor_comparison = {
            "left":   {"sensor": esp32_sensors["left"]},
            "center": {"sensor": esp32_sensors["center"]},
            "right":  {"sensor": esp32_sensors["right"]},
        }

        fusion_stats = {
            "total_detections": len(detections),
            "esp32_distances_used": len(detections) if (is_live and esp32_connected) else 0,
            "vision_distances_used": 0 if (is_live and esp32_connected) else len(detections),
            "esp32_connected": esp32_connected
        }
        fusion_result = {
            "fused_detections": detections,
            "conflicts": [],
            "fusion_stats": fusion_stats
        }

        # Ensure all detections have distance_bbox key
        for d in detections:
            if "distance_bbox" not in d:
                d["distance_bbox"] = float(d.get("distance", 0.0))

        # ── 7. Object tracking (DELETED) ──────────────────────────────────────
        tracked_objects = []
        for i, det in enumerate(detections):
            det["track_id"] = 0

        # ── 9. Motion analysis and TTC calculation ──────────────────────────
        motion_analysis = self.motion_analyzer.analyze_motion(
            tracked_objects,
            frame_height=frame.shape[0],
            ego_speed_kmh=self._ego_speed_kmh,
        )

        # ── 10. Filter objects by fixed box boundary ─────────────────────────
        in_lane_objects = []
        out_of_lane_objects = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            is_in_box = (left_x <= cx <= right_x) and (top_y <= cy <= bottom_y)
            if is_in_box:
                in_lane_objects.append(d)
            else:
                out_of_lane_objects.append(d)

        # ── 11. Risk-based alert generation ──────────────────────────────────
        risk_alerts = self.risk_alerts.evaluate_alerts(
            in_lane_objects,
            motion_analysis,
            now
        )

        # NEW: Confidence-gated alerts
        alert_gating = self.confidence_gating.update_detections(detections)

        # Red Glow / Brake-light detection (DELETED)

        # Add fixed box boundary overlay to annotated frame
        cv2.polylines(annotated_frame, [fixed_box_polygon], isClosed=True, color=(0, 255, 255), thickness=3)

        # ── 13. Distance estimation (using enhanced estimator or sensor data only for live feed) ───────────────
        if is_live:
            left_dist = esp32_sensors.get("left", 80.0)
            mid_dist = esp32_sensors.get("center", 80.0)
            right_dist = esp32_sensors.get("right", 80.0)
            distance_to_nearest = min(left_dist, mid_dist, right_dist)
            if distance_to_nearest >= 80.0:
                distance_to_nearest = 0.0
            nearest_label = "obstacle" if distance_to_nearest > 0.0 else "none"
            per_object_distances = [left_dist, mid_dist, right_dist]
        else:
            distances = [d.get('distance', float('inf')) for d in detections if 'distance' in d]
            distance_to_nearest = min(distances) if distances else 0.0
            nearest_label = next((d['label'] for d in detections if d.get('distance') == distance_to_nearest), "UNKNOWN")
            per_object_distances = [d.get('distance', 0.0) for d in detections]

        # ── 14. Traffic-signal extraction ─────────────────────────────────────
        traffic_signal = "UNKNOWN"
        for d in detections:
            if d.get("label") == "traffic light":
                traffic_signal = d.get("traffic_light_color") or "UNKNOWN"

        # ── 15. GPS + Road context (gated — every 10 s) ─────────────────────
        if (now - self.last_gps_time) > self.gps_interval_s:
            try:
                self.cached_road_context = get_road_context(lat, lon)
            except Exception as exc:
                print(f"[Pipeline] Road context error: {exc}")
                self.cached_road_context = {
                    "road":       "Unknown",
                    "road_type":  "Unknown",
                    "blackspots": [],
                }
            self.last_gps_time = time.time()

        road_context = self.cached_road_context

        # ── 16. Risk Score Engine (LLD module 9) ─────────────────────────────
        risk_result = compute_risk(
            fog_density         = fog_density,
            distance_to_nearest = distance_to_nearest,
            actual_speed        = float(self._ego_speed_kmh),
            road_context        = road_context,
            is_live             = is_live,
        )
        risk_score  = risk_result["risk_score"]
        risk_level  = risk_result["risk_level"]   # replaces fog-only risk_level

        # ── 17. Sensor-fusion context dict (LLD module 7) ───────────────────
        context = {
            "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
            "fog_density":        fog_density,
            "risk_level":         risk_level,
            "risk_score":         risk_score,
            "visibility":         visibility,
            "recommended_speed":  recommended_speed,
            "road_name":          road_context.get("road",      "Unknown"),
            "road_type":          road_context.get("road_type", "Unknown"),
            "blackspot_nearby":   len(road_context.get("blackspots", [])) > 0,
            "objects":            detections,
            "traffic_signal":     traffic_signal,
            "distance_to_nearest":distance_to_nearest,
            "nearest_object":     nearest_label,
            "location":           {"latitude": lat, "longitude": lon},
            "current_speed_kmh":  self._ego_speed_kmh,
            "is_live":            is_live,
            "esp32_sensor_data":  esp32_sensors,
        }

        # ── 18. Rule Engine → Alerts (LLD module 8) ────────────────────────
        alerts = _rule_engine(
            context             = context,
            risk_result         = risk_result,
            fog_density         = fog_density,
            distance_to_nearest = distance_to_nearest,
            detections          = detections,
            traffic_signal      = traffic_signal,
        )

        # NEW: Apply alert hysteresis
        hysteresis_alerts = []
        for alert in alerts:
            hysteresis_result = self.alert_hysteresis.process_alert(
                alert_type=f"rule_{alert.get('type', 'unknown')}",
                should_trigger=True  # Rules determine if alert should trigger
            )
            if hysteresis_result["should_alert"]:
                hysteresis_alerts.append({
                    **alert,
                    "hysteresis_level": hysteresis_result["alert_level"],
                    "escalated": hysteresis_result["escalated"]
                })

        # ── 19. LLM (Asynchronous background thread) ──────────────────────────
        is_alert_active = (risk_level in ["HIGH", "MEDIUM"]) or len(hysteresis_alerts) > 0
        time_since_last_llm = now - self.last_llm_time
        if not self._llm_in_progress and ((time_since_last_llm > self.llm_interval_s) or (is_alert_active and time_since_last_llm > 1.5)):
            self._llm_in_progress = True
            self.last_llm_time = now

            # Derive trend and construct structured DrivingContext
            fog_trend_analysis = self.temporal_fog.get_trend_analysis()
            fog_trend_raw = fog_trend_analysis.get("trend", "stable") if isinstance(fog_trend_analysis, dict) else "stable"
            trend_map = {
                "rapidly_increasing":  "worsening",
                "gradually_increasing":"worsening",
                "rapidly_decreasing":  "improving",
                "gradually_decreasing":"improving",
                "stable":              "stable",
                "insufficient_data":   "stable",
            }
            fog_trend_llm = trend_map.get(fog_trend_raw, "stable")

            driving_ctx = DrivingContext(
                fog_density=fog_density,
                fog_trend=fog_trend_llm,
                nearest_object_m=distance_to_nearest if distance_to_nearest > 0 else 999.0,
                nearest_object_label=nearest_label,
                road_type=road_context.get("road_type", "highway").lower(),
                current_speed_kmh=self._ego_speed_kmh,
            )

            def run_llm_async(ctx, alert_active, current_fog, current_risk):
                try:
                    response = get_llm_decision(ctx)
                    with self._llm_lock:
                        self.cached_llm_response = response
                        if response and response != self.last_llm_spoken:
                            if isinstance(response, dict):
                                alert_part = response.get("voice_alert", "")
                            else:
                                alert_part = _extract_alert_from_llm(response)
                            
                            should_speak = alert_part and (alert_active or (current_fog > 50 and current_risk > 50)) and self._ego_speed_kmh >= 3.0
                            if should_speak:
                                speak_alert("llm_recommendation", alert_part)
                                self.last_llm_spoken = str(response)
                except Exception as e:
                    print(f"[LLM Thread] Error: {e}")
                finally:
                    self._llm_in_progress = False

            t_llm = threading.Thread(
                target=run_llm_async,
                args=(driving_ctx, is_alert_active, fog_density, risk_score),
                daemon=True
            )
            t_llm.start()

        llm_response = self.cached_llm_response

        # ── 20. FPS ───────────────────────────────────────────────────────────
        fps = round(1.0 / max(time.time() - t_start, 1e-6), 2)

        # NEW: End latency profiling
        pipeline_latency = self.latency_profiler.end_pipeline()

        # ── 21. Compose full result ──────────────────────────────────────────
        result = {
            "frame":               annotated_frame,
            "fog_data":            fog_data,
            "detections":          detections,
            "is_dehazed":          is_dehazed,
            "road_context":        road_context,
            "alerts":              hysteresis_alerts,  # NEW: Hysteresis-filtered alerts
            "llm_response":        llm_response,
            "fps":                 fps,
            "context":             context,
            "automated_braking":   automated_braking,
            "esp32_sensor_data":   esp32_sensors,
            "sensor_comparison":   sensor_comparison,
            "relative_velocity":   self.relative_velocity,
            "target_speed":        speed,
            "target_turn":         turn,
            "control_mode":        control_mode,
            # NEW fields
            "distance_to_nearest": distance_to_nearest,
            "nearest_label":       nearest_label,
            "per_object_distances": per_object_distances,
            "risk_score":          risk_score,
            "risk_level":          risk_level,
            "risk_components":     risk_result["component_scores"],
            "hard_override":       risk_result["hard_override"],
            "override_reason":     risk_result["override_reason"],
            # NEW: Enhanced features
            "temporal_fog":        fog_prediction,
            "confidence_gating":   alert_gating,
            "sensor_fusion":       fusion_result["fusion_stats"],
            "fusion_conflicts":    fusion_result["conflicts"],
            "pipeline_latency_ms": pipeline_latency * 1000,
            "alert_hysteresis":    self.alert_hysteresis.get_all_alerts_status(),
            "roi_mask": {
                "enabled": False,
                "roi_polygon": [],
            },
            "lane_detection": {
                "detected": lane_info.get("detected", False),
                "lane_polygon": lane_info.get("lane_polygon", []).tolist() if hasattr(lane_info.get("lane_polygon"), "tolist") else [],
            },
            "object_tracking": {
                "tracked_objects": tracked_objects,
                "in_lane_count": len(in_lane_objects),
                "out_of_lane_count": len(out_of_lane_objects),
            },
            "motion_analysis": {
                "analysis": motion_analysis,
                "collision_threats": self.motion_analyzer.get_collision_threats(motion_analysis),
            },
            "risk_alerts": {
                "alerts": risk_alerts,
                "statistics": self.risk_alerts.get_alert_statistics(),
            },
            "fog_aware": {
                "fog_condition": {
                    "density": fog_condition.density,
                    "visibility": fog_condition.visibility,
                    "recommended_speed": fog_condition.recommended_speed,
                    "confidence_threshold": fog_condition.confidence_threshold,
                },
                "statistics": self.fog_preprocessor.get_fog_statistics(),
            },
            "in_lane_objects": in_lane_objects,
            "out_of_lane_objects": out_of_lane_objects,
        }

        self.cached_full_result = result
        return result

    # ------------------------------------------------------------------
    # Fog-derived helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recommended_speed_from_fog_density(fog_density: float) -> int:
        if fog_density >= 80:
            return 30
        if fog_density >= 60:
            return 40
        if fog_density >= 40:
            return 50
        if fog_density >= 20:
            return 60
        return 70

    @staticmethod
    def _visibility_from_fog_density(fog_density: float) -> str:
        if fog_density >= 80:
            return "Very Low"
        if fog_density >= 60:
            return "Low"
        if fog_density >= 40:
            return "Moderate"
        return "Good"


# ── Standalone rule engine ─────────────────────────────────────────────────────

def _rule_engine(
    context:             dict,
    risk_result:         dict,
    fog_density:         float,
    distance_to_nearest: float,
    detections:          list,
    traffic_signal:      str,
) -> list:
    """
    LLD module 8 — Rule filter.
    Calculates safety warnings using vehicular speed (TTC) and
    bounding-box distances (live: ESP32 sensor zones, video: YOLO heuristic).
    """
    alerts = []
    speed_kmh = float(context.get("current_speed_kmh", 50.0))
    is_live = bool(context.get("is_live", False))

    # 1. Dynamic speed threshold based on fog density
    max_speed = 100.0
    if is_live:
        # Scale speed threshold for physical prototype vehicle speed limit (max motor speed)
        if fog_density >= 80.0:
            max_speed = 30.0
        elif fog_density >= 60.0:
            max_speed = 50.0
        elif fog_density >= 40.0:
            max_speed = 70.0
    else:
        if fog_density >= 80.0:
            max_speed = 30.0
        elif fog_density >= 60.0:
            max_speed = 45.0
        elif fog_density >= 40.0:
            max_speed = 60.0
        elif fog_density >= 20.0:
            max_speed = 80.0

    if speed_kmh > max_speed:
        alerts.append({
            "type": "overspeeding",
            "message": f"🚨 OVERSPEEDING: Current speed {speed_kmh:.0f} exceeds safe threshold {max_speed:.0f} for fog density {fog_density:.1f}%!"
        })

    # 2. Case scenarios: If speed is zero or very slow (<= 5 km/h), do not raise collision warnings
    if speed_kmh <= 5.0:
        return alerts

    # For simulated highway, speed_m_s = speed_kmh / 3.6.
    # For live prototype vehicle, the speed_kmh is actually raw motor speed (0-100).
    # Let's map it roughly: 100 motor speed is approx 1.0 m/s for miniature car.
    # So speed_m_s = speed_kmh / 100.0 (in m/s).
    if is_live:
        speed_m_s = speed_kmh / 100.0
    else:
        speed_m_s = speed_kmh / 3.6

    # Scale distance warning limits
    crit_limit = 0.30 if is_live else 10.0
    warn_limit = 0.60 if is_live else 20.0
    caution_limit = 1.0 if is_live else 30.0

    # TTC alert thresholds (in seconds)
    crit_ttc = 0.8 if is_live else 2.0
    warn_ttc = 1.5 if is_live else 5.0

    if is_live:
        # Check the three sensor zones
        left_dist = float(context.get("esp32_sensor_data", {}).get("left", 80.0))
        mid_dist = float(context.get("esp32_sensor_data", {}).get("center", 80.0))
        right_dist = float(context.get("esp32_sensor_data", {}).get("right", 80.0))
        
        sensor_zones = [
            ("Left Zone", left_dist),
            ("center Zone", mid_dist),
            ("Right Zone", right_dist),
        ]
        
        for zone_name, dist in sensor_zones:
            ttc = dist / speed_m_s if speed_m_s > 0 else float('inf')
            
            if dist < crit_limit or ttc < crit_ttc:
                alerts.append({
                    "type": "collision_critical",
                    "message": f"🚨 CRITICAL: Collision imminent on {zone_name}! (Dist: {dist:.2f}m, TTC: {ttc:.1f}s)"
                })
            elif dist < warn_limit or ttc < warn_ttc:
                alerts.append({
                    "type": "distance_warning",
                    "message": f"⚠️ WARNING: Obstacle on {zone_name} at close range (Dist: {dist:.2f}m, TTC: {ttc:.1f}s)"
                })
            elif dist < caution_limit:
                alerts.append({
                    "type": "caution",
                    "message": f"⚠️ CAUTION: Obstacle detected on {zone_name} (Dist: {dist:.2f}m)"
                })
    else:
        for d in detections:
            dist_bbox = float(d.get("distance_bbox", d.get("distance", 0.0)))
            dist_midas = float(d.get("distance_midas", d.get("distance", 0.0)))
            label = d.get("label", "object")

            # Time-to-Collision (TTC) in seconds
            ttc_bbox = dist_bbox / speed_m_s if speed_m_s > 0 else float('inf')
            ttc_midas = dist_midas / speed_m_s if speed_m_s > 0 else float('inf')

            min_ttc = min(ttc_bbox, ttc_midas)

            # Determine severity and message based on speed (TTC) and both distances
            if dist_bbox < crit_limit or dist_midas < crit_limit or ttc_bbox < crit_ttc or ttc_midas < crit_ttc:
                alerts.append({
                    "type": "collision_critical",
                    "message": f"🚨 CRITICAL: Collision imminent with {label}! (Dist: {min(dist_bbox, dist_midas):.2f}m, TTC: {min_ttc:.1f}s)"
                })
            elif dist_bbox < warn_limit or dist_midas < warn_limit or ttc_bbox < warn_ttc or ttc_midas < warn_ttc:
                alerts.append({
                    "type": "distance_warning",
                    "message": f"⚠️ WARNING: {label} ahead at close range (Dist: {min(dist_bbox, dist_midas):.2f}m, TTC: {min_ttc:.1f}s)"
                })
            elif dist_bbox < caution_limit or dist_midas < caution_limit:
                alerts.append({
                    "type": "caution",
                    "message": f"⚠️ CAUTION: {label} detected (Dist: {min(dist_bbox, dist_midas):.2f}m)"
                })

    return alerts


import re


def _extract_alert_from_llm(llm_text: str) -> str:

    """
    Extract short voice alert from LLM response.

    Priority:
    1. Voice Alert
    2. Hazard Alert
    3. Object Detection Alert

    Returns:
        short speech-friendly alert string
    """

    if not llm_text:
        return ""

    # ---------------------------------------------------
    # CLEAN TEXT
    # ---------------------------------------------------

    cleaned = llm_text.replace("*", "")

    # ---------------------------------------------------
    # 1. VOICE ALERT (Highest Priority)
    # ---------------------------------------------------

    voice_match = re.search(

        r"Voice Alert:\s*(.+)",

        cleaned,

        re.IGNORECASE
    )

    if voice_match:

        return voice_match.group(1).strip()

    alerts = []

    # ---------------------------------------------------
    # 2. HAZARD ALERT
    # ---------------------------------------------------

    hazard_match = re.search(

        r"Hazard Alert:\s*(.+)",

        cleaned,

        re.IGNORECASE
    )

    if hazard_match:

        alerts.append(

            hazard_match.group(1).strip()
        )

    # ---------------------------------------------------
    # 3. OBJECT ALERT
    # ---------------------------------------------------

    object_match = re.search(

        r"Object Detection Alert:\s*(.+)",

        cleaned,

        re.IGNORECASE
    )

    if object_match:

        object_alert = object_match.group(1).strip()

        if object_alert:

            alerts.append(object_alert)

    # ---------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------

    if alerts:

        final_alert = ". ".join(alerts)

        # Prevent overly long TTS
        return final_alert[:180]

    return ""
