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

# ── Module imports ────────────────────────────────────────────────────────────
from fog_density         import estimate_fog_density
from object_detect       import process_frame
from road_context        import get_road_context
from voice_alert         import speak_alert
from llm                 import get_llm_decision
from dehaze              import DehazeModel
from red_glow            import detect_red_glow          # NEW – LLD module 5
from risk_score          import compute_risk              # NEW – LLD module 9

# NEW: Enhanced modules
from temporal_fog_predictor import TemporalFogPredictor
from confidence_gated_alerts import ConfidenceGatedAlerts
from alert_hysteresis import AlertHysteresis
from sensor_fusion import SensorFusion
from scene_memory_buffer import SceneMemoryBuffer
from evaluation_metrics import LatencyProfiler

# NEW: Advanced ADAS modules
from roi_mask             import ROIMask
from lane_detection       import LaneDetector
from object_tracker       import ObjectTracker
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
        self.scene_memory = SceneMemoryBuffer(buffer_size=5, time_window_seconds=30.0)
        self.latency_profiler = LatencyProfiler()

        # ── NEW: Advanced ADAS components ───────────────────────────────────
        self.roi_mask = ROIMask(frame_width=640, frame_height=480)
        self.lane_detector = LaneDetector(frame_width=640, frame_height=480)
        self.object_tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        self.motion_analyzer = MotionAnalyzer(fps=30.0)
        self.distance_estimator = EnhancedDistanceEstimator(frame_height=480)
        self.risk_alerts = RiskBasedAlertSystem(
            ttc_warning_threshold=5.0,
            ttc_critical_threshold=2.0,
            min_distance_warning=20.0,
            min_distance_critical=10.0,
        )
        self.fog_preprocessor = FogAwarePreprocessor(frame_width=640, frame_height=480)

        self._prev_frame_time = 0.0
        self._ego_speed_kmh = 50.0

    # ── Main entry point ───────────────────────────────────────────────────────

    def process(
        self,
        frame: np.ndarray,
        lat:   float,
        lon:   float,
        is_video: bool = False,
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

        # Start latency profiling
        self.latency_profiler.start_pipeline()

        # ── 1. Resize ─────────────────────────────────────────────────────────
        frame = cv2.resize(frame, (640, 480))

        # ── 2. Fog-aware preprocessing ────────────────────────────────────────
        fog_condition = self.fog_preprocessor.analyze_fog(frame)
        fog_density = fog_condition.density
        visibility = fog_condition.visibility
        recommended_speed = fog_condition.recommended_speed
        adaptive_conf_threshold = fog_condition.confidence_threshold

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

        # Apply adaptive detection threshold to YOLO model
        import object_detect as od
        od.model.conf = adaptive_conf_threshold

        self.cached_fog_data = fog_data

        # ── 3. ROI masking - ignore dashboard/hood region ───────────────────
        roi_filtered_frame = self.roi_mask.apply(frame)

        # ── 4. Lane detection ────────────────────────────────────────────────
        lane_info = self.lane_detector.detect_lanes(frame)

        # ── 5. Dehaze (LLD module 3) ──────────────────────────────────────────
        if fog_density >= 30:
            enhanced_frame = self.fog_preprocessor.apply_dehaze(frame)
        else:
            enhanced_frame = dehaze_model.process(frame)

        # ── 6. Object detection (LLD module 4) ───────────────────────────────
        annotated_frame, detections = process_frame(enhanced_frame)

        # Apply fog-adaptive detection filtering
        detections = self.fog_preprocessor.apply_adaptive_detection(detections, frame)

        # Filter detections by ROI
        detections = self.roi_mask.filter_detections(detections)

        # ── 8. Distance estimation & Sensor Fusion ───────────────────────────
        if is_video:
            for d in detections:
                d["distance"] = 20.0
                d["distance_source"] = "constant_video"
            fusion_stats = {
                "total_detections": len(detections),
                "lidar_distances_used": 0,
                "vision_distances_used": 0,
                "lidar_connected": False
            }
            fusion_result = {
                "fused_detections": detections,
                "conflicts": [],
                "fusion_stats": fusion_stats
            }
        else:
            # First get dynamic vision estimation
            detections = self.distance_estimator.estimate_distances(detections, frame.shape[0])
            # Fuse with LiDAR if available
            fusion_result = self.sensor_fusion.fuse_detections(detections, frame.shape[:2])
            detections = fusion_result["fused_detections"]

        # ── 7. Object tracking ────────────────────────────────────────────────
        tracked_objects = self.object_tracker.update(detections, now)

        # Add track IDs to detections
        track_id_map = {tuple(d["bbox"]): t.get("track_id", 0) for d, t in zip(detections, tracked_objects)}
        for det in detections:
            bbox_key = tuple(det["bbox"])
            if bbox_key in track_id_map:
                det["track_id"] = track_id_map[bbox_key]

        # ── 9. Motion analysis and TTC calculation ──────────────────────────
        motion_analysis = self.motion_analyzer.analyze_motion(
            tracked_objects,
            frame_height=frame.shape[0],
            ego_speed_kmh=self._ego_speed_kmh,
        )

        # ── 10. Filter objects by lane ────────────────────────────────────────
        in_lane_objects, out_of_lane_objects = self.lane_detector.filter_objects_in_lane(
            detections, lane_info
        )

        # ── 11. Risk-based alert generation ──────────────────────────────────
        risk_alerts = self.risk_alerts.evaluate_alerts(
            in_lane_objects,
            motion_analysis,
            now
        )

        # NEW: Confidence-gated alerts
        alert_gating = self.confidence_gating.update_detections(detections)

        # ── 12. Red Glow / Brake-light detection (LLD module 5) ───────────────
        glow_result   = detect_red_glow(annotated_frame, detections)
        red_glow      = glow_result["red_glow"]
        annotated_frame = glow_result["glow_frame"]   # now has glow overlays

        # Add lane overlay to annotated frame
        if lane_info.get("detected", False):
            annotated_frame = self.lane_detector.get_lane_overlay(annotated_frame, lane_info, color=(0, 255, 255))

        # ── 13. Distance estimation (using enhanced estimator) ───────────────
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
                # NEW: Use LiDAR-based road context instead of GPS simulator
                self.cached_road_context = self.sensor_fusion.get_road_context_fusion()
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
            recommended_speed   = float(recommended_speed),
            road_context        = road_context,
            red_glow            = red_glow,
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
            "red_glow":           red_glow,
            "distance_to_nearest":distance_to_nearest,
            "nearest_object":     nearest_label,
            "location":           {"latitude": lat, "longitude": lon},
        }

        # ── 18. Rule Engine → Alerts (LLD module 8) ────────────────────────
        alerts = _rule_engine(
            context             = context,
            risk_result         = risk_result,
            fog_density         = fog_density,
            red_glow            = red_glow,
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

        # ── 19. LLM (gated — every 5 s) ──────────────────────────────────────
        if (now - self.last_llm_time) > self.llm_interval_s:
            # NEW: Add current frame to scene memory
            self.scene_memory.add_frame_data(context)

            # Get temporal context for LLM
            temporal_context = self.scene_memory.get_context_for_llm()

            # Enhance context with temporal information
            enhanced_context = {**context, "temporal_context": temporal_context}

            self.cached_llm_response = get_llm_decision(enhanced_context)
            self.last_llm_time       = now

        llm_response = self.cached_llm_response

        if llm_response and llm_response != self.last_llm_spoken:
            if isinstance(llm_response, dict):
                alert_part = llm_response.get("voice_alert", "")
            else:
                alert_part = _extract_alert_from_llm(llm_response)
            if alert_part and fog_density > 50 and risk_score > 50:
                speak_alert("llm_recommendation", alert_part)
                self.last_llm_spoken = str(llm_response)

        # ── 20. FPS ───────────────────────────────────────────────────────────
        fps = round(1.0 / max(time.time() - t_start, 1e-6), 2)

        # NEW: End latency profiling
        pipeline_latency = self.latency_profiler.end_pipeline()

        # ── 21. Compose full result ──────────────────────────────────────────
        result = {
            "frame":               annotated_frame,
            "fog_data":            fog_data,
            "detections":          detections,
            "road_context":        road_context,
            "alerts":              hysteresis_alerts,  # NEW: Hysteresis-filtered alerts
            "llm_response":        llm_response,
            "fps":                 fps,
            "context":             context,
            # NEW fields
            "red_glow":            red_glow,
            "glow_boxes":          glow_result["glow_boxes"],
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
            "scene_memory":        self.scene_memory.get_memory_stats(),
            "pipeline_latency_ms": pipeline_latency * 1000,
            "alert_hysteresis":    self.alert_hysteresis.get_all_alerts_status(),
            # NEW: Advanced ADAS features
            "roi_mask": {
                "enabled": True,
                "roi_polygon": self.roi_mask.get_roi_polygon(),
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
    red_glow:            bool,
    distance_to_nearest: float,
    detections:          list,
    traffic_signal:      str,
) -> list:
    """
    LLD module 8 — Rule filter.
    This module returns no hard-coded alert messages.
    Alerting is driven by the LLM and risk engine instead.
    """
    return []


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
