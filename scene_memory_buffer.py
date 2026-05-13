"""
scene_memory_buffer.py
----------------------
Scene memory buffer for LLM context enhancement.

Maintains rolling buffer of recent scene data to provide temporal context
to LLM decision making, rather than just current frame data.
"""

from typing import List, Dict, Any, Optional
from collections import deque
import time
import numpy as np

class SceneMemoryBuffer:
    """
    Maintains temporal context for LLM decision making.
    """

    def __init__(self, buffer_size: int = 5, time_window_seconds: float = 30.0):
        """
        Parameters
        ----------
        buffer_size : int
            Maximum number of frames to keep in memory
        time_window_seconds : float
            Maximum age of frames to keep (seconds)
        """
        self.buffer_size = buffer_size
        self.time_window = time_window_seconds
        self.memory_buffer: deque = deque(maxlen=buffer_size)

    def add_frame_data(self, frame_data: Dict[str, Any]):
        """
        Add current frame data to memory buffer.

        Parameters
        ----------
        frame_data : dict
            Current frame context data
        """
        timestamp = time.time()

        memory_entry = {
            "timestamp": timestamp,
            "fog_density": frame_data.get("fog_density", 0.0),
            "risk_level": frame_data.get("risk_level", "UNKNOWN"),
            "detections": frame_data.get("objects", []),
            "traffic_signal": frame_data.get("traffic_signal", "UNKNOWN"),
            "red_glow": frame_data.get("red_glow", False),
            "distance_to_nearest": frame_data.get("distance_to_nearest", 0.0),
            "nearest_object": frame_data.get("nearest_object", "NONE"),
            "alerts": frame_data.get("alerts", []),
            "location": frame_data.get("location", {})
        }

        self.memory_buffer.append(memory_entry)

        # Clean old entries
        self._clean_old_entries()

    def get_context_for_llm(self) -> Dict[str, Any]:
        """
        Get formatted context from memory buffer for LLM.

        Returns
        -------
        Context dictionary with temporal information
        """
        if not self.memory_buffer:
            return {"temporal_context": "No historical data available"}

        current = self.memory_buffer[-1]
        historical = list(self.memory_buffer)[:-1]  # All except current

        # Compute temporal trends
        trends = self._compute_temporal_trends()

        context = {
            "current_frame": {
                "fog_density": current["fog_density"],
                "risk_level": current["risk_level"],
                "active_detections": len(current["detections"]),
                "traffic_signal": current["traffic_signal"],
                "red_glow_detected": current["red_glow"],
                "nearest_object_distance": current["distance_to_nearest"],
                "nearest_object_type": current["nearest_object"]
            },
            "temporal_trends": trends,
            "recent_history": self._summarize_recent_history(historical),
            "buffer_stats": {
                "frames_in_memory": len(self.memory_buffer),
                "time_span_seconds": self._get_time_span()
            }
        }

        return context

    def _compute_temporal_trends(self) -> Dict[str, Any]:
        """
        Compute trends from historical data.
        """
        if len(self.memory_buffer) < 2:
            return {"trend": "insufficient_data"}

        entries = list(self.memory_buffer)
        fog_values = [e["fog_density"] for e in entries]

        # Fog density trend
        fog_trend = "stable"
        if len(fog_values) >= 2:
            fog_change = fog_values[-1] - fog_values[0]
            if fog_change > 5:
                fog_trend = "increasing"
            elif fog_change < -5:
                fog_trend = "decreasing"

        # Risk level progression
        risk_levels = [e["risk_level"] for e in entries]
        risk_escalation = "stable"
        if len(risk_levels) >= 2:
            if risk_levels[-1] == "HIGH" and risk_levels[0] != "HIGH":
                risk_escalation = "escalated"
            elif risk_levels[-1] == "LOW" and risk_levels[0] == "HIGH":
                risk_escalation = "de-escalated"

        # Detection consistency
        detection_counts = [len(e["detections"]) for e in entries]
        avg_detections = sum(detection_counts) / len(detection_counts)

        return {
            "fog_trend": fog_trend,
            "fog_change": round(fog_values[-1] - fog_values[0], 1),
            "risk_escalation": risk_escalation,
            "average_detections": round(avg_detections, 1),
            "detection_consistency": "consistent" if np.std(detection_counts) < 1 else "variable"
        }

    def _summarize_recent_history(self, historical: List[Dict]) -> List[Dict]:
        """
        Create summary of recent frames for LLM context.
        """
        summary = []

        for i, entry in enumerate(historical[-3:]):  # Last 3 frames
            summary.append({
                "frames_ago": len(historical) - i,
                "fog_density": entry["fog_density"],
                "risk_level": entry["risk_level"],
                "detections_count": len(entry["detections"]),
                "had_red_glow": entry["red_glow"],
                "nearest_distance": entry["distance_to_nearest"]
            })

        return summary

    def _get_time_span(self) -> float:
        """
        Get time span covered by buffer.
        """
        if len(self.memory_buffer) < 2:
            return 0.0

        return self.memory_buffer[-1]["timestamp"] - self.memory_buffer[0]["timestamp"]

    def _clean_old_entries(self):
        """
        Remove entries older than time window.
        """
        current_time = time.time()
        while self.memory_buffer and (current_time - self.memory_buffer[0]["timestamp"]) > self.time_window:
            self.memory_buffer.popleft()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about memory buffer.
        """
        if not self.memory_buffer:
            return {"status": "empty"}

        return {
            "buffer_size": len(self.memory_buffer),
            "max_buffer_size": self.buffer_size,
            "time_window_seconds": self.time_window,
            "current_time_span": self._get_time_span(),
            "oldest_entry_age": time.time() - self.memory_buffer[0]["timestamp"] if self.memory_buffer else 0
        }