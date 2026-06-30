"""
fog_aware.py
-----------
Fog-aware preprocessing with dehazing and adaptive confidence thresholds
for reliable detection under foggy conditions.
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Append PyFADE src path to import pyfade
_pyfade_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyFADE", "src"))
if _pyfade_src_path not in sys.path:
    sys.path.insert(0, _pyfade_src_path)

from pyfade import fade


@dataclass
class FogCondition:
    """Represents the current fog condition."""
    density: float
    visibility: str
    recommended_speed: int
    confidence_threshold: float
    detection_multiplier: float


class FogAwarePreprocessor:
    """
    Fog-aware preprocessing that adapts detection parameters based on
    estimated fog density.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
    ):
        """
        Parameters
        ----------
        frame_width : int
            Frame width
        frame_height : int
            Frame height
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.base_confidence_threshold = 0.4
        self.fog_history = []
        self.history_size = 5

        self._current_condition = FogCondition(
            density=0.0,
            visibility="Good",
            recommended_speed=70,
            confidence_threshold=self.base_confidence_threshold,
            detection_multiplier=1.0,
        )

    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions."""
        self.frame_width = width
        self.frame_height = height

    def set_condition_density(self, density: float) -> FogCondition:
        """Manually set and calculate a static fog condition."""
        visibility, recommended_speed = self._get_visibility_info(density)
        threshold_multiplier = self._get_threshold_multiplier(density)
        confidence_threshold = self.base_confidence_threshold * threshold_multiplier
        detection_multiplier = self._get_detection_multiplier(density)
        
        self._current_condition = FogCondition(
            density=round(density, 1),
            visibility=visibility,
            recommended_speed=recommended_speed,
            confidence_threshold=round(confidence_threshold, 2),
            detection_multiplier=round(detection_multiplier, 2),
        )
        self.fog_history = [density]
        return self._current_condition

    def analyze_fog(self, frame: np.ndarray) -> FogCondition:
        """
        Analyze fog conditions in the frame.

        Parameters
        ----------
        frame : np.ndarray
            Input frame (BGR)

        Returns
        -------
        FogCondition with analysis results
        """
        try:
            fade_score = fade(frame)
            fog_density = ((fade_score - 0.3) / 2.7) * 100.0
            fog_density = max(0.0, min(100.0, fog_density))
        except Exception as e:
            print(f"[PyFADE] Error in analyze_fog: {e}")
            # fallback
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            brightness_score = (255 - mean_brightness) / 255
            contrast_score = std_brightness / 128
            fog_density = (brightness_score * 0.6 + (1 - contrast_score) * 0.4) * 100

        self.fog_history.append(fog_density)
        if len(self.fog_history) > self.history_size:
            self.fog_history.pop(0)

        smoothed_density = np.mean(self.fog_history) if self.fog_history else fog_density

        visibility, recommended_speed = self._get_visibility_info(smoothed_density)

        threshold_multiplier = self._get_threshold_multiplier(smoothed_density)
        confidence_threshold = self.base_confidence_threshold * threshold_multiplier

        detection_multiplier = self._get_detection_multiplier(smoothed_density)

        self._current_condition = FogCondition(
            density=round(smoothed_density, 1),
            visibility=visibility,
            recommended_speed=recommended_speed,
            confidence_threshold=round(confidence_threshold, 2),
            detection_multiplier=round(detection_multiplier, 2),
        )

        return self._current_condition

    def _get_visibility_info(self, density: float) -> Tuple[str, int]:
        """Get visibility rating and recommended speed."""
        if density >= 80:
            return "Very Low", 30
        elif density >= 60:
            return "Low", 40
        elif density >= 40:
            return "Moderate", 50
        elif density >= 20:
            return "Good", 60
        else:
            return "Excellent", 70

    def _get_threshold_multiplier(self, density: float) -> float:
        """Increase confidence threshold in fog for fewer false positives."""
        if density >= 70:
            return 1.5
        elif density >= 50:
            return 1.3
        elif density >= 30:
            return 1.15
        else:
            return 1.0

    def _get_detection_multiplier(self, density: float) -> float:
        """Adjust detection parameters based on density."""
        if density >= 60:
            return 0.8
        elif density >= 40:
            return 0.9
        else:
            return 1.0

    def apply_adaptive_detection(
        self,
        detections: list,
        frame: np.ndarray,
    ) -> list:
        """
        Apply adaptive confidence thresholds to detections.

        Parameters
        ----------
        detections : list
            Raw detections from YOLO
        frame : np.ndarray
            Current frame for fog analysis

        Returns
        -------
        Filtered detections based on fog conditions
        """
        condition = self.analyze_fog(frame)
        threshold = condition.confidence_threshold

        filtered = []
        for det in detections:
            conf = det.get("confidence", 0.0)

            adjusted_conf = conf * condition.detection_multiplier

            if adjusted_conf >= threshold:
                filtered.append({
                    **det,
                    "confidence": round(adjusted_conf, 2),
                    "fog_adjusted": True,
                    "original_confidence": round(conf, 2),
                })

        return filtered

    def get_adaptive_nms_params(self) -> Dict:
        """Get adaptive NMS parameters based on fog."""
        density = self._current_condition.density

        if density >= 60:
            iou_threshold = 0.4
            score_threshold = 0.3
        elif density >= 40:
            iou_threshold = 0.45
            score_threshold = 0.35
        else:
            iou_threshold = 0.5
            score_threshold = 0.4

        return {
            "iou_threshold": iou_threshold,
            "score_threshold": score_threshold,
        }

    def get_fog_statistics(self) -> Dict:
        """Get fog analysis statistics."""
        if not self.fog_history:
            return {
                "current_density": 0.0,
                "trend": "stable",
            }

        recent = self.fog_history[-3:] if len(self.fog_history) >= 3 else self.fog_history

        trend = "stable"
        if len(self.fog_history) >= 3:
            if np.mean(self.fog_history[-3:]) > np.mean(self.fog_history[-5:-2]):
                trend = "increasing"
            else:
                trend = "decreasing"

        return {
            "current_density": self._current_condition.density,
            "history": self.fog_history.copy(),
            "trend": trend,
            "visibility": self._current_condition.visibility,
            "recommended_speed": self._current_condition.recommended_speed,
        }

    def apply_dehaze(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply simple dehazing to improve visibility in fog.

        Parameters
        ----------
        frame : np.ndarray
            Input frame (BGR)

        Returns
        -------
        Dehazed frame
        """
        fog_density = self._current_condition.density

        if fog_density < 20:
            return frame

        fog_factor = min(fog_density / 100, 0.7)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = clahe.apply(l)
        enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        dehazed = cv2.addWeighted(
            frame, 1 - fog_factor,
            enhanced, fog_factor,
            0
        )

        return dehazed

    @property
    def current_condition(self) -> FogCondition:
        """Get current fog condition."""
        return self._current_condition


def get_adaptive_threshold(fog_density: float) -> float:
    """Simple functional interface for adaptive threshold."""
    preprocessor = FogAwarePreprocessor()
    condition = preprocessor.analyze_fog(np.zeros((480, 640, 3), dtype=np.uint8))
    return condition.confidence_threshold


def filter_detections_by_fog(detections: list, fog_density: float) -> list:
    """Simple functional interface for fog-based filtering."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    preprocessor = FogAwarePreprocessor()
    return preprocessor.apply_adaptive_detection(detections, frame)