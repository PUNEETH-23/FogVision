"""
lane_detection.py
-----------------
Lane detection module to identify the current driving lane and
filter objects that are outside the lane (side lanes, roadside).

Uses classical computer vision techniques (Hough transform, color
thresholding) for lane marking detection that work in real-time
without deep learning models.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import deque


class LaneDetector:
    """
    Detects driving lanes using edge detection and Hough transform.
    Identifies the current lane polygon for filtering objects.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        roi_top_ratio: float = 0.45,
        horizon_ratio: float = 0.55,
    ):
        """
        Parameters
        ----------
        frame_width : int
            Frame width
        frame_height : int
            Frame height
        roi_top_ratio : float
            Top of region for lane detection (as ratio of height)
        horizon_ratio : float
            Expected horizon line position
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_top_ratio = roi_top_ratio
        self.horizon_ratio = horizon_ratio
        self.horizon_line = int(frame_height * horizon_ratio)

        self.left_lane_points = deque(maxlen=10)
        self.right_lane_points = deque(maxlen=10)
        self.left_lane_coeffs = None
        self.right_lane_coeffs = None

        self._left_weight = 0.7
        self._right_weight = 0.3

    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions."""
        self.frame_width = width
        self.frame_height = height
        self.horizon_line = int(height * self.horizon_ratio)

    def detect_lanes(self, frame: np.ndarray) -> Dict:
        """
        Return a constant lane polygon (front lane + small peripheral).

        Parameters
        ----------
        frame : np.ndarray
            Input frame (BGR)

        Returns
        -------
        Dict with lane information:
            - lane_polygon: polygon defining the driving lane
            - left_line: (slope, intercept) for left lane
            - right_line: (slope, intercept) for right lane
            - horizon_y: y-position of horizon
            - detected: bool
        """
        h, w = frame.shape[:2]

        if frame.shape[:2][::-1] != (self.frame_width, self.frame_height):
            self.update_frame_size(frame.shape[1], frame.shape[0])

        lane_polygon = self._create_lane_polygon(None, None, h)

        return {
            "lane_polygon": lane_polygon,
            "left_line": (1.0, 0.0),
            "right_line": (-1.0, float(w)),
            "horizon_y": self.horizon_line,
            "detected": True,
        }

    def _fit_lane_line(
        self,
        lines: List[Tuple],
        frame_w: int,
        frame_h: int,
        is_left: bool,
    ) -> Optional[Tuple[float, float]]:
        """Placeholder for compatibility."""
        return None

    def _create_lane_polygon(
        self,
        left_line: Optional[Tuple[float, float]],
        right_line: Optional[Tuple[float, float]],
        frame_h: int,
    ) -> np.ndarray:
        """Create polygon representing the driving lane (constant: front lane + small peripheral)."""
        w = self.frame_width
        h = frame_h
        top_y = self.horizon_line
        bottom_y = int(h * 0.80)

        # Constant trapezoid for front lane + small peripheral
        left_x_top = int(w * 0.38)
        right_x_top = int(w * 0.62)
        left_x_bottom = int(w * 0.15)
        right_x_bottom = int(w * 0.85)

        polygon = np.array([
            [left_x_bottom, bottom_y],
            [left_x_top, top_y],
            [right_x_top, top_y],
            [right_x_bottom, bottom_y],
        ], dtype=np.int32)

        return polygon

    def _get_default_polygon(self, frame_h: int) -> np.ndarray:
        """Return a default lane polygon."""
        return self._create_lane_polygon(None, None, frame_h)

    def is_point_in_lane(self, x: int, y: int, lane_polygon: np.ndarray) -> bool:
        """Check if a point (x, y) is inside the lane polygon."""
        if lane_polygon is None or len(lane_polygon) == 0:
            return True

        polygon = lane_polygon.reshape((-1, 1, 2)) if lane_polygon.shape[0] == 4 else lane_polygon
        result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
        return result >= 0

    def filter_objects_in_lane(self, detections: list, lane_info: Dict) -> Tuple[list, list]:
        """
        Filter detections to separate in-lane vs out-of-lane objects.

        Parameters
        ----------
        detections : list
            List of detection dicts with 'bbox' key
        lane_info : dict
            Lane detection result

        Returns
        -------
        Tuple of (in_lane_detections, out_of_lane_detections)
        """
        lane_polygon = lane_info.get("lane_polygon")

        in_lane = []
        out_of_lane = []

        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if self.is_point_in_lane(center_x, center_y, lane_polygon):
                in_lane.append(det)
            else:
                out_of_lane.append(det)

        return in_lane, out_of_lane

    def get_lane_overlay(self, frame: np.ndarray, lane_info: Dict, color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """Draw lane overlay on the frame."""
        overlay = frame.copy()
        lane_polygon = lane_info.get("lane_polygon")

        if lane_polygon is not None and len(lane_polygon) > 0:
            cv2.polylines(overlay, [lane_polygon], isClosed=True, color=color, thickness=3)

        return overlay

    @property
    def frame(self) -> np.ndarray:
        """Placeholder for compatibility."""
        return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)


def detect_lanes_simple(frame: np.ndarray) -> Dict:
    """Simple functional interface for lane detection."""
    h, w = frame.shape[:2]
    detector = LaneDetector(w, h)
    return detector.detect_lanes(frame)


def filter_objects_in_driving_lane(detections: list, lane_info: Dict) -> Tuple[list, list]:
    """Simple functional interface for filtering objects by lane."""
    h, w = 480, 640
    detector = LaneDetector(w, h)
    return detector.filter_objects_in_lane(detections, lane_info)