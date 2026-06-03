"""
roi_mask.py
-----------
Region-of-Interest (ROI) masking to ignore the lower dashboard/hood
region and prevent self-car detections.

The ROI is defined as a trapezoid in the lower portion of the frame
where the vehicle's own hood/dashboard would appear in a forward-facing
dashcam view.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class ROIMask:
    """
    Creates and applies a region-of-interest mask to filter out
    detections in the dashboard/hood area of a forward-facing dashcam.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        roi_top_ratio: float = 0.55,
        roi_bottom_ratio: float = 0.80,
        roi_left_ratio: float = 0.0,
        roi_right_ratio: float = 1.0,
    ):
        """
        Parameters
        ----------
        frame_width : int
            Frame width in pixels
        frame_height : int
            Frame height in pixels
        roi_top_ratio : float
            Top edge of ROI as ratio of frame height (0.55 = 55% from top)
        roi_bottom_ratio : float
            Bottom edge of ROI as ratio of frame height
        roi_left_ratio : float
            Left edge of ROI as ratio of frame width
        roi_right_ratio : float
            Right edge of ROI as ratio of frame width
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        self.roi_left_ratio = roi_left_ratio
        self.roi_right_ratio = roi_right_ratio
        self._mask = self._create_trapezoid_mask()

    def _create_trapezoid_mask(self) -> np.ndarray:
        """
        Create a trapezoid mask that defines the valid ROI.
        The mask is white (255) in the valid region and black (0) in
        the excluded (hood/dashboard) region.
        """
        h, w = self.frame_height, self.frame_width

        top_y = int(h * self.roi_top_ratio)
        bottom_y = int(h * self.roi_bottom_ratio)

        top_left_x = int(w * self.roi_left_ratio)
        top_right_x = int(w * self.roi_right_ratio)
        bottom_left_x = int(w * 0.05)
        bottom_right_x = int(w * 0.95)

        vertices = np.array([
            [top_left_x, top_y],
            [top_right_x, top_y],
            [bottom_right_x, bottom_y],
            [bottom_left_x, bottom_y],
        ], dtype=np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

    def update_frame_size(self, width: int, height: int):
        """Update frame dimensions and recalculate mask."""
        self.frame_width = width
        self.frame_height = height
        self._mask = self._create_trapezoid_mask()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply ROI mask to frame, returning masked frame."""
        if frame.shape[:2][::-1] != (self.frame_width, self.frame_height):
            self.update_frame_size(frame.shape[1], frame.shape[0])
        masked = cv2.bitwise_and(frame, frame, mask=self._mask)
        return masked

    def filter_detections(self, detections: list) -> list:
        """
        Filter out detections that fall outside the ROI.

        A detection is kept if its bounding box center is within
        the valid ROI region.

        Parameters
        ----------
        detections : list
            List of detection dicts with 'bbox' key [x1, y1, x2, y2]

        Returns
        -------
        Filtered detections list
        """
        filtered = []
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if self._mask[center_y, center_x] > 0:
                filtered.append(det)

        return filtered

    def is_in_roi(self, x: int, y: int) -> bool:
        """Check if a point (x, y) is within the valid ROI."""
        if 0 <= y < self.frame_height and 0 <= x < self.frame_width:
            return bool(self._mask[y, x])
        return False

    def get_roi_polygon(self) -> list:
        """Return the ROI polygon vertices for visualization."""
        h, w = self.frame_height, self.frame_width
        top_y = int(h * self.roi_top_ratio)
        bottom_y = int(h * self.roi_bottom_ratio)
        top_left_x = int(w * self.roi_left_ratio)
        top_right_x = int(w * self.roi_right_ratio)
        bottom_left_x = int(w * 0.05)
        bottom_right_x = int(w * 0.95)
        return [
            (top_left_x, top_y),
            (top_right_x, top_y),
            (bottom_right_x, bottom_y),
            (bottom_left_x, bottom_y),
        ]

    def get_roi_overlay(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), alpha: float = 0.3) -> np.ndarray:
        """Create overlay showing the ROI region on the frame."""
        overlay = frame.copy()
        polygon = self.get_roi_polygon()
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def apply_roi_to_frame(frame: np.ndarray, top_ratio: float = 0.55, bottom_ratio: float = 0.95) -> np.ndarray:
    """
    Simple functional interface for applying ROI mask to a frame.

    Parameters
    ----------
    frame : np.ndarray
        Input frame
    top_ratio : float
        Top edge of ROI as ratio of frame height
    bottom_ratio : float
        Bottom edge of ROI as ratio of frame height

    Returns
    -------
    Masked frame
    """
    h, w = frame.shape[:2]
    roi = ROIMask(w, h, top_ratio, bottom_ratio)
    return roi.apply(frame)


def filter_detections_by_roi(detections: list, frame_shape: tuple) -> list:
    """
    Simple functional interface for filtering detections by ROI.

    Parameters
    ----------
    detections : list
        List of detection dicts
    frame_shape : tuple
        (height, width) of the frame

    Returns
    -------
    Filtered detections
    """
    h, w = frame_shape[:2]
    roi = ROIMask(w, h)
    return roi.filter_detections(detections)