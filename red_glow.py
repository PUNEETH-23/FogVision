"""
red_glow.py
-----------
Module 5 of the LLD: Red Glow / Brake-Light Detection.

Detects red-glow regions in HSV colour space, filters noise,
and associates red-glow events with nearby detected objects
(cars, trucks, buses) from the YOLO detection pass.

Returns:
    red_glow  : bool   — True when a confirmed red-glow source is found
    glow_boxes: list   — bounding boxes of detected glow regions [[x1,y1,x2,y2], …]
    glow_frame: ndarray— annotated copy of the input frame
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


# ------------------------------------------------------------------
# HSV ranges for red (two hue-wraps needed in OpenCV)
# ------------------------------------------------------------------
_RED_LO1 = np.array([0,   120,  70], dtype=np.uint8)
_RED_HI1 = np.array([10,  255, 255], dtype=np.uint8)
_RED_LO2 = np.array([160, 120,  70], dtype=np.uint8)
_RED_HI2 = np.array([180, 255, 255], dtype=np.uint8)

# Enhanced blob detection parameters
_MIN_AREA: int = 300      # Minimum contour area
_MAX_AREA: int = 50000    # Maximum contour area (avoid huge false positives)
_MIN_ASPECT_RATIO: float = 0.2  # Minimum width/height ratio
_MAX_ASPECT_RATIO: float = 5.0  # Maximum width/height ratio
_MIN_SOLIDITY: float = 0.3      # Minimum solidity (area/convex_hull_area)

# IoU overlap threshold to associate a glow blob with a YOLO box
_IOU_THRESH: float = 0.05

# Vehicle labels that can carry brake lights
_VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle", "motorbike"}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def detect_red_glow(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    frame      : BGR frame (already dehazed)
    detections : list of dicts from object_detect.process_frame()

    Returns
    -------
    {
        "red_glow"   : bool,
        "glow_boxes" : [[x1,y1,x2,y2], …],
        "glow_frame" : annotated BGR ndarray,
        "glow_count" : int,
    }
    """
    annotated = frame.copy()

    # --- Build HSV red mask -------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, _RED_LO1, _RED_HI1)
    mask2 = cv2.inRange(hsv, _RED_LO2, _RED_HI2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological cleanup: remove specks, fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- Find contours -----------------------------------------------
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    glow_boxes: List[List[int]] = []
    glow_properties: List[Dict[str, Any]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_AREA or area > _MAX_AREA:
            continue

        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)

        # Aspect ratio check
        aspect_ratio = w / h if h > 0 else 0
        if not (_MIN_ASPECT_RATIO <= aspect_ratio <= _MAX_ASPECT_RATIO):
            continue

        # Solidity check (area / convex hull area)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < _MIN_SOLIDITY:
            continue

        # Additional HSV validation - check if blob is predominantly red
        roi_hsv = hsv[y:y+h, x:x+w]
        if roi_hsv.size > 0:
            red_pixels = cv2.countNonZero(cv2.inRange(roi_hsv, _RED_LO1, _RED_HI1)) + \
                        cv2.countNonZero(cv2.inRange(roi_hsv, _RED_LO2, _RED_HI2))
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
            if red_ratio < 0.4:  # At least 40% of blob should be red
                continue

        glow_boxes.append([x, y, x + w, y + h])
        glow_properties.append({
            "area": area,
            "aspect_ratio": round(aspect_ratio, 2),
            "solidity": round(solidity, 2),
            "red_ratio": round(red_ratio, 2),
            "bbox": [x, y, x + w, y + h]
        })

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            annotated, f"Red Glow (A:{int(area)})",
            (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
        )

    # --- Associate with vehicle detections ---------------------------
    red_glow = False
    if glow_boxes:
        vehicle_boxes = [
            d["bbox"]
            for d in detections
            if d.get("label", "").lower() in _VEHICLE_LABELS
        ]

        if vehicle_boxes:
            # Confirmed only when a glow overlaps a known vehicle
            for gb in glow_boxes:
                for vb in vehicle_boxes:
                    if _iou(gb, vb) > _IOU_THRESH:
                        red_glow = True
                        break
                if red_glow:
                    break
        else:
            # No vehicle detections at all → accept glow on its own
            # (handles cases where the vehicle box was missed)
            red_glow = len(glow_boxes) > 0

    # Overlay status banner
    banner_color = (0, 0, 200) if red_glow else (0, 180, 0)
    status_text  = "Red Glow Detected: TRUE" if red_glow else "Red Glow Detected: FALSE"
    cv2.rectangle(annotated, (0, 0), (340, 32), banner_color, -1)
    cv2.putText(
        annotated, status_text,
        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
    )

    return {
        "red_glow":   red_glow,
        "glow_boxes": glow_boxes,
        "glow_frame": annotated,
        "glow_count": len(glow_boxes),
        "glow_properties": glow_properties,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _iou(
    box_a: List[int],
    box_b: List[int],
) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0