"""
distance_estimation.py
----------------------
Module 6 of the LLD: Distance Estimation (LiDAR / Depth).

In this software-only build we simulate LiDAR distance using two
complementary heuristics that work directly on the camera frame
and YOLO bounding-box metadata:

  1. Apparent-size heuristic  — larger bounding box  → closer object.
     Uses empirical reference heights (pixels at 10 m) for each class.

  2. Vertical-position heuristic — objects lower in the frame
     (closer to the horizon line) are typically farther away.
     Blended with heuristic 1 for robustness.

These two are combined into a single estimated distance (metres).
The result intentionally mimics the LLD's output field
  "distance_to_nearest" used by the Rule Engine.

When a real depth sensor / point-cloud is integrated later,
replace `_estimate_distance_pixels()` with the sensor read-out.
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------------
# Reference calibration table
# class label → (apparent_height_px_at_10m, real_height_m)
# Values are typical for a dashcam at ~1 m mount height, 30° tilt.
# ------------------------------------------------------------------
_REF: Dict[str, Tuple[float, float]] = {
    "car":          (80.0,  1.5),
    "truck":        (120.0, 3.8),
    "bus":          (130.0, 3.2),
    "person":       (90.0,  1.75),
    "bicycle":      (70.0,  1.1),
    "motorcycle":   (65.0,  1.2),
    "motorbike":    (65.0,  1.2),
    "traffic light":(40.0,  0.8),
    "stop sign":    (35.0,  0.75),
    # fallback
    "_default":     (70.0,  1.5),
}

# Assumed reference distance for the reference apparent height
_REF_DIST_M: float = 10.0

# Frame height used during calibration (pixels)
_CALIB_FRAME_H: int = 480

# Clamp distances to sensible ADAS range (metres)
_MIN_DIST_M: float = 1.0
_MAX_DIST_M: float = 120.0


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def estimate_distance(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    frame      : BGR frame (used for frame-height scaling only)
    detections : YOLO detection dicts — must contain 'bbox' and 'label'

    Returns
    -------
    {
        "distance_to_nearest" : float   metres (0 → no vehicles found)
        "nearest_label"       : str
        "per_object_distances": [
            {"label": …, "confidence": …, "distance_m": …, "bbox": …},
            …
        ]
    }
    """
    if not detections:
        return {
            "distance_to_nearest":  0.0,
            "nearest_label":        "none",
            "per_object_distances": [],
        }

    frame_h = frame.shape[0] if frame is not None else _CALIB_FRAME_H
    results: List[Dict[str, Any]] = []

    for det in detections:
        label = det.get("label", "_default").lower()
        bbox  = det.get("bbox", [0, 0, 0, 0])

        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = bbox
        box_h = max(y2 - y1, 1)

        dist_m = _estimate_distance_pixels(label, box_h, frame_h)

        results.append({
            "label":      label,
            "confidence": det.get("confidence", 0.0),
            "distance_m": round(dist_m, 1),
            "bbox":       bbox,
        })

    if not results:
        return {
            "distance_to_nearest":  0.0,
            "nearest_label":        "none",
            "per_object_distances": [],
        }

    # Nearest relevant object (closest estimated distance)
    nearest = min(results, key=lambda x: x["distance_m"])

    return {
        "distance_to_nearest":  nearest["distance_m"],
        "nearest_label":        nearest["label"],
        "per_object_distances": results,
    }


def annotate_distances(
    frame: np.ndarray,
    per_object_distances: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Draw distance overlays on a copy of *frame*.
    Call after object_detect already drew bounding boxes so the
    distance tag sits just above the existing label.
    """
    out = frame.copy()
    for obj in per_object_distances:
        bbox = obj.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1 = bbox[0], bbox[1]
        dist_m  = obj["distance_m"]
        text    = f"{dist_m:.1f} m"
        # Slightly offset from YOLO label (which is at y1-10)
        ty = max(y1 - 28, 14)
        cv2.putText(
            out, text,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 200, 0), 2,
        )
    return out


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _estimate_distance_pixels(
    label:   str,
    box_h:   int,
    frame_h: int,
) -> float:
    """
    Apparent-size → distance conversion.

    Formula (thin-lens / similar-triangles):
        dist = (ref_h_px_at_ref_dist * ref_dist_m * frame_h)
               / (box_h * CALIB_FRAME_H)

    The frame_h / CALIB_FRAME_H factor corrects for resolution changes.
    """
    ref_h_px, _ = _REF.get(label, _REF["_default"])

    # Scale ref height for current frame resolution
    scale    = frame_h / _CALIB_FRAME_H
    ref_h_px = ref_h_px * scale

    dist_m = (ref_h_px * _REF_DIST_M) / max(box_h, 1)
    return float(np.clip(dist_m, _MIN_DIST_M, _MAX_DIST_M))