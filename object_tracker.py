"""
object_tracker.py
-----------------
Object tracking module to maintain persistent object IDs and analyze
object motion across frames.

This implementation uses a lightweight IOU-based tracker that associates
detections across frames using IoU matching, similar to the approach
used in ByteTrack but simplified for standalone use.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    label: str
    bbox: List[int]
    confidence: float
    timestamp: float
    age: int = 0
    hits: int = 1
    motion_history: List[Tuple[float, List[int]]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_bbox: List[int] = field(default_factory=list)
    distance: float = 10.0

    def to_dict(self) -> Dict:
        """Convert track to dictionary format."""
        return {
            "track_id": self.track_id,
            "label": self.label,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "age": self.age,
            "hits": self.hits,
            "velocity": self.velocity,
            "distance": self.distance,
        }


class ObjectTracker:
    """
    IOU-based multi-object tracker.

    Maintains tracks across frames, updates their states,
    and provides motion analysis for each tracked object.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_velocity_change: float = 100.0,
    ):
        """
        Parameters
        ----------
        max_age : int
            Maximum frames to keep a track without detection before removal
        min_hits : int
            Minimum detections before track is considered confirmed
        iou_threshold : float
            IoU threshold for matching detections to tracks
        max_velocity_change : float
            Maximum allowed velocity change between frames
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_velocity_change = max_velocity_change

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0

    def update(self, detections: List[Dict], timestamp: Optional[float] = None) -> List[Dict]:
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : List[Dict]
            List of detection dicts with 'bbox', 'label', 'confidence'
        timestamp : float, optional
            Current timestamp (defaults to time.time())

        Returns
        -------
        List of active tracks as dicts
        """
        if timestamp is None:
            timestamp = time.time()

        self.frame_count += 1

        if not detections:
            self._age_tracks()
            return self._get_active_tracks()

        detection_boxes = [np.array(d["bbox"]) for d in detections]
        detection_labels = [d.get("label", "unknown") for d in detections]
        detection_confs = [d.get("confidence", 0.0) for d in detections]

        if not self.tracks:
            for det in detections:
                self._create_new_track(det, timestamp)
            return self._get_active_tracks()

        iou_matrix = self._compute_iou_matrix(detection_boxes)

        matched, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(
            iou_matrix, len(detections)
        )

        for det_idx, track_id in matched:
            self._update_track(
                track_id,
                detections[det_idx],
                detection_boxes[det_idx],
                timestamp,
            )

        for det_idx in unmatched_dets:
            self._create_new_track(detections[det_idx], timestamp)

        for track_id in unmatched_tracks:
            self.tracks[track_id].age += 1

        self._age_tracks()

        return self._get_active_tracks()

    def _compute_iou_matrix(self, boxes: List[np.ndarray]) -> np.ndarray:
        """Compute IoU matrix between all boxes."""
        n = len(boxes)
        iou_matrix = np.zeros((n, len(self.tracks)))

        for d_idx, box in enumerate(boxes):
            for t_idx, track in enumerate(self.tracks.values()):
                track_box = np.array(track.bbox)
                iou = self._compute_iou(box, track_box)
                iou_matrix[d_idx, t_idx] = iou

        return iou_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _match_detections_to_tracks(
        self,
        iou_matrix: np.ndarray,
        num_detections: int,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using greedy matching."""
        matched = []
        unmatched_detections = list(range(num_detections))
        unmatched_tracks = list(self.tracks.keys())

        if iou_matrix.size == 0:
            return matched, unmatched_detections, unmatched_tracks

        track_indices = list(self.tracks.keys())

        for _ in range(min(num_detections, len(self.tracks))):
            if not unmatched_detections or not unmatched_tracks:
                break

            max_iou = -1
            best_det_idx = -1
            best_track_idx = -1

            for d_idx in unmatched_detections:
                for t_idx, track_id in enumerate(track_indices):
                    if track_id not in unmatched_tracks:
                        continue

                    t_idx_global = list(self.tracks.keys()).index(track_id)
                    if d_idx < iou_matrix.shape[0] and t_idx_global < iou_matrix.shape[1]:
                        iou_val = iou_matrix[d_idx, t_idx_global]
                        if iou_val > max_iou:
                            max_iou = iou_val
                            best_det_idx = d_idx
                            best_track_idx = track_id

            if max_iou >= self.iou_threshold:
                matched.append((best_det_idx, best_track_idx))
                unmatched_detections.remove(best_det_idx)
                unmatched_tracks.remove(best_track_idx)
            else:
                break

        return matched, unmatched_detections, unmatched_tracks

    def _create_new_track(self, detection: Dict, timestamp: float):
        """Create a new track from a detection."""
        track_id = self.next_track_id
        self.next_track_id += 1

        track = Track(
            track_id=track_id,
            label=detection.get("label", "unknown"),
            bbox=detection["bbox"],
            confidence=detection.get("confidence", 0.0),
            timestamp=timestamp,
            last_bbox=detection["bbox"].copy(),
            distance=float(detection.get("distance", 10.0)),
        )

        self.tracks[track_id] = track

    def _update_track(
        self,
        track_id: int,
        detection: Dict,
        box: np.ndarray,
        timestamp: float,
    ):
        """Update an existing track with new detection."""
        track = self.tracks[track_id]

        old_bbox = np.array(track.bbox)
        new_bbox = box

        dx = (new_bbox[0] + new_bbox[2]) / 2 - (old_bbox[0] + old_bbox[2]) / 2
        dy = (new_bbox[1] + new_bbox[3]) / 2 - (old_bbox[1] + old_bbox[3]) / 2

        if track.last_bbox:
            old_center_x = (track.last_bbox[0] + track.last_bbox[2]) / 2
            old_center_y = (track.last_bbox[1] + track.last_bbox[3]) / 2
            new_center_x = (new_bbox[0] + new_bbox[2]) / 2
            new_center_y = (new_bbox[1] + new_bbox[3]) / 2
            vx = new_center_x - old_center_x
            vy = new_center_y - old_center_y
            track.velocity = (
                track.velocity[0] * 0.7 + vx * 0.3,
                track.velocity[1] * 0.7 + vy * 0.3,
            )

        distance = np.sqrt(dx**2 + dy**2)
        track.motion_history.append((timestamp, list(new_bbox)))

        if len(track.motion_history) > 30:
            track.motion_history = track.motion_history[-30:]

        track.bbox = detection["bbox"].copy()
        track.last_bbox = old_bbox.tolist()
        track.label = detection.get("label", track.label)
        track.confidence = detection.get("confidence", track.confidence)
        track.timestamp = timestamp
        track.age = 0
        track.hits += 1
        track.distance = float(detection.get("distance", track.distance))

    def _age_tracks(self):
        """Remove old tracks that have exceeded max age."""
        to_remove = [
            track_id
            for track_id, track in self.tracks.items()
            if track.age > self.max_age
        ]
        for track_id in to_remove:
            del self.tracks[track_id]

    def _get_active_tracks(self) -> List[Dict]:
        """Get list of active (confirmed) tracks."""
        active = []
        for track in self.tracks.values():
            if track.hits >= self.min_hits:
                active.append(track.to_dict())
        return active

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self.tracks.get(track_id)

    def get_tracks_by_label(self, label: str) -> List[Track]:
        """Get all tracks with a specific label."""
        return [t for t in self.tracks.values() if t.label == label and t.hits >= self.min_hits]

    def reset(self):
        """Reset the tracker state."""
        self.tracks = {}
        self.next_track_id = 1
        self.frame_count = 0


def update_tracker(tracker: ObjectTracker, detections: List[Dict]) -> List[Dict]:
    """Simple functional interface for updating tracker."""
    return tracker.update(detections)


def get_track_motion(track: Track) -> Dict:
    """Get motion information for a track."""
    if len(track.motion_history) < 2:
        return {
            "velocity": track.velocity,
            "speed": 0.0,
            "direction": "stationary",
            "is_approaching": False,
        }

    vx, vy = track.velocity
    speed = np.sqrt(vx**2 + vy**2)

    direction = "stationary"
    if abs(vy) > abs(vx):
        direction = "approaching" if vy > 0 else "receding"
    elif abs(vx) > abs(vy):
        direction = "moving_left" if vx < 0 else "moving_right"

    is_approaching = vy > 0

    return {
        "velocity": (vx, vy),
        "speed": speed,
        "direction": direction,
        "is_approaching": is_approaching,
    }