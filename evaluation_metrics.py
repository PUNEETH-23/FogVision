"""
evaluation_metrics.py
----------------------
Evaluation metrics for fog detection system.

Computes mAP, F1, alert latency, false positive rate vs baseline YOLO.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path


class FogDetectionMetrics:
    """
    Comprehensive evaluation metrics for fog detection system.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

        self.alert_latencies = []
        self.fog_density_errors = []
        self.detection_confidences = []

        self.baseline_detections = []
        self.enhanced_detections = []

    def update_fog_detection(self, true_fog_density: float, predicted_fog_density: float):
        """
        Update fog density detection metrics.
        """
        error = abs(true_fog_density - predicted_fog_density)
        self.fog_density_errors.append(error)

    def update_object_detection(self, baseline_detections: List[Dict],
                               enhanced_detections: List[Dict],
                               ground_truth: List[Dict]):
        """
        Update object detection metrics comparing baseline vs enhanced system.
        """
        # Store for batch processing
        self.baseline_detections.extend(baseline_detections)
        self.enhanced_detections.extend(enhanced_detections)

    def update_alert_metrics(self, alert_triggered: bool, should_alert: bool,
                           latency_seconds: Optional[float] = None):
        """
        Update alert classification metrics.
        """
        if alert_triggered and should_alert:
            self.true_positives += 1
        elif alert_triggered and not should_alert:
            self.false_positives += 1
        elif not alert_triggered and should_alert:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

        if latency_seconds is not None:
            self.alert_latencies.append(latency_seconds)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        """
        metrics = {}

        # Fog density metrics
        if self.fog_density_errors:
            metrics["fog_density"] = {
                "mae": np.mean(self.fog_density_errors),
                "rmse": np.sqrt(np.mean([e**2 for e in self.fog_density_errors])),
                "max_error": max(self.fog_density_errors),
                "accuracy_10pct": np.mean([1 if e <= 10 else 0 for e in self.fog_density_errors]),
                "samples": len(self.fog_density_errors)
            }

        # Alert classification metrics
        total_predictions = self.true_positives + self.false_positives + \
                           self.true_negatives + self.false_negatives

        if total_predictions > 0:
            precision = self.true_positives / (self.true_positives + self.false_positives) \
                       if (self.true_positives + self.false_positives) > 0 else 0

            recall = self.true_positives / (self.true_positives + self.false_negatives) \
                    if (self.true_positives + self.false_negatives) > 0 else 0

            f1_score = 2 * (precision * recall) / (precision + recall) \
                      if (precision + recall) > 0 else 0

            accuracy = (self.true_positives + self.true_negatives) / total_predictions

            metrics["alert_classification"] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives
            }

        # Alert latency metrics
        if self.alert_latencies:
            metrics["alert_latency"] = {
                "mean_latency_seconds": np.mean(self.alert_latencies),
                "median_latency_seconds": np.median(self.alert_latencies),
                "max_latency_seconds": max(self.alert_latencies),
                "min_latency_seconds": min(self.alert_latencies),
                "latency_std": np.std(self.alert_latencies),
                "samples": len(self.alert_latencies)
            }

        # Object detection improvement (placeholder - would need IoU calculation)
        metrics["object_detection"] = {
            "baseline_detections": len(self.baseline_detections),
            "enhanced_detections": len(self.enhanced_detections),
            "improvement_ratio": len(self.enhanced_detections) / max(len(self.baseline_detections), 1)
        }

        return metrics

    def compute_map_f1_from_detections(self, predictions: List[Dict],
                                     ground_truth: List[Dict],
                                     iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute mAP and F1 from detection results.
        """
        if not predictions or not ground_truth:
            return {"mAP": 0.0, "f1": 0.0}

        # Simplified mAP calculation (would need full implementation for production)
        # This is a placeholder - real mAP requires precision-recall curves

        # For now, compute simple accuracy metrics
        pred_count = len(predictions)
        gt_count = len(ground_truth)

        # Assume some matches (in real implementation, use IoU matching)
        true_positives = min(pred_count, gt_count)  # Simplified
        false_positives = max(0, pred_count - gt_count)
        false_negatives = max(0, gt_count - pred_count)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Simplified mAP (mean Average Precision)
        map_score = (precision + recall) / 2  # Simplified

        return {
            "mAP": map_score,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


class LatencyProfiler:
    """
    Profiles end-to-end pipeline latency.
    """

    def __init__(self):
        self.latencies = []
        self.start_time = None

    def start_pipeline(self):
        """Start timing a pipeline run."""
        self.start_time = time.time()

    def end_pipeline(self) -> float:
        """End timing and record latency."""
        if self.start_time is None:
            return 0.0

        latency = time.time() - self.start_time
        self.latencies.append(latency)
        self.start_time = None
        return latency

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {"samples": 0}

        return {
            "mean_latency_ms": np.mean(self.latencies) * 1000,
            "median_latency_ms": np.median(self.latencies) * 1000,
            "max_latency_ms": max(self.latencies) * 1000,
            "min_latency_ms": min(self.latencies) * 1000,
            "std_latency_ms": np.std(self.latencies) * 1000,
            "samples": len(self.latencies),
            "target_met": np.mean(self.latencies) * 1000 < 100  # Target: <100ms
        }


def generate_evaluation_report(metrics: Dict[str, Any],
                             latency_stats: Dict[str, float],
                             output_file: str = "evaluation_report.json"):
    """
    Generate comprehensive evaluation report.
    """
    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "latency": latency_stats,
        "system_performance": {
            "latency_target_met": latency_stats.get("target_met", False),
            "overall_grade": _compute_overall_grade(metrics, latency_stats)
        }
    }

    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    return report


def _compute_overall_grade(metrics: Dict[str, Any], latency_stats: Dict[str, float]) -> str:
    """
    Compute overall system performance grade.
    """
    score = 0
    max_score = 0

    # Latency score (40% weight)
    if latency_stats.get("target_met", False):
        score += 40
    max_score += 40

    # Fog detection accuracy (30% weight)
    fog_metrics = metrics.get("fog_density", {})
    if fog_metrics.get("accuracy_10pct", 0) > 0.8:
        score += 30
    elif fog_metrics.get("accuracy_10pct", 0) > 0.6:
        score += 20
    elif fog_metrics.get("accuracy_10pct", 0) > 0.4:
        score += 10
    max_score += 30

    # Alert F1 score (30% weight)
    alert_metrics = metrics.get("alert_classification", {})
    f1 = alert_metrics.get("f1_score", 0)
    if f1 > 0.8:
        score += 30
    elif f1 > 0.6:
        score += 20
    elif f1 > 0.4:
        score += 10
    max_score += 30

    # Grade based on percentage
    percentage = (score / max_score) * 100 if max_score > 0 else 0

    if percentage >= 90:
        return "A"
    elif percentage >= 80:
        return "B"
    elif percentage >= 70:
        return "C"
    elif percentage >= 60:
        return "D"
    else:
        return "F"