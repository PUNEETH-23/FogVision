"""
Fog Impact on Object Detection Testing Framework
-------------------------------------------------
Tests how fog levels affect YOLO object detection accuracy.
Tests fog density estimation using fog_density.py (DCP) and Density.py (PyFADE).
Generates synthetic fog at different levels and compares inference results.
Includes average time metric for video processing.
"""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from object_detect import process_frame
from synthetic_fog_dataset import FogAugmentation

# Import fog density estimation modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from fog_density import estimate_fog_density as dcp_estimate_fog_density
from Density import get_fog_density as pyfade_estimate_fog_density


class FogImpactTester:
    """
    Tests object detection performance under different fog conditions.
    Also tests fog density estimation using DCP and PyFADE methods.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Parameters
        ----------
        output_dir : str
            Directory to save results and charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.fog_augmenter = FogAugmentation()
        self.fog_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 6 levels from clear to very dense
        self._reset_metrics()

    def _reset_metrics(self):
        """Reset metrics so process_video() can be called multiple times cleanly."""
        self.metrics = {
            "fog_level": [],
            "avg_confidence": [],
            "detection_count": [],
            "objects_per_frame": [],
            "confidence_std": [],
            "avg_time_per_frame": [],  # Average processing time per frame
            "dcp_fog_density": [],      # Estimated fog density (DCP method)
            "pyfade_fog_density": [],    # Estimated fog density (PyFADE method)
            "dcp_avg_time": [],          # Average time for DCP estimation
            "pyfade_avg_time": []        # Average time for PyFADE estimation
        }
        # Per-frame detailed metrics
        self.per_frame_metrics = []  # List of dicts: {frame_idx, fog_level, confidence, detection_count, time, dcp_est, pyfade_est}
        self.sample_frames = {}  # Store sample foggy frames for composite
        # Timing tracking variables
        self._current_dcp_time = 0.0
        self._current_pyfade_time = 0.0

    def _estimate_fog_dcp(self, frame: np.ndarray) -> float:
        """Estimate fog density using Dark Channel Prior method."""
        try:
            start_time = time.time()
            result = dcp_estimate_fog_density(frame)
            end_time = time.time()
            self._current_dcp_time = end_time - start_time
            return result.get("fog_density", 0.0)
        except Exception as e:
            print(f"  DCP estimation error: {e}")
            self._current_dcp_time = 0.0
            return 0.0

    def _estimate_fog_pyfade(self, frame: np.ndarray) -> float:
        """Estimate fog density using PyFADE method."""
        try:
            # PyFADE requires image path, save temporarily
            temp_path = self.output_dir / "temp_frame.jpg"
            cv2.imwrite(str(temp_path), frame)
            start_time = time.time()
            density = pyfade_estimate_fog_density(str(temp_path))
            end_time = time.time()
            temp_path.unlink(missing_ok=True)
            self._current_pyfade_time = end_time - start_time
            return density
        except Exception as e:
            print(f"  PyFADE estimation error: {e}")
            self._current_pyfade_time = 0.0
            return 0.0

    def process_video(self, video_path: str, frame_limit: int = 30) -> dict:
        """
        Process video file and test fog impact.

        Parameters
        ----------
        video_path : str
            Path to input video file
        frame_limit : int
            Maximum frames to process

        Returns
        -------
        Comprehensive test results
        """
        self._reset_metrics()  # ensure clean state on repeated calls

        video_path = Path(video_path)

        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return {}

        print(f"Loading video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return {}

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # If frame_limit is 0, process all frames
            actual_limit = frame_count if frame_limit == 0 else min(frame_limit, frame_count)
            print(f"Video properties: {width}x{height} @ {fps} fps, {frame_count} frames")
            print(f"Processing {actual_limit} frames...\n")

            frames = []
            frame_idx = 0

            # Extract frames
            while frame_idx < actual_limit:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize to standard size for consistency
                frame = cv2.resize(frame, (640, 480))
                frames.append(frame.copy())
                frame_idx += 1
        finally:
            cap.release()  # always release, even on exception

        if not frames:
            print("Error: No frames extracted from video")
            return {}

        print(f"Extracted {len(frames)} frames\n")

        # Test each fog level
        for fog_level in self.fog_levels:
            print(f"Testing fog level: {fog_level:.1%}")
            confidences = []
            detection_counts = []
            frame_times = []  # Time per frame
            dcp_estimations = []  # DCP fog density estimates
            pyfade_estimations = []  # PyFADE fog density estimates
            dcp_times = []  # DCP estimation times
            pyfade_times = []  # PyFADE estimation times

            for frame_idx, frame in enumerate(frames):
                # Apply fog
                if fog_level > 0:
                    foggy_frame = self.fog_augmenter.apply_realistic_fog(frame, fog_level)
                else:
                    foggy_frame = frame.copy()

                # Time the detection
                start_time = time.time()
                annotated, detections = process_frame(foggy_frame)
                end_time = time.time()
                frame_time = end_time - start_time
                frame_times.append(frame_time)

                # Extract metrics
                num_detections = len(detections)
                detection_counts.append(num_detections)
                frame_confidences = []
                if detections:
                    confs = [d.get('confidence', 0.0) for d in detections]
                    confidences.extend(confs)
                    frame_confidences = confs

                # Record per-frame metrics
                frame_avg_conf = np.mean(frame_confidences) if frame_confidences else 0.0
                self.per_frame_metrics.append({
                    "frame_index": frame_idx,
                    "fog_level": fog_level,
                    "num_detections": num_detections,
                    "avg_confidence": round(frame_avg_conf, 3),
                    "processing_time_sec": round(frame_time, 4),
                    "detected_objects": [d.get('label', 'unknown') for d in detections]
                })

                # Estimate fog density using both methods (on first frame only for speed)
                if frame_idx == 0:
                    dcp_density = self._estimate_fog_dcp(foggy_frame)
                    pyfade_density = self._estimate_fog_pyfade(foggy_frame)
                    dcp_estimations.append(dcp_density)
                    pyfade_estimations.append(pyfade_density)
                    dcp_times.append(self._current_dcp_time)
                    pyfade_times.append(self._current_pyfade_time)
                    # Also record for first frame in per_frame_metrics
                    self.per_frame_metrics[-1]["dcp_fog_density"] = round(dcp_density, 2)
                    self.per_frame_metrics[-1]["pyfade_fog_density"] = round(pyfade_density, 2)
                    self.per_frame_metrics[-1]["dcp_time"] = round(self._current_dcp_time, 4)
                    self.per_frame_metrics[-1]["pyfade_time"] = round(self._current_pyfade_time, 4)
                    print(f"  DCP Estimated Fog Density: {dcp_density:.2f} (time: {self._current_dcp_time:.4f}s)")
                    print(f"  PyFADE Estimated Fog Density: {pyfade_density:.2f} (time: {self._current_pyfade_time:.4f}s)")

                # Save sample frame for first frame at each fog level
                if frame_idx == 0:
                    fog_label = f"{int(fog_level * 100):02d}"
                    output_path = self.output_dir / f"sample_fog_{fog_label}.jpg"
                    cv2.imwrite(str(output_path), annotated)
                    # Store for composite
                    self.sample_frames[fog_level] = foggy_frame.copy()

            # Compute statistics for this fog level
            avg_confidence = np.mean(confidences) if confidences else 0.0
            confidence_std = np.std(confidences) if confidences else 0.0
            avg_detections = np.mean(detection_counts) if detection_counts else 0.0
            avg_time = np.mean(frame_times) if frame_times else 0.0

            # Use average of DCP/PyFADE estimates across all frames if available
            avg_dcp = np.mean(dcp_estimations) if dcp_estimations else 0.0
            avg_pyfade = np.mean(pyfade_estimations) if pyfade_estimations else 0.0
            avg_dcp_time = np.mean(dcp_times) if dcp_times else 0.0
            avg_pyfade_time = np.mean(pyfade_times) if pyfade_times else 0.0

            self.metrics["fog_level"].append(fog_level)
            self.metrics["avg_confidence"].append(avg_confidence)
            self.metrics["detection_count"].append(sum(detection_counts))
            self.metrics["objects_per_frame"].append(avg_detections)
            self.metrics["confidence_std"].append(confidence_std)
            self.metrics["avg_time_per_frame"].append(avg_time)
            self.metrics["dcp_fog_density"].append(avg_dcp)
            self.metrics["pyfade_fog_density"].append(avg_pyfade)
            self.metrics["dcp_avg_time"].append(avg_dcp_time)
            self.metrics["pyfade_avg_time"].append(avg_pyfade_time)

            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Avg Objects/Frame: {avg_detections:.2f}")
            print(f"  Total Detections: {sum(detection_counts)}")
            print(f"  Avg Time per Frame: {avg_time:.4f}s")
            print()

        return self._compile_results()

    def _compile_results(self) -> dict:
        """Compile all test results with correlation analysis."""
        from scipy import stats

        # Calculate correlation between estimated and actual fog levels
        actual_fog = [f * 100 for f in self.metrics["fog_level"]]  # Convert to 0-100 scale

        dcp_corr = 0.0
        pyfade_corr = 0.0
        if len(actual_fog) > 2:
            dcp_corr, _ = stats.pearsonr(actual_fog, self.metrics["dcp_fog_density"])
            pyfade_corr, _ = stats.pearsonr(actual_fog, self.metrics["pyfade_fog_density"])

        return {
            "timestamp": datetime.now().isoformat(),
            "fog_levels": self.metrics["fog_level"],
            "total_frames_tested": len(self.per_frame_metrics) // len(self.fog_levels) if self.fog_levels else 0,
            "metrics": {
                "avg_confidence": self.metrics["avg_confidence"],
                "detection_count": self.metrics["detection_count"],
                "objects_per_frame": self.metrics["objects_per_frame"],
                "confidence_std": self.metrics["confidence_std"],
                "avg_time_per_frame": self.metrics["avg_time_per_frame"],
                "dcp_fog_density": self.metrics["dcp_fog_density"],
                "pyfade_fog_density": self.metrics["pyfade_fog_density"],
                "dcp_avg_time": self.metrics["dcp_avg_time"],
                "pyfade_avg_time": self.metrics["pyfade_avg_time"]
            },
            "per_frame_metrics": self.per_frame_metrics,  # Detailed per-frame data
            "correlation_analysis": {
                "dcp_correlation": round(dcp_corr, 4),
                "pyfade_correlation": round(pyfade_corr, 4),
                "dcp_mean_error": round(np.mean([abs(e - a) for e, a in zip(self.metrics["dcp_fog_density"], actual_fog)]), 2),
                "pyfade_mean_error": round(np.mean([abs(e - a) for e, a in zip(self.metrics["pyfade_fog_density"], actual_fog)]), 2)
            }
        }

    def generate_charts(self):
        """
        Generate visualization charts of test results.
        """
        if not self.metrics["fog_level"]:
            print("No test data available for charting")
            return

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        fig.suptitle('Fog Impact on Object Detection & Fog Density Estimation', fontsize=16, fontweight='bold')

        # Chart 1: Average Confidence vs Fog Level
        ax1 = axes[0, 0]
        ax1.plot(self.metrics["fog_level"], self.metrics["avg_confidence"],
                marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax1.fill_between(self.metrics["fog_level"],
                        [max(0.0, c - s) for c, s in zip(self.metrics["avg_confidence"], self.metrics["confidence_std"])],
                        [min(1.0, c + s) for c, s in zip(self.metrics["avg_confidence"], self.metrics["confidence_std"])],
                        alpha=0.2, color='#2E86AB')
        ax1.set_xlabel('Fog Level', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Average Confidence', fontsize=11, fontweight='bold')
        ax1.set_title('Detection Confidence Degradation', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Chart 2: Total Detections vs Fog Level
        ax2 = axes[0, 1]
        ax2.bar(self.metrics["fog_level"], self.metrics["detection_count"],
               width=0.08, color='#A23B72', edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Fog Level', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Total Detections', fontsize=11, fontweight='bold')
        ax2.set_title('Detection Count vs Fog', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Chart 3: Average Time per Frame vs Fog Level (enhanced with min/max)
        ax3 = axes[1, 0]
        # Calculate min/max times from per_frame_metrics
        time_by_fog = {}
        for entry in self.per_frame_metrics:
            fl = entry['fog_level']
            if fl not in time_by_fog:
                time_by_fog[fl] = []
            time_by_fog[fl].append(entry['processing_time_sec'])

        min_times = [min(time_by_fog.get(fl, [0])) for fl in self.metrics["fog_level"]]
        max_times = [max(time_by_fog.get(fl, [0])) for fl in self.metrics["fog_level"]]

        # Plot with error bars
        ax3.errorbar(self.metrics["fog_level"], self.metrics["avg_time_per_frame"],
                    yerr=[np.array(self.metrics["avg_time_per_frame"]) - np.array(min_times),
                          np.array(max_times) - np.array(self.metrics["avg_time_per_frame"])],
                    marker='s', linewidth=2, markersize=8, color='#28A745',
                    capsize=5, capthick=2, ecolor='#28A745', alpha=0.7)
        ax3.set_xlabel('Fog Level', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Time per Frame (seconds)', fontsize=11, fontweight='bold')
        ax3.set_title('Processing Time (avg with min/max)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Add time annotations
        for i, (fl, t) in enumerate(zip(self.metrics["fog_level"], self.metrics["avg_time_per_frame"])):
            ax3.annotate(f'{t:.3f}s', (fl, t), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

        # Chart 4: Fog Density Estimation Comparison (DCP)
        ax4 = axes[1, 1]
        ax4.plot(self.metrics["fog_level"], self.metrics["dcp_fog_density"],
                marker='o', linewidth=2, markersize=8, color='#6F42C1', label='DCP Estimated')
        ax4.plot(self.metrics["fog_level"], [f * 100 for f in self.metrics["fog_level"]],
                linestyle='--', linewidth=2, color='#DC3545', label='Actual Fog')
        # Add correlation annotation
        dcp_corr = np.corrcoef([f * 100 for f in self.metrics["fog_level"]], self.metrics["dcp_fog_density"])[0,1]
        ax4.text(0.05, 0.95, f'Correlation: {dcp_corr:.3f}', transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # Add DCP value annotations
        for i, (fl, d) in enumerate(zip(self.metrics["fog_level"], self.metrics["dcp_fog_density"])):
            ax4.annotate(f'{d:.1f}', (fl, d), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax4.set_xlabel('Fog Level', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Estimated Fog Density (DCP)', fontsize=11, fontweight='bold')
        ax4.set_title('DCP Fog Density Estimation', fontsize=12, fontweight='bold')
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)

        # Chart 5: Fog Density Estimation Comparison (PyFADE)
        ax5 = axes[2, 0]
        ax5.plot(self.metrics["fog_level"], self.metrics["pyfade_fog_density"],
                marker='s', linewidth=2, markersize=8, color='#FD7E14', label='PyFADE Estimated')
        ax5.plot(self.metrics["fog_level"], [f * 100 for f in self.metrics["fog_level"]],
                linestyle='--', linewidth=2, color='#DC3545', label='Actual Fog')
        # Add correlation annotation
        pyfade_corr = np.corrcoef([f * 100 for f in self.metrics["fog_level"]], self.metrics["pyfade_fog_density"])[0,1]
        ax5.text(0.05, 0.95, f'Correlation: {pyfade_corr:.3f}', transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # Add PyFADE value annotations
        for i, (fl, p) in enumerate(zip(self.metrics["fog_level"], self.metrics["pyfade_fog_density"])):
            ax5.annotate(f'{p:.1f}', (fl, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        ax5.set_xlabel('Fog Level', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Estimated Fog Density (PyFADE)', fontsize=11, fontweight='bold')
        ax5.set_title('PyFADE Fog Density Estimation', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Chart 6: Detection Rate Decline
        ax6 = axes[2, 1]
        ax6.plot(self.metrics["fog_level"], self.metrics["objects_per_frame"],
                marker='^', linewidth=2, markersize=8, color='#F18F01')
        ax6.set_xlabel('Fog Level', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Objects per Frame (Average)', fontsize=11, fontweight='bold')
        ax6.set_title('Detection Rate Decline', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save chart
        chart_path = self.output_dir / "fog_impact_analysis.png"
        plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
        print(f"Chart saved: {chart_path}\n")

        plt.close()

    def generate_detailed_report(self):
        """
        Generate a detailed text report.
        """
        report_path = self.output_dir / "fog_impact_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("FOG IMPACT ON OBJECT DETECTION - ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 70 + "\n")

            # Find baseline (clear)
            baseline_conf = self.metrics["avg_confidence"][0]
            baseline_objs = self.metrics["objects_per_frame"][0]
            baseline_time = self.metrics["avg_time_per_frame"][0]

            f.write(f"Baseline (Clear) Average Confidence: {baseline_conf:.3f}\n")
            f.write(f"Baseline (Clear) Objects/Frame: {baseline_objs:.2f}\n")
            f.write(f"Baseline (Clear) Avg Time/Frame: {baseline_time:.4f}s\n\n")

            f.write("DETAILED RESULTS BY FOG LEVEL\n")
            f.write("-" * 70 + "\n")

            for i, fog_level in enumerate(self.metrics["fog_level"]):
                conf = self.metrics["avg_confidence"][i]
                objs = self.metrics["objects_per_frame"][i]
                detections = self.metrics["detection_count"][i]
                std = self.metrics["confidence_std"][i]
                avg_time = self.metrics["avg_time_per_frame"][i]
                dcp_density = self.metrics["dcp_fog_density"][i]
                pyfade_density = self.metrics["pyfade_fog_density"][i]
                dcp_time = self.metrics["dcp_avg_time"][i]
                pyfade_time = self.metrics["pyfade_avg_time"][i]

                conf_drop = ((baseline_conf - conf) / baseline_conf * 100) if baseline_conf > 0 else 0
                obj_drop = ((baseline_objs - objs) / baseline_objs * 100) if baseline_objs > 0 else 0
                time_change = ((avg_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0

                f.write(f"\nFog Level: {fog_level:.1%}\n")
                f.write(f"  Average Confidence: {conf:.3f} ({conf_drop:+.1f}% from baseline)\n")
                f.write(f"  Objects per Frame:  {objs:.2f} ({obj_drop:+.1f}% from baseline)\n")
                f.write(f"  Total Detections:   {detections}\n")
                f.write(f"  Confidence Std Dev:  {std:.3f}\n")
                f.write(f"  Avg Time per Frame: {avg_time:.4f}s ({time_change:+.1f}% from baseline)\n")
                f.write(f"  DCP Estimated Fog:   {dcp_density:.2f} (avg time: {dcp_time:.4f}s)\n")
                f.write(f"  PyFADE Estimated:   {pyfade_density:.2f} (avg time: {pyfade_time:.4f}s)\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("KEY FINDINGS\n")
            f.write("=" * 70 + "\n")

            # Critical fog level (>50% confidence drop or 50% detection drop)
            for i, fog_level in enumerate(self.metrics["fog_level"]):
                if i == 0:
                    continue  # skip baseline
                conf = self.metrics["avg_confidence"][i]
                objs = self.metrics["objects_per_frame"][i]

                conf_drop = ((baseline_conf - conf) / baseline_conf * 100) if baseline_conf > 0 else 0
                obj_drop = ((baseline_objs - objs) / baseline_objs * 100) if baseline_objs > 0 else 0

                if conf_drop > 50 or obj_drop > 50:
                    f.write(f"\n[CRITICAL] Fog level {fog_level:.1%} causes significant degradation:\n")
                    f.write(f"   - Confidence drop: {conf_drop:.1f}%\n")
                    f.write(f"   - Detection rate drop: {obj_drop:.1f}%\n")

            # Fog density estimation accuracy
            f.write("\n" + "=" * 70 + "\n")
            f.write("FOG DENSITY ESTIMATION COMPARISON\n")
            f.write("=" * 70 + "\n")
            f.write("(Comparing estimated vs actual fog level)\n\n")

            # Calculate correlation
            actual_fog = [f * 100 for f in self.metrics["fog_level"]]
            dcp_corr = np.corrcoef(actual_fog, self.metrics["dcp_fog_density"])[0,1]
            pyfade_corr = np.corrcoef(actual_fog, self.metrics["pyfade_fog_density"])[0,1]

            f.write("CORRELATION ANALYSIS:\n")
            f.write(f"  DCP Correlation with Actual Fog:    {dcp_corr:.4f}\n")
            f.write(f"  PyFADE Correlation with Actual Fog: {pyfade_corr:.4f}\n\n")

            f.write("NOTE: Both DCP and PyFADE are designed for REAL atmospheric fog,\n")
            f.write("not synthetic fog overlay. These methods analyze:\n")
            f.write("  - Atmospheric scattering patterns\n")
            f.write("  - Depth-based contrast loss\n")
            f.write("  - Color attenuation due to water droplets\n")
            f.write("Synthetic fog (white overlay) doesn't exhibit these properties.\n\n")

            for i, fog_level in enumerate(self.metrics["fog_level"]):
                actual = fog_level * 100
                dcp_est = self.metrics["dcp_fog_density"][i]
                pyfade_est = self.metrics["pyfade_fog_density"][i]

                dcp_error = abs(dcp_est - actual)
                pyfade_error = abs(pyfade_est - actual)

                f.write(f"Fog Level {fog_level:.1%}:\n")
                f.write(f"  DCP Estimated:   {dcp_est:.2f} (error: {dcp_error:.2f})\n")
                f.write(f"  PyFADE Estimated: {pyfade_est:.2f} (error: {pyfade_error:.2f})\n")

            # Recommendations
            f.write("\n" + "=" * 70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 70 + "\n")
            f.write("1. For synthetic fog testing, use the KNOWN fog level (ground truth)\n")
            f.write("2. For real fog detection, train a custom model on real foggy images\n")
            f.write("3. DCP/PyFADE work best with natural fog, not synthetic overlays\n")

        print(f"Report saved: {report_path}\n")

    def generate_json_results(self):
        """
        Save results as JSON for further analysis.
        """
        results = self._compile_results()
        json_path = self.output_dir / "fog_impact_results.json"

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"JSON results saved: {json_path}\n")

    def generate_composite_image(self):
        """
        Generate a composite image showing all fog levels side by side.
        """
        if not self.sample_frames:
            print("No sample frames available for composite")
            return

        # Sort fog levels
        sorted_levels = sorted(self.sample_frames.keys())

        # Calculate dimensions
        n_levels = len(sorted_levels)
        frame_shape = list(self.sample_frames[sorted_levels[0]].shape)

        # Create composite: 2 rows x 3 cols (or appropriate layout)
        cols = 3
        rows = (n_levels + cols - 1) // cols

        # Resize frames for consistent display
        target_height = 200
        target_width = 320

        composite_rows = []
        for row_idx in range(rows):
            row_frames = []
            for col_idx in range(cols):
                level_idx = row_idx * cols + col_idx
                if level_idx < n_levels:
                    fog_level = sorted_levels[level_idx]
                    frame = self.sample_frames[fog_level]
                    resized = cv2.resize(frame, (target_width, target_height))

                    # Add label
                    label = f"Fog: {fog_level:.0%}"
                    cv2.putText(resized, label, (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(resized, label, (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    row_frames.append(resized)
                else:
                    # Empty placeholder
                    placeholder = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    row_frames.append(placeholder)

            if row_frames:
                composite_rows.append(np.hstack(row_frames))

        if composite_rows:
            composite = np.vstack(composite_rows)

            # Add title
            title = "Fog Levels Comparison - Object Detection Test"
            h, w = composite.shape[:2]
            title_bg = np.zeros((40, w, 3), dtype=np.uint8)
            cv2.putText(title_bg, title, (w//2 - 200, 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            final_composite = np.vstack([title_bg, composite])

            composite_path = self.output_dir / "fog_levels_composite.png"
            cv2.imwrite(str(composite_path), final_composite)
            print(f"Composite image saved: {composite_path}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test fog impact on object detection"
    )
    parser.add_argument("video_path", type=str,
                       help="Path to input video file (clear/fog-free)")
    parser.add_argument("--frames", type=int, default=0,
                       help="Maximum frames to process (0 = all frames, default: 0)")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results (default: results)")

    args = parser.parse_args()

    # Create tester
    tester = FogImpactTester(output_dir=args.output)

    # Run tests
    print("\n" + "=" * 70)
    print("FOG IMPACT ON OBJECT DETECTION TEST")
    print("=" * 70 + "\n")

    results = tester.process_video(args.video_path, frame_limit=args.frames)

    if results:
        # Generate outputs
        print("Generating visualizations...")
        tester.generate_composite_image()
        tester.generate_charts()
        tester.generate_detailed_report()
        tester.generate_json_results()

        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {tester.output_dir}\n")
    else:
        print("Test failed - no results generated")


if __name__ == "__main__":
    main()