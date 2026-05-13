"""
Test Synthetic Fog Dataset Generation
--------------------------------------
Tests the fog augmentation and saves comparison images.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from synthetic_fog_dataset import FogAugmentation


def test_fog_augmentation(video_path: str, output_dir: str = "results/fog_comparison"):
    """
    Test fog augmentation and save comparison images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fog_augmenter = FogAugmentation()
    fog_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Cannot read video {video_path}")
        return

    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # Save original
    cv2.imwrite(str(output_path / "original_clear.jpg"), frame)

    # Generate foggy versions
    print("Generating foggy images at different levels...\n")

    for fog_level in fog_levels:
        foggy_frame = fog_augmenter.apply_realistic_fog(frame, fog_level)

        label = f"{int(fog_level * 100):02d}"
        cv2.imwrite(str(output_path / f"fog_{label}.jpg"), foggy_frame)

        # Calculate fog intensity metrics
        diff = cv2.absdiff(frame, foggy_frame)
        mean_diff = np.mean(diff)
        print(f"Fog {fog_level:.0%}: Mean pixel difference from original: {mean_diff:.2f}")

    print(f"\nComparison images saved to: {output_path}")

    # Create side-by-side comparison
    create_comparison_grid(frame, fog_augmenter, output_path, fog_levels)

    return output_path


def create_comparison_grid(original, fog_augmenter, output_path, fog_levels):
    """Create a grid comparison image."""
    # Resize for grid
    h, w = original.shape[:2]
    thumb_w = 320
    thumb_h = 240

    # Create row with original + all fog levels
    row_images = [cv2.resize(original, (thumb_w, thumb_h))]

    for fog_level in fog_levels:
        foggy = fog_augmenter.apply_realistic_fog(original, fog_level)
        resized = cv2.resize(foggy, (thumb_w, thumb_h))

        # Add label
        label = f"Fog {fog_level:.0%}"
        cv2.putText(resized, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(resized, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        row_images.append(resized)

    # Combine horizontally
    grid = np.hstack(row_images)

    # Add title
    title_bar = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
    title = "Synthetic Fog Comparison - Original to 100% Fog"
    cv2.putText(title_bar, title, (grid.shape[1]//2 - 300, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    final_grid = np.vstack([title_bar, grid])

    cv2.imwrite(str(output_path / "fog_comparison_grid.jpg"), final_grid)
    print(f"Grid comparison saved: {output_path / 'fog_comparison_grid.jpg'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test synthetic fog generation")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--output", type=str, default="results/fog_comparison",
                       help="Output directory")

    args = parser.parse_args()

    output = test_fog_augmentation(args.video_path, args.output)
    print(f"\nTest complete! Images saved to: {output}")