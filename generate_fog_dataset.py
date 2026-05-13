#!/usr/bin/env python3
"""
Generate Synthetic Fog Dataset
-------------------------------
Creates training/validation datasets with 5 density levels for fog detection evaluation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from synthetic_fog_dataset import SyntheticFogDataset
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic fog dataset')
    parser.add_argument('--source_dir', type=str, default='sample_images',
                       help='Directory containing source images')
    parser.add_argument('--output_dir', type=str, default='synthetic_fog_dataset',
                       help='Output directory for generated dataset')
    parser.add_argument('--samples_per_density', type=int, default=50,
                       help='Number of samples to generate per density level')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    # Find source images
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Source directory {source_dir} not found. Creating sample images...")

        # Create sample directory and generate some test images
        source_dir.mkdir(exist_ok=True)

        # Generate sample images (roads, vehicles, etc.)
        for i in range(10):
            # Create a simple road-like image
            img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Road surface
            cv2.rectangle(img, (0, 240), (640, 480), (50, 50, 50), -1)

            # Lane markings
            cv2.line(img, (320, 240), (320, 480), (255, 255, 255), 2)
            cv2.line(img, (200, 240), (200, 480), (255, 255, 255), 2)
            cv2.line(img, (440, 240), (440, 480), (255, 255, 255), 2)

            # Add some "vehicles" as rectangles
            cv2.rectangle(img, (250, 300), (350, 400), (0, 0, 255), -1)  # Red car
            cv2.rectangle(img, (150, 350), (200, 380), (255, 0, 0), -1)  # Blue car

            # Save sample image
            cv2.imwrite(str(source_dir / "02d"), img)

    # Get list of source images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    source_images = []
    for ext in image_extensions:
        source_images.extend(source_dir.glob(f"*{ext}"))

    if not source_images:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(source_images)} source images")
    source_images = [str(img) for img in source_images]

    # Generate synthetic dataset
    print("Generating synthetic fog dataset...")
    dataset_generator = SyntheticFogDataset(args.output_dir)

    stats = dataset_generator.generate_from_images(
        source_images,
        samples_per_density=args.samples_per_density
    )

    print("Dataset generation complete!")
    print(f"Total samples: {stats['total_samples']}")
    print("Samples per density level:")
    for level, count in stats['samples_per_density'].items():
        density = int(level) / 10.0
        print(".1f")

    # Generate evaluation script
    eval_script = dataset_generator.generate_evaluation_script()
    print(f"Evaluation script generated: {eval_script}")

    print(f"\nDataset saved to: {args.output_dir}")
    print("Run the evaluation script to test fog detection accuracy.")

if __name__ == "__main__":
    main()