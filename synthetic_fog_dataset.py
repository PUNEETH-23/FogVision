"""
synthetic_fog_dataset.py
------------------------
Synthetic fog dataset generator using OpenCV fog augmentation.

Creates training/validation datasets with 5 density levels for fog detection
model evaluation and benchmarking.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json


class FogAugmentation:
    """
    OpenCV-based fog augmentation for creating synthetic foggy images.
    """

    def __init__(self):
        self.density_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # 5 fog density levels

    def apply_fog(self, image: np.ndarray, density: float) -> np.ndarray:
        """
        Apply fog effect to image using depth-based fog simulation.

        Parameters
        ----------
        image : np.ndarray
            Input BGR image
        density : float
            Fog density (0.0 = clear, 1.0 = heavy fog)

        Returns
        -------
        Foggy image
        """
        if density <= 0:
            return image.copy()

        # Create depth map (simplified - distance from center)
        h, w = image.shape[:2]
        center_y, center_x = h // 2, w // 2

        # Create distance-based depth map
        y_coords, x_coords = np.ogrid[:h, :w]
        depth_map = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        depth_map = depth_map / depth_map.max()  # Normalize to 0-1

        # Atmospheric scattering model
        beta = density * 0.1  # Scattering coefficient

        # Transmission map
        transmission = np.exp(-beta * depth_map)

        # Airlight (fog color - slightly bluish)
        airlight = np.full_like(image, [200, 210, 220], dtype=np.uint8)

        # Apply fog
        foggy_image = image.astype(np.float32) * transmission[:, :, np.newaxis] + \
                     airlight.astype(np.float32) * (1 - transmission[:, :, np.newaxis])

        return np.clip(foggy_image, 0, 255).astype(np.uint8)

    def apply_realistic_fog(self, image: np.ndarray, fog_level: float = 0.5) -> np.ndarray:
        """
        Realistic fog simulation - adds haze without inverting colors
        fog_level: 0.0 to 1.0
        """
        if fog_level <= 0:
            return image.copy()

        img = image.astype(np.float32) / 255.0
        h, w = img.shape[:2]

        # Create vertical depth map - distant area = more fog
        y = np.linspace(0, 1, h)
        depth_map = np.tile(y[:, np.newaxis], (1, w))

        # Exponential fog based on depth
        beta = 3.0 * fog_level
        transmission = np.exp(-beta * depth_map)
        transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)

        # Atmospheric light - grayish fog color
        A = np.array([0.7, 0.7, 0.75])  # Slight blue-gray

        # Add fog overlay - blend original with atmospheric light
        foggy = img * transmission + A * (1 - transmission)

        # Add slight blur for haze effect
        blur_strength = int(9 * fog_level)
        if blur_strength % 2 == 0:
            blur_strength += 1
        if blur_strength > 1:
            foggy = cv2.GaussianBlur(foggy, (blur_strength, blur_strength), 0)

        # Reduce contrast slightly
        contrast_factor = 1.0 - (fog_level * 0.3)
        foggy = np.clip((foggy - 0.5) * contrast_factor + 0.5, 0, 1)

        # Reduce saturation
        hsv = cv2.cvtColor((foggy * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        sat_factor = 1.0 - (fog_level * 0.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255).astype(np.uint8)
        foggy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

        return (foggy * 255).astype(np.uint8)


class SyntheticFogDataset:
    """
    Generates synthetic fog dataset with multiple density levels.
    """

    def __init__(self, output_dir: str = "synthetic_fog_dataset"):
        """
        Parameters
        ----------
        output_dir : str
            Directory to save generated dataset
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.fog_augmenter = FogAugmentation()
        self.density_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        # Create subdirectories
        for density in self.density_levels + [0.0]:  # Include clear images
            (self.output_dir / "images" / "02d").mkdir(parents=True, exist_ok=True)

    def generate_from_images(self, source_images: List[str],
                           samples_per_density: int = 50) -> Dict[str, Any]:
        """
        Generate synthetic fog dataset from source images.

        Parameters
        ----------
        source_images : list
            List of paths to source images
        samples_per_density : int
            Number of samples to generate per density level

        Returns
        -------
        Dataset statistics
        """
        dataset_info = {
            "total_samples": 0,
            "samples_per_density": {},
            "source_images_used": len(source_images),
            "density_levels": self.density_levels
        }

        for density in self.density_levels + [0.0]:  # Include clear baseline
            density_str = "02d"
            density_dir = self.output_dir / "images" / density_str
            annotations = []

            samples_created = 0

            for img_path in source_images:
                if samples_created >= samples_per_density:
                    break

                try:
                    # Load source image
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    # Apply fog
                    if density > 0:
                        foggy_image = self.fog_augmenter.apply_realistic_fog(image, density)
                    else:
                        foggy_image = image

                    # Save image
                    img_filename = f"{Path(img_path).stem}_fog_{density_str}_{samples_created:04d}.jpg"
                    img_path_out = density_dir / img_filename
                    cv2.imwrite(str(img_path_out), foggy_image)

                    # Create annotation
                    annotation = {
                        "image_path": str(img_path_out.relative_to(self.output_dir)),
                        "source_image": img_path,
                        "fog_density": density,
                        "fog_level": self._density_to_level(density),
                        "image_shape": image.shape[:2]
                    }
                    annotations.append(annotation)

                    samples_created += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

            # Save annotations for this density
            annotations_file = self.output_dir / "annotations" / f"fog_density_{density_str}.json"
            annotations_file.parent.mkdir(exist_ok=True)

            with open(annotations_file, 'w') as f:
                json.dump({
                    "density": density,
                    "level": self._density_to_level(density),
                    "samples": annotations
                }, f, indent=2)

            dataset_info["samples_per_density"][density_str] = samples_created
            dataset_info["total_samples"] += samples_created

            print(f"Generated {samples_created} samples for fog density {density}")

        # Save overall dataset info
        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        return dataset_info

    def _density_to_level(self, density: float) -> str:
        """
        Convert density value to descriptive level.
        """
        if density == 0.0:
            return "clear"
        elif density <= 0.2:
            return "light_fog"
        elif density <= 0.4:
            return "moderate_fog"
        elif density <= 0.6:
            return "heavy_fog"
        elif density <= 0.8:
            return "dense_fog"
        else:
            return "very_dense_fog"

    def generate_evaluation_script(self) -> str:
        """
        Generate Python script for evaluating fog detection on synthetic dataset.
        """
        script_content = '''"""
Fog Detection Evaluation Script
-------------------------------
Evaluates fog detection accuracy on synthetic fog dataset.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from fog_density import estimate_fog_density
from typing import Dict, List, Any

def evaluate_fog_detection(dataset_dir: str) -> Dict[str, Any]:
    """
    Evaluate fog detection performance on synthetic dataset.
    """
    dataset_path = Path(dataset_dir)
    annotations_dir = dataset_path / "annotations"

    results = {
        "density_levels": {},
        "overall_metrics": {},
        "confusion_matrix": {}
    }

    all_true_densities = []
    all_predicted_densities = []

    for annotation_file in annotations_dir.glob("*.json"):
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        density = data["density"]
        level = data["level"]
        samples = data["samples"]

        level_results = {
            "true_density": density,
            "samples_evaluated": 0,
            "mae": [],  # Mean Absolute Error
            "mse": [],  # Mean Squared Error
            "accuracy": []  # Within 10% accuracy
        }

        for sample in samples:
            img_path = dataset_path / sample["image_path"]

            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Estimate fog density
                fog_result = estimate_fog_density(image)
                predicted_density = fog_result["fog_density"] / 100.0  # Convert to 0-1

                true_density = sample["fog_density"]

                # Calculate errors
                abs_error = abs(predicted_density - true_density)
                sq_error = abs_error ** 2
                accuracy = 1.0 if abs_error <= 0.1 else 0.0

                level_results["mae"].append(abs_error)
                level_results["mse"].append(sq_error)
                level_results["accuracy"].append(accuracy)
                level_results["samples_evaluated"] += 1

                all_true_densities.append(true_density)
                all_predicted_densities.append(predicted_density)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        # Compute averages for this level
        if level_results["mae"]:
            level_results["mean_mae"] = np.mean(level_results["mae"])
            level_results["mean_mse"] = np.mean(level_results["mse"])
            level_results["mean_accuracy"] = np.mean(level_results["accuracy"])
            level_results["std_mae"] = np.std(level_results["mae"])

        results["density_levels"][level] = level_results

    # Overall metrics
    if all_true_densities:
        overall_mae = np.mean([abs(p - t) for p, t in zip(all_predicted_densities, all_true_densities)])
        overall_mse = np.mean([(p - t)**2 for p, t in zip(all_predicted_densities, all_true_densities)])
        overall_accuracy = np.mean([1.0 if abs(p - t) <= 0.1 else 0.0
                                   for p, t in zip(all_predicted_densities, all_true_densities)])

        results["overall_metrics"] = {
            "mae": overall_mae,
            "mse": overall_mse,
            "rmse": np.sqrt(overall_mse),
            "accuracy": overall_accuracy,
            "total_samples": len(all_true_densities)
        }

    return results

if __name__ == "__main__":
    results = evaluate_fog_detection("synthetic_fog_dataset")
    print("Fog Detection Evaluation Results:")
    print(json.dumps(results, indent=2))

    # Save results
    with open("fog_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
'''

        script_path = self.output_dir / "evaluate_fog_detection.py"
        with open(script_path, 'w') as f:
            f.write(script_content)

        return str(script_path)
    