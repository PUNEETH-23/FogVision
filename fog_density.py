import cv2
import numpy as np
from typing import Union
import sys
import os

# Append PyFADE src path to import pyfade
_pyfade_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "PyFADE", "src"))
if _pyfade_src_path not in sys.path:
    sys.path.insert(0, _pyfade_src_path)

from pyfade import fade


def _prepare_image(image_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(image_or_path, np.ndarray):
        img = image_or_path
    else:
        img = cv2.imread(image_or_path)

    if img is None:
        raise ValueError(
            f"Could not open image: {image_or_path}"
        )

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def estimate_fog_density(image_or_path: Union[str, np.ndarray]) -> float:
    """
    Estimate fog density using PyFADE only.
    Returns:
        float: percentage fog density (0.0 to 100.0)
    """
    img = _prepare_image(image_or_path)

    try:
        fade_score = fade(img)
        # Scale FADE score from range [0.3, 3.0] to [0.0, 100.0] percentage
        fog_density_score = ((fade_score - 0.3) / 2.7) * 100.0
        fog_density_score = max(0.0, min(100.0, fog_density_score))
    except Exception as e:
        print(f"[PyFADE] Error in estimate_fog_density: {e}")
        # fallback based on basic image statistics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        brightness_score = (255 - mean_brightness) / 255
        contrast_score = std_brightness / 128
        fog_density_score = (brightness_score * 0.6 + (1 - contrast_score) * 0.4) * 100
        fog_density_score = max(0.0, min(100.0, fog_density_score))

    return round(fog_density_score, 2)