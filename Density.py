from pyfade import fade
import numpy as np
import cv2


def get_fog_density(image_path):

    # Read image once
    img = cv2.imread(image_path)

    # Direct FADE on NumPy array
    score = fade(img)

    # Normalize
    fog_density = np.interp(
        score,
        [0, 8],
        [0, 100]
    )

    return round(fog_density, 2)