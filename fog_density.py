import cv2
import numpy as np
from typing import Union


def get_dark_channel(img, window_size=15):

    min_channel = np.min(img, axis=2)

    kernel = cv2.getStructuringElement(

        cv2.MORPH_RECT,

        (window_size, window_size)
    )

    dark_channel = cv2.erode(

        min_channel,

        kernel
    )

    return dark_channel


def get_atmospheric_light(img, dark_channel):

    h, w = img.shape[:2]

    num_pixels = h * w

    num_brightest = int(max(num_pixels * 0.001, 1))

    dark_vec = dark_channel.reshape(num_pixels)

    img_vec = img.reshape(num_pixels, 3)

    indices = np.argsort(dark_vec)[::-1][:num_brightest]

    atmospheric_light = np.mean(

        img_vec[indices],

        axis=0
    )

    return atmospheric_light


def get_transmission_map(

    img,

    atmospheric_light,

    window_size=15,

    omega=0.95
):

    norm_img = np.zeros_like(

        img,

        dtype=np.float64
    )

    for i in range(3):

        norm_img[:, :, i] = (

            img[:, :, i]
            /
            max(atmospheric_light[i], 1e-6)
        )

    transmission = (

        1
        -
        omega
        *
        get_dark_channel(norm_img, window_size)
    )

    return transmission


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


def estimate_fog_density(image_or_path: Union[str, np.ndarray]):

    img = _prepare_image(image_or_path)
    img_float = img.astype(np.float64) / 255.0

    # ---------------------------------------------------
    # DARK CHANNEL
    # ---------------------------------------------------

    dark_channel = get_dark_channel(img_float)

    # ---------------------------------------------------
    # ATMOSPHERIC LIGHT
    # ---------------------------------------------------

    atmospheric_light = get_atmospheric_light(

        img_float,

        dark_channel
    )

    # ---------------------------------------------------
    # TRANSMISSION MAP
    # ---------------------------------------------------

    transmission_map = get_transmission_map(

        img_float,

        atmospheric_light
    )

    # ---------------------------------------------------
    # VISUAL MAPS
    # ---------------------------------------------------

    dark_channel_visual = (

        dark_channel * 255
    ).astype(np.uint8)

    transmission_visual = (

        transmission_map * 255
    ).astype(np.uint8)

    # ---------------------------------------------------
    # FOG DENSITY SCORE
    # ---------------------------------------------------

    fog_density_score = np.mean(

        dark_channel
    ) * 100

    fog_density_score = round(

        fog_density_score,

        2
    )

    # ---------------------------------------------------
    # RETURN
    # ---------------------------------------------------

    return {

        "dark_channel": dark_channel_visual,

        "transmission_map": transmission_visual,

        "fog_density": fog_density_score
    }