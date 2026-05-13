import cv2
import numpy as np


# =========================================================
# DARK CHANNEL PRIOR DEHAZING CLASS
# =========================================================

class DehazeModel:

    def __init__(
        self,
        window_size=15,
        omega=0.95,
        t0=0.1,
        refine=True
    ):

        self.window_size = window_size
        self.omega = omega
        self.t0 = t0
        self.refine = refine
        self.guided_filter_available = self._check_guided_filter()

    def _check_guided_filter(self) -> bool:
        """Check if guided filter is available."""
        try:
            # Try to access ximgproc
            cv2.ximgproc.guidedFilter
            return True
        except (AttributeError, ImportError):
            print("[Info] Guided filter not available, using bilateral filter instead")
            return False

    # =====================================================
    # MAIN PROCESS FUNCTION
    # =====================================================

    def process(self, frame):

        img = frame.astype(np.float64) / 255.0

        # -------------------------------------------------
        # DARK CHANNEL
        # -------------------------------------------------

        min_channel = np.min(img, axis=2)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.window_size, self.window_size)
        )

        dark = cv2.erode(min_channel, kernel)

        # -------------------------------------------------
        # ATMOSPHERIC LIGHT
        # -------------------------------------------------

        h, w = dark.shape

        num_pixels = h * w

        num_brightest = max(int(num_pixels * 0.001), 1)

        dark_vec = dark.reshape(num_pixels)

        img_vec = img.reshape(num_pixels, 3)

        indices = np.argsort(dark_vec)[-num_brightest:]

        atmosphere = np.mean(
            img_vec[indices],
            axis=0
        )

        # -------------------------------------------------
        # TRANSMISSION MAP
        # -------------------------------------------------

        norm_img = np.empty_like(img)

        for i in range(3):

            norm_img[:, :, i] = (
                img[:, :, i]
                /
                max(atmosphere[i], 1e-6)
            )

        transmission = (
            1.0
            -
            self.omega
            *
            cv2.erode(
                np.min(norm_img, axis=2),
                kernel
            )
        )

        transmission = np.clip(
            transmission,
            0,
            1
        )

        # -------------------------------------------------
        # TRANSMISSION REFINEMENT (Guided Filter or Bilateral)
        # -------------------------------------------------

        if self.refine:
            if self.guided_filter_available:
                try:
                    gray = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY
                    ).astype(np.float32) / 255.0

                    transmission = cv2.ximgproc.guidedFilter(
                        guide=(gray * 255).astype(np.float32),
                        src=transmission.astype(np.float32),
                        radius=60,
                        eps=1e-4
                    )
                except Exception as e:
                    print(f"[Warning] Guided filter failed: {e}, using bilateral")
                    transmission = self._bilateral_refine(transmission)
            else:
                # Fallback to bilateral filter
                transmission = self._bilateral_refine(transmission)

        # -------------------------------------------------
        # RECOVER IMAGE
        # -------------------------------------------------

        transmission = np.maximum(
            transmission,
            self.t0
        )

        recovered = np.empty_like(img)

        for i in range(3):

            recovered[:, :, i] = (
                (
                    img[:, :, i]
                    -
                    atmosphere[i]
                )
                /
                transmission
            ) + atmosphere[i]

        recovered = np.clip(
            recovered,
            0,
            1
        )

        dehazed = (
            recovered * 255
        ).astype(np.uint8)

        return dehazed

    def _bilateral_refine(self, transmission: np.ndarray) -> np.ndarray:
        """Fallback bilateral filter for transmission refinement."""
        # Convert to uint8 for bilateral filter
        trans_uint8 = (transmission * 255).astype(np.uint8)
        # Apply bilateral filter (edge-preserving smoothing)
        refined = cv2.bilateralFilter(trans_uint8, 9, 75, 75)
        return refined.astype(np.float32) / 255.0