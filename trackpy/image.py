# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import os
import cv2
from typing import Optional
from .target import Target


# =============================================================================
# Class Image
# =============================================================================
class Image:
    """
    Represents a single frame with its processing pipeline:
    loading, binarization, contour detection, and per-target results.
    """

    def __init__(self, _path: str, _frame_idx: int):
        self._path = _path
        self._frame_idx = _frame_idx
        self._img_gray = None
        self._img_color = None
        self._thresh = None
        self._contours = None
        self._results = {}  # dict: target_id -> {"contour", "center_x", "center_y"}

    # --- Properties ---
    @property
    def path(self) -> str:
        """Path to the image file."""
        return self._path

    @property
    def frame_idx(self) -> int:
        """Frame index in the sequence."""
        return self._frame_idx

    @property
    def img_gray(self) -> Optional[np.ndarray]:
        """Grayscale image array."""
        return self._img_gray

    @property
    def img_color(self) -> Optional[np.ndarray]:
        """Color (BGR) image array."""
        return self._img_color

    @property
    def thresh(self) -> Optional[np.ndarray]:
        """Binarized image array."""
        return self._thresh

    @property
    def contours(self):
        """Detected contours."""
        return self._contours

    @property
    def results(self) -> dict:
        """Per-target results: {target_id: {contour, center_x, center_y}}."""
        return self._results

    # --- Methods ---
    def load(self):
        """Load image from disk."""
        self._img_gray = cv2.imread(self._path, cv2.IMREAD_GRAYSCALE)
        if self._img_gray is None:
            raise FileNotFoundError(f"Image not found : {self._path}")
        self._img_color = cv2.cvtColor(self._img_gray, cv2.COLOR_GRAY2BGR)

    def binarize(self, _threshold: int = 70):
        """Apply binary thresholding."""
        _, self._thresh = cv2.threshold(
            self._img_gray, _threshold, 255, cv2.THRESH_BINARY
        )

    def find_contours(self):
        """Detect contours on the binarized image."""
        self._contours, _ = cv2.findContours(
            self._thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

    def process_target(self, _target: Target) -> bool:
        """
        Find the contour containing the target's current center,
        compute the barycenter and store the result.
        Returns True if found, False otherwise.
        """
        px, py = _target.center

        # If target was lost on previous frame, skip processing
        if px == -1 or py == -1:
            print(
                f"[Image {self._frame_idx}] Target {_target.id} was lost on previous frame, skipping."
            )
            return False

        for c in self._contours:
            if cv2.pointPolygonTest(c, (float(px), float(py)), False) >= 0:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    center_x, center_y = px, py  # fallback

                self._results[_target.id] = {
                    "contour": c,
                    "center_x": center_x,
                    "center_y": center_y,
                }
                return True

        print(
            f"[Image {self._frame_idx}] Target {_target.id} not found near ({px}, {py})"
        )
        return False

    def draw(self, _targets: list[Target]) -> np.ndarray:
        """
        Draw on a copy of the color image:
          - filled white region
          - colored contour
          - barycenter dot (target color)
          - initial reference point (fixed across all frames)
        Returns the annotated image.
        """
        output = self._img_color.copy()

        for target in _targets:
            if target.id not in self._results:
                continue

            res = self._results[target.id]
            contour = res["contour"]
            center_x = res["center_x"]
            center_y = res["center_y"]
            color = target.color

            # Fill region in white
            mask = np.zeros(self._img_gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            output[mask == 255] = (255, 255, 255)

            # Colored contour
            cv2.drawContours(output, [contour], -1, color, 2)

            # Barycenter dot (target color)
            cv2.circle(output, (center_x, center_y), 6, color, -1)

            # Initial reference point: fixed, never changes
            cv2.circle(output, target.pointer, 4, (26, 153, 204), -1)

        return output

    def __repr__(self):
        return f"Image(frame={self._frame_idx}, path={os.path.basename(self._path)})"
