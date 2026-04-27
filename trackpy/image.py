# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import cv2
from pathlib import Path

from .target import Target
# from target import Target


# =============================================================================
# Class Image
# =============================================================================
class Image:
    """
    Represents a single frame identified by its path.

    Design choice: images are NOT kept in memory between calls.
    Each method that needs pixel data reads from disk on the fly,
    processes, and discards — suitable for large high-resolution sequences.
    """

    def __init__(self, path: Path, frame_idx: int):
        self._path = Path(path)
        self._frame_idx = frame_idx

    # --- Properties ---
    @property
    def path(self) -> Path:
        """Path to the image file."""
        return self._path

    @property
    def frame_idx(self) -> int:
        """Frame index in the sequence."""
        return self._frame_idx

    # --- Private helpers ---
    def _load_gray(self) -> np.ndarray:
        """Read image from disk and return as grayscale array. Never stored."""
        img = cv2.imread(str(self._path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found : {self._path}")
        return img

    # --- Public methods ---
    def process(self, threshold: int):
        """
        Detection step for a single frame.

        Pipeline:
        1. Load image from disk (grayscale)
        2. Apply binary threshold
        3. Extract contours
        """
        # Load grayscale image (discarded after function returns)
        img_gray = self._load_gray()

        # Apply binary threshold
        _, thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

        # Detect contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def draw(self, targets: list[Target], frame_idx: int) -> np.ndarray:
        """
        Load the image from disk, draw annotations, and return the result.
        Pixel data is not stored after this call.

        Draws for each target (if found on this frame):
          - filled white region inside the contour
          - colored contour border
          - barycenter dot (target color)
          - initial reference point (fixed teal dot)
        """
        # Reload from disk for drawing — discarded after return
        img_gray = self._load_gray()
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        for target in targets:
            # Find the result corresponding to this specific frame
            result = next(
                (
                    (fi, c, cx, cy)
                    for fi, c, cx, cy in target.results
                    if fi == frame_idx
                ),
                None,
            )

            # Skip if no result for this frame or if target was lost
            if result is None:
                continue
            _, contour, cx, cy = result
            if cx == -1 or cy == -1 or contour is None:
                continue

            color = target.color

            mask = np.zeros(img_gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            img_color[mask == 255] = (255, 255, 255)
            cv2.drawContours(img_color, [contour], -1, color, 2)
            cv2.circle(img_color, (cx, cy), 6, color, -1)
            cv2.circle(img_color, target.pointer, 4, (26, 153, 204), -1)

        return img_color

    def __repr__(self):
        return f"Image(frame={self._frame_idx}, path={self._path.name})"
