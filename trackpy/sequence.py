# =============================================================================
# Libraries
# =============================================================================
from pathlib import Path
import re

from .image import Image
# from image import Image


# =============================================================================
# Class Sequence
# =============================================================================
class Sequence:
    """
    Represents an ordered series of images from a folder.

    Responsibilities:
      - discover and sort image files (natural sort: img_2 before img_10)
      - expose the ordered list of Image objects (paths only, no pixel data)
      - provide total count
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, folder_path: Path):
        self._folder_path = Path(folder_path)
        self._images: list[Image] = []

    # --- Properties ---
    @property
    def folder_path(self) -> Path:
        """Path to the image folder."""
        return self._folder_path

    @property
    def images(self) -> list[Image]:
        """Ordered list of Image objects (paths only, no pixel data)."""
        return self._images

    @property
    def n_images(self) -> int:
        """Number of images in the sequence."""
        return len(self._images)

    # --- Methods ---
    def _natural_sort_key(self, filename: str) -> list:
        """
        Natural sort key so that img_2 sorts before img_10.
        Splits the filename into alternating text and numeric chunks,
        comparing numeric parts as integers rather than strings.
        """
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", filename)
        ]

    def load_all(self):
        """
        Discover all supported image files in the folder,
        sort them in natural order, and build the Image list.
        No pixel data is loaded at this stage.
        """
        files = sorted(
            [
                f
                for f in self._folder_path.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ],
            key=lambda f: self._natural_sort_key(f.name),
        )

        if not files:
            raise FileNotFoundError(f"No image found in : {self._folder_path}")

        self._images = [Image(f, idx) for idx, f in enumerate(files)]
        print(f"[Sequence] {len(self._images)} images loaded from {self._folder_path}")

    def __len__(self):
        return len(self._images)

    def __repr__(self):
        return f"Sequence(folder={self._folder_path}, n_images={self.n_images})"
