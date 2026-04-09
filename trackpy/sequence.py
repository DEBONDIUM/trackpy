# =============================================================================
# Libraries
# =============================================================================
import os
import re
from .image import Image
from .target import Target


# =============================================================================
# Class Sequence
# =============================================================================
class Sequence:
    """
    Represents an ordered series of images from a folder.
    Handles loading, natural sorting, and trajectory export.
    """

    SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, _folder_path: str):
        self._folder_path = _folder_path
        self._images: list[Image] = []

    # --- Properties ---
    @property
    def folder_path(self) -> str:
        """Path to the image folder."""
        return self._folder_path

    @property
    def images(self) -> list[Image]:
        """Ordered list of Image objects."""
        return self._images

    @property
    def n_images(self) -> int:
        """Number of images in the sequence."""
        return len(self._images)

    # --- Methods ---
    def _natural_sort_key(self, _filename: str):
        """Natural sort key: numbers are compared numerically."""
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", _filename)
        ]

    def load_all(self):
        """Load all images from the folder in natural order."""
        files = sorted(
            [
                f
                for f in os.listdir(self._folder_path)
                if os.path.splitext(f)[1].lower() in self.SUPPORTED_EXTENSIONS
            ],
            key=self._natural_sort_key,
        )

        if not files:
            raise FileNotFoundError(f"No image found in : {self._folder_path}")

        self._images = [
            Image(os.path.join(self._folder_path, f), idx)
            for idx, f in enumerate(files)
        ]
        print(f"[Sequence] {len(self._images)} images loaded from {self._folder_path}")

    def export_all(self, save_dir: str, targets: list[Target]):
        """Export trajectory of each target to a separate .txt file."""
        os.makedirs(save_dir, exist_ok=True)
        for target in targets:
            path = os.path.join(save_dir, f"target_{target.id}_trajectory.txt")
            target.export(path)

    def __len__(self):
        return len(self._images)

    def __repr__(self):
        return f"Sequence(folder={self._folder_path}, n_images={self.n_images})"
