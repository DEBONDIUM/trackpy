# =============================================================================
# Libraries
# =============================================================================
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random


# =============================================================================
# Class Target
# =============================================================================
class Target:
    """
    Represents a single target to track across frames.

    Stores:
      - the initial reference point (pointer) — never updated
      - the current center (x, y) — updated each frame
      - the full trajectory as a list of (frame_idx, contour, x, y)

    Lost frames are recorded as (-1, -1) sentinel values.
    """

    # Random color seed
    SEED = 42

    def __init__(self, target_id: int, init_x: int, init_y: int):
        self._id = target_id
        self._init_x = init_x  # initial reference point — never updated
        self._init_y = init_y
        self._center_x = init_x  # current center — updated each frame
        self._center_y = init_y
        self._results = []  # list of (frame_idx, contour, center_x, center_y)

        # Deterministic colour per target
        rng = random.Random(self.SEED + self._id)
        self._color = (
            rng.randint(50, 255),
            rng.randint(50, 255),
            rng.randint(50, 255),
        )

    # --- Properties ---
    @property
    def id(self) -> int:
        """Target unique identifier."""
        return self._id

    @property
    def init_x(self) -> int:
        """Initial x reference point (never changes)."""
        return self._init_x

    @property
    def init_y(self) -> int:
        """Initial y reference point (never changes)."""
        return self._init_y

    @property
    def center_x(self) -> int:
        """Current center x. -1 if lost on the last processed frame."""
        return self._center_x

    @property
    def center_y(self) -> int:
        """Current center y. -1 if lost on the last processed frame."""
        return self._center_y

    @property
    def center(self) -> tuple[int, int]:
        """Current center as (x, y) — used as pointer for the next frame."""
        return (self._center_x, self._center_y)

    @property
    def pointer(self) -> tuple[int, int]:
        """Initial reference point as (x, y) — fixed across all frames."""
        return (self._init_x, self._init_y)

    @property
    def color(self) -> tuple[int, int, int]:
        """BGR color assigned to this target."""
        return self._color

    @property
    def results(self) -> list[tuple[int, np.ndarray, int, int]]:
        """List of (x, y) positions across frames. (-1, -1) means lost."""
        return self._results

    @property
    def trajectory(self) -> list[tuple[int, int]]:
        """List of (x, y) positions across frames. (-1, -1) means lost."""
        return [(x, y) for _, _, x, y in self._results]

    @property
    def frames(self) -> list[int]:
        """List of frame indices where the target was processed."""
        return [f for f, _, _, _ in self._results]

    @property
    def contours(self) -> list[np.ndarray]:
        """List of contours across frames."""
        return [c for _, c, _, _ in self._results]

    @property
    def n_frames(self) -> int:
        """Number of frames processed (including lost ones)."""
        return len(self._results)

    @property
    def n_lost(self) -> int:
        """Number of frames where the target was lost."""
        return sum(1 for _, _, x, y in self._results if x == -1)

    # --- Methods ---
    def update(
        self, center_x: int, center_y: int, frame_idx: int, contour: np.ndarray | None
    ):
        """
        Update current center and append to trajectory.
        Call with (-1, -1) when the target is lost on this frame.
        """
        self._center_x = center_x
        self._center_y = center_y
        self._results.append((frame_idx, contour, center_x, center_y))

    def export(self, save_dir: Path):
        """
        Save trajectory to a tab-separated .txt file.
        Lost frames are written as (-1, -1).
        """
        save_dir = Path(save_dir)
        save_dir.parent.mkdir(parents=True, exist_ok=True)

        with open(save_dir, "w") as f:
            f.write("frame_idx\tcenter_x\tcenter_y\n")
            for frame_idx, _, x, y in self._results:
                f.write(f"{frame_idx}\t{x}\t{y}\n")

        print(f"[Target {self._id}] Trajectory saved in : {save_dir}")

    def display_center_tracking(self, save_dir: Path | None = None):
        """
        Plot the target center (x, y) evolution across frames.
        Lost frames (-1, -1) are excluded from the plot.
        Optionally saves the figure to _save_dir.
        """
        if not self._results:
            print(f"[Target {self._id}] No data to display.")
            return

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Filter out lost frames before plotting
        valid = [(f, x, y) for f, _, x, y in self._results if x != -1]
        frames = [f for f, _, _ in valid]
        xs = [x for _, x, _ in valid]
        ys = [y for _, _, y in valid]

        plt.figure(figsize=(6, 4))
        plt.plot(frames, xs, "-o", label="x")
        plt.plot(frames, ys, "-o", label="y")
        plt.xlabel("Frame index [-]")
        plt.ylabel("Position [px]")
        plt.title(f"Target {self._id} — center tracking")
        plt.legend()
        plt.grid()

        if save_dir:
            path = Path(save_dir) / f"target_{self._id}_tracking.png"
            plt.savefig(path)
            print(f"[Target {self._id}] Plot saved in : {path}")

        plt.show()

    def __repr__(self):
        return (
            f"Target(id={self._id}, "
            f"pointer=({self._init_x},{self._init_y}), "
            f"frames={self.n_frames}, lost={self.n_lost})"
        )
