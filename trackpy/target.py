# =============================================================================
# Librairies
# =============================================================================
import os
from typing import Optional
import matplotlib.pyplot as plt


# =============================================================================
# Class Target
# =============================================================================
class Target:
    """
    Represents a single target to track across frames.
    Stores the initial reference point, current center, and full trajectory.
    """

    _COLORS = [
        (255, 0, 0),  # blue
        (0, 165, 255),  # orange
        (0, 255, 255),  # yellow
        (255, 0, 255),  # magenta
        (0, 255, 0),  # green
    ]

    def __init__(self, _target_id: int, _init_x: int, _init_y: int):
        self._id = _target_id
        self._init_x = _init_x  # never updated
        self._init_y = _init_y
        self._center_x = _init_x  # updated each frame
        self._center_y = _init_y
        self._trajectory = []  # list of (frame_idx, x, y)
        self._color = self._COLORS[_target_id % len(self._COLORS)]

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
    def trajectory(self) -> list[tuple[int, int]]:
        """List of (x, y) positions across frames. (-1, -1) means lost."""
        return [(x, y) for _, x, y in self._trajectory]

    @property
    def frames(self) -> list[int]:
        """List of frame indices where the target was processed."""
        return [f for f, _, _ in self._trajectory]

    @property
    def n_frames(self) -> int:
        """Number of frames processed (including lost ones)."""
        return len(self._trajectory)

    @property
    def n_lost(self) -> int:
        """Number of frames where the target was lost."""
        return sum(1 for _, x, y in self._trajectory if x == -1)

    # --- Methods ---
    def update(self, _center_x: int, _center_y: int, _frame_idx: int):
        """
        Update current center and append to trajectory.
        Pass (-1, -1) when the target is lost on this frame.
        """
        self._center_x = _center_x
        self._center_y = _center_y
        self._trajectory.append((_frame_idx, _center_x, _center_y))

    def export(self, save_dir: str):
        """Save trajectory to a tab-separated .txt file. Lost frames are (-1, -1)."""
        with open(save_dir, "w") as f:
            f.write("frame_idx\tcenter_x\tcenter_y\n")
            for frame_idx, x, y in self._trajectory:
                f.write(f"{frame_idx}\t{x}\t{y}\n")
        print(f"[Target {self._id}] Trajectory saved in : {save_dir}")

    def display_center_tracking(self, save_dir: Optional[str] = None):
        """Display the target center evolution across frames. Lost frames are skipped."""
        if not self._trajectory:
            print(f"[Target {self._id}] No data to display.")
            return

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Skip lost frames (x == -1) for display
        valid = [(f, x, y) for f, x, y in self._trajectory if x != -1]
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
            path = os.path.join(save_dir, f"target_{self._id}_tracking.png")
            plt.savefig(path)
            print(f"[Target {self._id}] Plot saved in : {path}")

        plt.show()

    def __repr__(self):
        return (
            f"Target(id={self._id}, "
            f"pointer=({self._init_x},{self._init_y}), "
            f"frames={self.n_frames}, lost={self.n_lost})"
        )
