# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from .target import Target
from .sequence import Sequence
# from target import Target
# from sequence import Sequence


# =============================================================================
# Class Tracker
# =============================================================================
class Tracker:
    """
    Orchestrates multi-target tracking across a sequence of images.

    Workflow:
      - opens interactive threshold tuning on the first frame
      - opens interactive point selection for each target
      - processes all frames, updates target trajectories
      - navigable annotated frame viewer
      - saves all target trajectories to .txt files
    """

    _BANNER_COLOR = (0.8, 0.6, 0.1)  # golden yellow (matplotlib RGB, range 0-1)

    # Fixed OpenCV display window size — independent of image resolution
    _WIN_W = 800
    _WIN_H = 600

    # Fixed overlay style — text and progress bar always the same size
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.5
    _FONT_THICK = 1
    _BAR_H = 10  # progress bar height in pixels
    _PAD = 5  # text background padding in pixels

    def __init__(
        self, sequence: Sequence, init_threshold: int = 70, img_idx_calibration: int = 0
    ):
        self._sequence = sequence
        self._threshold = init_threshold
        self._img_idx_calibration = img_idx_calibration
        self._targets: list[Target] = []

        # Immediately open the interactive threshold tuning window
        self._tune_threshold()

    # --- Properties ---
    @property
    def sequence(self) -> Sequence:
        """The image sequence being tracked."""
        return self._sequence

    @property
    def threshold(self) -> int:
        """Binarization threshold used for all frames."""
        return self._threshold

    @property
    def img_idx_calibration(self) -> int:
        """The indice of the image used to tune threshold and pick target."""
        return self._img_idx_calibration

    @threshold.setter
    def threshold(self, value: int):
        self._threshold = value

    @property
    def targets(self) -> list[Target]:
        """List of registered targets."""
        return self._targets

    @property
    def n_targets(self) -> int:
        """Number of registered targets."""
        return len(self._targets)

    # --- Private methods ---
    def _tune_threshold(self):
        """
        Opens an interactive matplotlib window on the first frame.
        Shows the original image side by side with the binarized result.

        Controls:
          slider        : drag to adjust threshold
          +/right arrow : threshold + 1
          -/left arrow  : threshold - 1
          u             : threshold + 10
          d             : threshold - 10
          q             : abort (raises InterruptedError)
          ENTER         : confirm and close
        """
        if not self._sequence.images:
            print("[Tracker] No images in sequence, skipping threshold tuning.")
            return

        # Load image directly — not stored in the Image object
        first = self._sequence.images[self._img_idx_calibration]
        img_gray = cv2.imread(str(first.path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Cannot load first image : {first.path}")

        result = {"confirmed": False}

        # --- Figure layout: [images | images] / [slider] / [banner] ---
        fig = plt.figure(figsize=(9, 4))
        gs = fig.add_gridspec(
            3, 2, height_ratios=[15, 1.5, 1.5], hspace=0.15, wspace=0.05
        )
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_slider = fig.add_subplot(gs[1, :])
        ax_banner = fig.add_subplot(gs[2, :])

        # Original image (left panel — static)
        ax_left.imshow(img_gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax_left.set_title("Original", fontsize=10)
        ax_left.axis("off")

        # Binarized image (right panel — updated live as threshold changes)
        _, binary = cv2.threshold(img_gray, self._threshold, 255, cv2.THRESH_BINARY)
        im_right = ax_right.imshow(binary, cmap="gray", vmin=0, vmax=255, aspect="auto")
        title_right = ax_right.set_title(
            f"Binary  |  threshold = {self._threshold}", fontsize=10
        )
        ax_right.axis("off")

        # Threshold slider (integer steps, range 0-255)
        slider = Slider(
            ax_slider,
            "Threshold",
            0,
            255,
            valinit=self._threshold,
            valstep=1,
            color=self._BANNER_COLOR,
        )

        # Keyboard shortcuts banner
        ax_banner.set_facecolor("black")
        ax_banner.axis("off")
        ax_banner.text(
            0.5,
            0.5,
            "+/- or arrows : ±1    u/d : ±10    ENTER : confirm    q : abort",
            ha="center",
            va="center",
            fontsize=9,
            color=self._BANNER_COLOR,
            transform=ax_banner.transAxes,
        )

        def _update(_val):
            # Called whenever the slider moves — recompute and refresh binary image
            self._threshold = int(slider.val)
            _, b = cv2.threshold(img_gray, self._threshold, 255, cv2.THRESH_BINARY)
            im_right.set_data(b)
            title_right.set_text(f"Binary  |  threshold = {self._threshold}")
            fig.canvas.draw_idle()

        def _on_key(_event):
            if _event.key == "enter":
                # Confirm: store threshold and close
                result["confirmed"] = True
                plt.close(fig)
            elif _event.key == "q":
                # Abort: close without confirming
                plt.close(fig)
            elif _event.key in ["+", "right"]:
                slider.set_val(min(255, self._threshold + 1))
            elif _event.key in ["-", "left"]:
                slider.set_val(max(0, self._threshold - 1))
            elif _event.key == "u":
                slider.set_val(min(255, self._threshold + 10))
            elif _event.key == "d":
                slider.set_val(max(0, self._threshold - 10))

        # Connect slider and keyboard events
        slider.on_changed(_update)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        # Block until the user closes the window (ENTER or q)
        plt.show(block=True)

        if not result["confirmed"]:
            raise InterruptedError("Threshold tuning aborted by user.")

        print(f"[Tracker] Threshold set to {self._threshold}")

    def _pick_target_point(self) -> tuple[int, int]:
        """
        Opens the first binarized frame in a matplotlib window.
        The user clicks to place/replace a selection marker.

        Controls:
          left click : place/move the selection marker
          ENTER      : confirm selection and close
          q          : abort (raises InterruptedError)
        Closing the window without pressing ENTER also aborts.
        """
        # Load and binarize image — not stored in the Image object
        first = self._sequence.images[self._img_idx_calibration]
        img_gray = cv2.imread(str(first.path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Cannot load first image : {first.path}")

        _, binary = cv2.threshold(img_gray, self._threshold, 255, cv2.THRESH_BINARY)

        clicked = {"x": None, "y": None}  # last clicked pixel coordinates
        result = {"confirmed": False}  # set to True on ENTER
        marker = [None]  # holds the cross plot artist

        # --- Figure: binarized image + fixed bottom banner ---
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07)

        ax.imshow(binary, cmap="gray")
        ax.set_title(f"Target {self.n_targets} — select point", fontsize=10)
        ax.axis("off")

        # Fixed banner placed at the bottom of the figure (outside axes)
        banner_text = fig.text(
            0.5,
            0.02,
            "Click on the target",
            ha="center",
            va="bottom",
            fontsize=9,
            color=self._BANNER_COLOR,
            backgroundcolor="black",
        )

        def _update_banner(_msg, _color=None):
            """Update banner message and optionally change its color."""
            banner_text.set_text(_msg)
            banner_text.set_color(_color or self._BANNER_COLOR)
            fig.canvas.draw_idle()

        def _on_click(_event):
            """Handle left mouse clicks on the image axes."""
            # Ignore clicks outside the image or non-left-button clicks
            if _event.inaxes != ax or _event.button != 1:
                return

            # Record clicked coordinates (rounded to nearest pixel)
            clicked["x"] = int(round(_event.xdata))
            clicked["y"] = int(round(_event.ydata))

            # Remove previous marker if any, place a new cross at clicked position
            if marker[0] is not None:
                marker[0].remove()
            (marker[0],) = ax.plot(
                clicked["x"],
                clicked["y"],
                "+",
                color=self._BANNER_COLOR,
                markersize=14,
                markeredgewidth=2,
            )

            # Update banner with coordinates and available actions
            _update_banner(
                f"Selected : ({clicked['x']}, {clicked['y']})    |    ENTER : confirm    q : abort"
            )

        def _on_key(_event):
            """Handle keyboard input."""
            if _event.key == "enter":
                # Confirm only if a point has already been clicked
                if clicked["x"] is None:
                    _update_banner("Click on the target first !", "red")
                    return
                result["confirmed"] = True
                plt.close(fig)
            elif _event.key == "q":
                # Abort without confirming
                plt.close(fig)

        # Connect mouse and keyboard callbacks
        fig.canvas.mpl_connect("button_press_event", _on_click)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        # Block until the user closes the window
        plt.show(block=True)

        if not result["confirmed"] or clicked["x"] is None:
            raise InterruptedError("Target selection aborted by user.")

        print(f"[Tracker] Target point selected : ({clicked['x']}, {clicked['y']})")
        return clicked["x"], clicked["y"]

    def _make_display_frame(
        self, annotated: np.ndarray, idx: int, total: int, frame_idx: int
    ) -> np.ndarray:
        """
        Prepare a display-ready frame from a raw annotated image:
          1. Scale to fit _WIN_W x _WIN_H while preserving aspect ratio
          2. Center on a black canvas of exactly _WIN_W x _WIN_H
          3. Overlay a progress bar at the bottom
          4. Overlay a frame counter in the top-left corner
        Returns the composited canvas.
        """
        h_img, w_img = annotated.shape[:2]

        # Compute scale factor to fit within the fixed window
        scale = min(self._WIN_W / w_img, self._WIN_H / h_img)
        new_w = int(w_img * scale)
        new_h = int(h_img * scale)

        # Resize the annotated image
        resized = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Place on a black canvas, centered
        canvas = np.zeros((self._WIN_H, self._WIN_W, 3), dtype=np.uint8)
        x_off = (self._WIN_W - new_w) // 2
        y_off = (self._WIN_H - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        h, w = canvas.shape[:2]

        # Progress bar: dark background + light fill proportional to current frame
        cv2.rectangle(canvas, (0, h - self._BAR_H), (w, h), (50, 50, 50), -1)
        bar_w = int((idx + 1) / total * w)
        cv2.rectangle(canvas, (0, h - self._BAR_H), (bar_w, h), (200, 200, 200), -1)

        # Frame counter: white background box + black text
        text = f"Frame {frame_idx} ({idx + 1}/{total})"
        (tw, th), baseline = cv2.getTextSize(
            text, self._FONT, self._FONT_SCALE, self._FONT_THICK
        )
        x, y = 10, 25
        cv2.rectangle(
            canvas,
            (x - self._PAD, y - th - self._PAD),
            (x + tw + self._PAD, y + baseline + self._PAD),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (x, y),
            self._FONT,
            self._FONT_SCALE,
            (0, 0, 0),
            self._FONT_THICK,
            cv2.LINE_AA,
        )

        return canvas

    # --- Public methods ---
    def add_target(self) -> Target:
        """
        Open an interactive selection window on the first binarized frame,
        then register a new Target at the clicked position.
        Can be called multiple times to add several targets.
        """
        init_x, init_y = self._pick_target_point()
        target = Target(len(self._targets), init_x, init_y)
        self._targets.append(target)
        print(f"[Tracker] Added {target}")
        return target

    def run(self):
        """
        Main tracking loop.

        For each frame:
          1. Detect contours (Image.process)
          2. Associate each target with a contour (point-in-polygon test)
          3. Compute barycenter of matched contour
          4. Update target state
          5. Handle lost targets

        Notes:
            - Tracking logic is centralized here
            - Targets store their own history
            - Images are processed one by one (low memory usage)
        """
        if not self._targets:
            raise ValueError("No target defined. Use add_target() first.")

        for img_obj in self._sequence.images:
            # Step 1: detect contours in current frame
            contours = img_obj.process(self._threshold)

            # Step 2–4: associate contours to targets and update them
            for target in self._targets:
                px, py = target.center

                # Skip already lost targets
                if px == -1 or py == -1:
                    target.update(-1, -1, img_obj.frame_idx, None)
                    continue

                found = False
                for c in contours:
                    # Check if previous center lies inside contour
                    if cv2.pointPolygonTest(c, (float(px), float(py)), False) >= 0:
                        # Compute barycenter using image moments
                        M = cv2.moments(c)

                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            # Fallback: keep previous position
                            cx, cy = px, py

                        # Update target with new position
                        target.update(cx, cy, img_obj.frame_idx, c)
                        found = True
                        break

                # Step 5: handle lost target
                if not found:
                    target.update(-1, -1, img_obj.frame_idx, None)
                    print(
                        f"[Tracker] Target {target.id} lost on frame {img_obj.frame_idx}"
                    )

            print(f"[Tracker] Frame {img_obj.frame_idx} processed")

    def display_tracking(self, save_dir: Path | None = None):
        """
        Display all annotated frames in a fixed-size OpenCV window.

        If save_dir is provided, ALL frames are saved to disk first
        (independently of which frames are visited during navigation).

        Navigation keys:
          right arrow : next frame
          left arrow  : previous frame
          r           : reset to first frame
          q           : quit viewer
        """
        images = self._sequence.images
        total = len(images)

        # --- Save all frames to disk before opening the viewer ---
        if save_dir:
            _save_dir = Path(save_dir)
            _save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Tracker] Saving {total} frames to {_save_dir} ...")
            for img_obj in images:
                annotated = img_obj.draw(self._targets, img_obj.frame_idx)
                out_path = _save_dir / f"frame_{img_obj.frame_idx:04d}.png"
                cv2.imwrite(str(out_path), annotated)
            print(f"[Tracker] {total} frames saved.")

        # --- Interactive frame viewer ---
        idx = 0
        cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracker", self._WIN_W, self._WIN_H)

        while True:
            img_obj = images[idx]
            annotated = img_obj.draw(self._targets, img_obj.frame_idx)
            display = self._make_display_frame(annotated, idx, total, img_obj.frame_idx)
            cv2.imshow("Tracker", display)

            # Block until a key is pressed
            key = cv2.waitKey(0)

            if key == ord("q"):
                break
            elif key in [83, 2555904]:  # right arrow - advance one frame
                idx = (idx + 1) % total
            elif key in [81, 2424832]:  # left arrow  - go back one frame
                idx = (idx - 1) % total
            elif key == ord("r"):  # r - jump back to first frame
                idx = 0

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def export_trajectories(self, save_dir: Path, targets: list[Target] | None = None):
        """
        Export the trajectory of every target to a separate .txt file.
        Files are named: target_0_trajectory.txt, target_1_trajectory.txt, ...
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if targets == None:
        	targets = self._targets

        for target in targets:
            path = save_dir / f"target_{target.id}_trajectory.txt"
            target.export(path)

    def __repr__(self):
        return (
            f"Tracker(sequence={self._sequence}, "
            f"n_targets={self.n_targets}, threshold={self._threshold})"
        )
