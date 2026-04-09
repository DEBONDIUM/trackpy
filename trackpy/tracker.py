# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import os
import cv2
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from .target import Target
from .sequence import Sequence


# =============================================================================
# Class Tracker
# =============================================================================
class Tracker:
    """
    Orchestrates multi-target tracking across a sequence of images.
    On init, opens an interactive threshold tuning window on the first frame.
    """

    _BANNER_COLOR = (0.8, 0.6, 0.1)  # golden yellow (matplotlib RGB)

    # Fixed display window size for display_tracking
    _WIN_W = 800
    _WIN_H = 600

    # Fixed overlay parameters (independent of image resolution)
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.5
    _FONT_THICK = 1
    _BAR_H = 10  # progress bar height in pixels
    _PAD = 5  # text padding in pixels

    def __init__(self, sequence: Sequence, threshold: int = 70):
        self._sequence = sequence
        self._threshold = threshold
        self._targets: list[Target] = []

        # Open interactive threshold tuning on the first frame
        self._tune_threshold()

    # --- Properties ---
    @property
    def sequence(self) -> Sequence:
        """The image sequence."""
        return self._sequence

    @property
    def threshold(self) -> int:
        """Binarization threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, _value: int):
        self._threshold = _value

    @property
    def targets(self) -> list[Target]:
        """List of tracked targets."""
        return self._targets

    @property
    def n_targets(self) -> int:
        """Number of targets."""
        return len(self._targets)

    # --- Private methods ---
    def _tune_threshold(self):
        """
        Opens an interactive matplotlib window on the first frame.
        Shows the original image side by side with the binarized result.

        Controls:
          slider        : adjust threshold
          +/right arrow : threshold + 1
          -/left arrow  : threshold - 1
          u             : threshold + 10
          d             : threshold - 10
          q             : abort
          ENTER         : confirm and close
        """
        if not self._sequence.images:
            print("[Tracker] No images in sequence, skipping threshold tuning.")
            return

        first = self._sequence.images[0]
        first.load()
        img_gray = first.img_gray

        result = {"confirmed": False}

        fig = plt.figure(figsize=(9, 4))
        gs = fig.add_gridspec(
            3, 2, height_ratios=[15, 1.5, 1.5], hspace=0.15, wspace=0.05
        )

        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        ax_slider = fig.add_subplot(gs[1, :])
        ax_banner = fig.add_subplot(gs[2, :])

        # Original image
        ax_left.imshow(img_gray, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax_left.set_title("Original", fontsize=10)
        ax_left.axis("off")

        # Binarized image
        _, binary = cv2.threshold(img_gray, self._threshold, 255, cv2.THRESH_BINARY)
        im_right = ax_right.imshow(binary, cmap="gray", vmin=0, vmax=255, aspect="auto")
        title_right = ax_right.set_title(
            f"Binary  |  threshold = {self._threshold}", fontsize=10
        )
        ax_right.axis("off")

        # Slider
        slider = Slider(
            ax_slider,
            "Threshold",
            0,
            255,
            valinit=self._threshold,
            valstep=1,
            color=self._BANNER_COLOR,
        )

        # Banner
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
            self._threshold = int(slider.val)
            _, b = cv2.threshold(img_gray, self._threshold, 255, cv2.THRESH_BINARY)
            im_right.set_data(b)
            title_right.set_text(f"Binary  |  threshold = {self._threshold}")
            fig.canvas.draw_idle()

        def _on_key(_event):
            if _event.key == "enter":
                result["confirmed"] = True
                plt.close(fig)
            elif _event.key == "q":
                plt.close(fig)
            elif _event.key in ["+", "right"]:
                slider.set_val(min(255, self._threshold + 1))
            elif _event.key in ["-", "left"]:
                slider.set_val(max(0, self._threshold - 1))
            elif _event.key == "u":
                slider.set_val(min(255, self._threshold + 10))
            elif _event.key == "d":
                slider.set_val(max(0, self._threshold - 10))

        slider.on_changed(_update)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        plt.show(block=True)

        if not result["confirmed"]:
            raise InterruptedError("Threshold tuning aborted by user.")

        print(f"[Tracker] Threshold set to {self._threshold}")

    def _pick_target_point(self) -> tuple[int, int]:
        """
        Opens the first binarized frame in a matplotlib window.
        Click to select/reselect, ENTER to confirm, q to abort.
        Closing the window without ENTER aborts.
        """
        first = self._sequence.images[0]
        first.load()  # safe reload in case img_gray was cleared
        img_gray = first.img_gray
        _, binary = cv2.threshold(img_gray, self._threshold, 255, cv2.THRESH_BINARY)

        clicked = {"x": None, "y": None}
        result = {"confirmed": False}
        marker = [None]

        fig, ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07)

        ax.imshow(binary, cmap="gray")
        ax.set_title(f"Target {self.n_targets} — select point", fontsize=10)
        ax.axis("off")

        # Fixed banner at the bottom of the figure
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
            banner_text.set_text(_msg)
            banner_text.set_color(_color or self._BANNER_COLOR)
            fig.canvas.draw_idle()

        def _on_click(_event):
            if _event.inaxes != ax or _event.button != 1:
                return
            clicked["x"] = int(round(_event.xdata))
            clicked["y"] = int(round(_event.ydata))
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
            _update_banner(
                f"Selected : ({clicked['x']}, {clicked['y']})    |    ENTER : confirm    q : abort"
            )

        def _on_key(_event):
            if _event.key == "enter":
                if clicked["x"] is None:
                    _update_banner("Click on the target first !", "red")
                    return
                result["confirmed"] = True
                plt.close(fig)
            elif _event.key == "q":
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", _on_click)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        plt.show(block=True)

        if not result["confirmed"] or clicked["x"] is None:
            raise InterruptedError("Target selection aborted by user.")

        print(f"[Tracker] Target point selected : ({clicked['x']}, {clicked['y']})")
        return clicked["x"], clicked["y"]

    def _make_display_frame(
        self, _annotated: np.ndarray, _idx: int, _total: int, _frame_idx: int
    ) -> np.ndarray:
        """
        Resize the annotated image to fit the fixed display window,
        then overlay the progress bar and frame counter.
        Returns the display-ready frame.
        """
        h_img, w_img = _annotated.shape[:2]

        # Scale to fit within fixed window while preserving aspect ratio
        scale = min(self._WIN_W / w_img, self._WIN_H / h_img)
        new_w = int(w_img * scale)
        new_h = int(h_img * scale)
        display = cv2.resize(_annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad to exact window size with black borders
        canvas = np.zeros((self._WIN_H, self._WIN_W, 3), dtype=np.uint8)
        x_off = (self._WIN_W - new_w) // 2
        y_off = (self._WIN_H - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = display

        h, w = canvas.shape[:2]

        # Progress bar
        cv2.rectangle(canvas, (0, h - self._BAR_H), (w, h), (50, 50, 50), -1)
        bar_w = int((_idx + 1) / _total * w)
        cv2.rectangle(canvas, (0, h - self._BAR_H), (bar_w, h), (200, 200, 200), -1)

        # Frame counter with white background
        text = f"Frame {_frame_idx} ({_idx + 1}/{_total})"
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
        Opens the binarized first frame for interactive point selection,
        then registers a new target at the clicked position.
        """
        init_x, init_y = self._pick_target_point()
        target = Target(len(self._targets), init_x, init_y)
        self._targets.append(target)
        print(f"[Tracker] Added {target}")
        return target

    def run(self):
        """
        Main loop: for each image, for each target,
        find the contour, compute barycenter, propagate position.
        Lost targets (not found in contours) are recorded as (-1, -1).
        """
        if not self._targets:
            raise ValueError("No target defined. Use add_target() first.")

        for img_obj in self._sequence.images:
            img_obj.load()
            img_obj.binarize(self._threshold)
            img_obj.find_contours()

            for target in self._targets:
                found = img_obj.process_target(target)
                if found:
                    res = img_obj.results[target.id]
                    target.update(res["center_x"], res["center_y"], img_obj.frame_idx)
                else:
                    # Target lost on this frame: record (-1, -1) as sentinel value
                    target.update(-1, -1, img_obj.frame_idx)
                    print(
                        f"[Tracker] Target {target.id} lost on frame {img_obj.frame_idx}"
                    )

            print(f"[Tracker] Frame {img_obj.frame_idx} processed")

    def display_tracking(self, save_dir: Optional[str] = None):
        """
        Display each annotated frame in a fixed-size window (_WIN_W x _WIN_H).
        Image is scaled to fit while preserving aspect ratio, with black padding.
        If _save_dir is provided, ALL frames are saved before display opens.
    
        Navigation:
          right arrow : next frame
          left arrow  : previous frame
          r           : reset to first frame
          q           : quit
        Closing the window raises InterruptedError.
        """
        images = self._sequence.images
        total  = len(images)
    
        # --- Save all frames first, independently of navigation ---
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"[Tracker] Saving {total} frames to {save_dir} ...")
            for img_obj in images:
                annotated = img_obj.draw(self._targets)
                out_path  = os.path.join(save_dir, f"frame_{img_obj.frame_idx:04d}.png")
                cv2.imwrite(out_path, annotated)
            print(f"[Tracker] {total} frames saved.")
    
        # --- Interactive display ---
        idx = 0
        cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracker", self._WIN_W, self._WIN_H)
    
        while True:
            img_obj   = images[idx]
            annotated = img_obj.draw(self._targets)
            display   = self._make_display_frame(annotated, idx, total, img_obj.frame_idx)
            cv2.imshow("Tracker", display)
    
            key = cv2.waitKey(0)
    
            # Check if window was closed by user
            try:
                if cv2.getWindowProperty("Tracker", cv2.WND_PROP_VISIBLE) < 1:
                    raise InterruptedError("Tracking aborted by user (window closed).")
            except cv2.error:
                raise InterruptedError("Tracking aborted by user (window closed).")
    
            if key == ord("q"):
                break
            elif key in [83, 2555904]:   # right arrow → next frame
                idx = (idx + 1) % total
            elif key in [81, 2424832]:   # left arrow  → previous frame
                idx = (idx - 1) % total
            elif key == ord("r"):        # r           → reset to first frame
                idx = 0
    
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    # def display_tracking(self, save_dir: Optional[str] = None):
    #     """
    #     Display each annotated frame in a fixed-size window (_WIN_W x _WIN_H).
    #     Image is scaled to fit while preserving aspect ratio, with black padding.

    #     Navigation:
    #       right arrow : next frame
    #       left arrow  : previous frame
    #       r           : reset to first frame
    #       q           : quit
    #     Closing the window raises InterruptedError.
    #     """
    #     if save_dir:
    #         os.makedirs(save_dir, exist_ok=True)

    #     images = self._sequence.images
    #     total = len(images)
    #     idx = 0

    #     cv2.namedWindow("Tracker", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("Tracker", self._WIN_W, self._WIN_H)

    #     while True:
    #         img_obj = images[idx]
    #         annotated = img_obj.draw(self._targets)

    #         if save_dir:
    #             out_path = os.path.join(save_dir, f"frame_{img_obj.frame_idx:04d}.png")
    #             cv2.imwrite(out_path, annotated)

    #         display = self._make_display_frame(annotated, idx, total, img_obj.frame_idx)
    #         cv2.imshow("Tracker", display)

    #         key = cv2.waitKey(0)

    #         # Check if window was closed by user
    #         try:
    #             if cv2.getWindowProperty("Tracker", cv2.WND_PROP_VISIBLE) < 1:
    #                 raise InterruptedError("Tracking aborted by user (window closed).")
    #         except cv2.error:
    #             raise InterruptedError("Tracking aborted by user (window closed).")

    #         if key == ord("q"):
    #             break
    #         elif key in [83, 2555904]:  # next frame
    #             idx = (idx + 1) % total
    #         elif key in [81, 2424832]:  # previous frame
    #             idx = (idx - 1) % total
    #         elif key == ord("r"):  # reset to first frame
    #             idx = 0

    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)

    def __repr__(self):
        return (
            f"Tracker(sequence={self._sequence}, "
            f"n_targets={self.n_targets}, threshold={self._threshold})"
        )
