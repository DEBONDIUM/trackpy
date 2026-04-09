# Trackpy (simple object tracker)

A simple Python package to track objects (e.g. white spheres) in image sequences using OpenCV.

The tracker detects contours in binarized images and follows object centers across frames.

---

## ✨ Features

- Automatic threshold tuning (interactive)
- Interactive target selection (click on image)
- Multi-target tracking
- Frame-by-frame visualization with navigation
- Export trajectories to .txt
- Plot center evolution

---

## 📦 Installation

Clone the repository:

git clone https://github.com/DEBONDIUM/trackpy.git
cd trackpy

Install the package:

pip install .

For development (optional):
pip install -e .

---

## 📁 Project structure

trackpy/
├── trackpy/        # core package
├── examples/       # usage examples
├── README.md
├── setup.py
└── LICENSE

---

## 🚀 Quick example

from trackpy import Sequence, Tracker

# 1. Create sequence
seq = Sequence("img_exp")
seq.load_all()

# 2. Create tracker (interactive threshold tuning)
tracker = Tracker(seq, threshold=70)

# 3. Add target (interactive click)
tracker.add_target()

# 4. Run tracking
tracker.run()

# 5. Display results
tracker.display_tracking(save_dir="res")

# 6. Plot trajectory
target = tracker.targets[0]
target.display_center_tracking()

# 7. Export trajectories
seq.export_all("res", tracker.targets)

---

## 🎮 Controls

Threshold tuning window:
- ← / → or + / - : adjust threshold
- u / d : ±10 threshold
- ENTER : confirm
- q : abort

Target selection:
- Mouse click : select point
- ENTER : confirm
- q : abort

Tracking visualization:
- → : next frame
- ← : previous frame
- r : reset to first frame
- q : quit
- Window close (X) : abort

---

## 📊 Output

- Annotated frames (optional)
- Trajectory files (.txt) with:

frame_idx    center_x    center_y

- Center evolution plots

---

## 🧠 How it works

1. Images are converted to grayscale
2. A binary threshold is applied
3. Contours are detected
4. The contour containing the target is selected
5. The center (barycenter) is computed
6. The position is propagated to the next frame

---

## 🔧 Requirements

- numpy
- opencv-python
- matplotlib

---

## 📌 Notes

- Works best with:
  - high contrast images
  - isolated objects
- Threshold tuning is critical for good results

---

## 📜 License

MIT License

---

## 👤 Author

Luc Brémaud, Jérémie Girardot, Vincent Fournier
