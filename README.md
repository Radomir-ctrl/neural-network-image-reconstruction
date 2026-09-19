# Neural Network Image Reconstruction

## Overview

This Python/Tkinter application learns to reconstruct a 32×32 source image. The GUI shows the source image beside the network's reconstructed image as training proceeds. The network is an educational implementation written directly in Python.

Originally developed as part of a Python neural-network course and later cleaned up for publication.

**Author:** Radomyr Karpan

## How it works

The network maps a pixel's normalized `(x, y)` coordinates to three RGB values. The coordinates are calculated as `x / 16 - 1` and `y / 16 - 1` for pixel indexes 0–31. The displayed RGB prediction is scaled to 0–255 and clamped for drawing.

Forward propagation, the custom piecewise-linear activation, backpropagation, and weight updates are implemented manually in Python. Pillow loads the source image; it does not perform machine learning.

## Network architecture

```text
2 coordinate inputs + bias
    → 32 active hidden units + bias
    → 32 active hidden units + bias
    → 3 RGB outputs
```

The weight matrices have dimensions 3×33, 33×33, and 33×3. Each active unit applies the original activation: slope 0.01 below 0, slope 1 from 0 to 1, and slope 0.01 above 1. The final slot in each hidden layer is a constant bias, not a trainable neuron.

## Training process

Each of the 1,024 pixels in `ex3.png` is a training example. Its coordinates are the input; its RGB channels divided by 255 are the target. Training uses squared error and per-pixel online gradient descent with a learning rate of 0.03. Pixels are processed in the original fixed order, and the same image is trained repeatedly until the window closes. Each GUI update displays a reconstruction before the next training pass.

## Technologies

Python, Tkinter, Pillow for image loading, and a handwritten feed-forward neural network with backpropagation.

## Requirements

- Python 3 with Tkinter support
- Pillow (the only third-party dependency)

## Installation

From this project directory, install the dependency:

```powershell
python -m pip install -r requirements.txt
```

## How to run

```powershell
python neuCoImag.py
```

The source image is resolved relative to the script, so the program can be launched from another working directory. Close the GUI window to stop training.

## Project structure

- `neuCoImag.py` — GUI, network, and training loop
- `ex3.png` — source image used for training
- `requirements.txt` — Pillow dependency
- `.gitignore` — Python and editor-generated files
