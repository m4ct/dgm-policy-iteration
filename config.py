"""
config.py — Shared problem parameters and device configuration.

All exercises use the same LQR matrices. Centralised here so nothing is
duplicated and any change propagates everywhere.
"""

import numpy as np
import torch

# ── Device ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Enable cuDNN autotuner — finds fastest convolution/matmul algorithms
torch.backends.cudnn.benchmark = True

# ── LQR problem matrices (d = 2) ───────────────────────────────────────
H = np.array([[0.5, 0.1],
              [0.1, 0.3]], dtype=np.float64)

M = np.array([[1.0, 0.5],
              [0.0, 1.0]], dtype=np.float64)

C = np.array([[2.0, 0.5],
              [0.5, 1.0]], dtype=np.float64)

D = np.array([[2.0, 0.0],
              [0.0, 1.0]], dtype=np.float64)

R = np.array([[1.0, 0.0],
              [0.0, 2.0]], dtype=np.float64)

SIGMA = np.array([[0.3, 0.1],
                  [0.0, 0.2]], dtype=np.float64)

T = 1.0
DIM = 2  # state / control dimension

# ── Plot output ─────────────────────────────────────────────────────────
PLOT_DIR = "plots"
PLOT_DPI = 150