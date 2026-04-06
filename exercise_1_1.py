"""
exercise_1_1.py — LQR solution via Riccati ODE.

Solves the Riccati ODE, prints diagnostic values, and generates:
  - plots/exercise_1_1_riccati.png       (S(t) components)
  - plots/exercise_1_1_value_function.png (value function surfaces)

Usage:
    python exercise_1_1.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DEVICE, PLOT_DIR, PLOT_DPI
from lqr_solver import LQRSolver


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Solve Riccati ODE ───────────────────────────────────────────────
    solver = LQRSolver(H, M, C, D, R, SIGMA, T)
    time_grid = np.linspace(0, T, 2000)
    S_vals = solver.solve_riccati(n_grid=2000)
    print(f"Riccati ODE solved ({len(time_grid)} grid points)")

    print(f"\nS(0) =\n{S_vals[0]}")
    print(f"\nS(T) =\n{S_vals[-1]}")
    print(f"Terminal error ‖S(T) − R‖ = {np.linalg.norm(S_vals[-1] - R):.2e}")

    # ── Value function + control at test points ─────────────────────────
    t_test = torch.tensor([0.0, 0.5, 1.0], device=DEVICE)
    x_test = torch.tensor([[1.0, 1.0], [0.5, -0.5], [0.0, 0.0]], device=DEVICE)

    v = solver.value_function(t_test, x_test)
    a = solver.optimal_control(t_test, x_test)

    print("\nTest evaluations:")
    for i in range(3):
        xi = x_test[i].cpu().numpy()
        print(f"  t={t_test[i]:.1f}, x=[{xi[0]:.1f}, {xi[1]:.1f}]  →  "
              f"v={v[i]:.4f}, a=[{a[i,0]:.4f}, {a[i,1]:.4f}]")

    # ── Symmetry check ──────────────────────────────────────────────────
    for t_chk in [0.0, 0.5, 1.0]:
        S = solver.get_S(np.array([t_chk]))[0]
        print(f"  t={t_chk}: ‖S − Sᵀ‖ = {np.linalg.norm(S - S.T):.2e}")

    # ── Plot 1: Riccati solution components ─────────────────────────────
    print("\nGenerating Riccati component plot...")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.6))
    labels = [["S₁₁", "S₁₂"], ["S₂₁", "S₂₂"]]

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.plot(time_grid, S_vals[:, i, j], linewidth=1.5)
            ax.axhline(R[i, j], color="r", linestyle="--", alpha=0.7,
                       label=f"R({i+1},{j+1}) = {R[i,j]:.0f}")
            ax.set_xlabel("t")
            ax.set_title(labels[i][j])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Exercise 1.1 — Riccati solution S(t)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=1.1)
    path = os.path.join(PLOT_DIR, "exercise_1_1_riccati.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 2: Value function surfaces ─────────────────────────────────
    print("Generating value function surface plot...")
    x_range = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x_range, x_range)
    x_flat = torch.tensor(
        np.stack([X1.ravel(), X2.ravel()], axis=1),
        dtype=torch.float32, device=DEVICE,
    )

    fig = plt.figure(figsize=(16.5, 5.8))
    for idx, (t_val, lbl) in enumerate([(0.0, "t = 0"), (0.5, "t = 0.5"), (1.0, "t = T")]):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        t_flat = torch.full((x_flat.shape[0],), t_val, device=DEVICE)
        V = solver.value_function(t_flat, x_flat).cpu().numpy().reshape(X1.shape)
        ax.plot_surface(X1, X2, V, cmap="viridis", alpha=0.85)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_zlabel(f"v({t_val}, x)", labelpad=8)
        ax.set_title(lbl)
        ax.view_init(elev=28, azim=-58)
        ax.set_box_aspect((1.0, 1.0, 0.8))

    fig.suptitle("Exercise 1.1 — Value function v(t, x)", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), pad=1.2, w_pad=2.2)
    path = os.path.join(PLOT_DIR, "exercise_1_1_value_function.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nExercise 1.1 complete.")


if __name__ == "__main__":
    main()
