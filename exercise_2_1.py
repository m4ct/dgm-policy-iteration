"""
exercise_2_1.py — Supervised learning of v(t, x) with a DGM network.

Trains Net_DGM(hidden=100, 3 layers) to approximate the analytical value
function from Exercise 1.1 via MSE regression on uniform (t, x) samples.

Generates:
  - plots/exercise_2_1_loss.png       (training loss curve)
  - plots/exercise_2_1_surfaces.png   (analytical vs NN surfaces)

Usage:
    python exercise_2_1.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DIM, DEVICE, DTYPE, PLOT_DIR, PLOT_DPI
from lqr_solver import LQRSolver
from networks import Net_DGM


# ── Hyperparameters ─────────────────────────────────────────────────────
HIDDEN_SIZE = 100
N_DGM_LAYERS = 3
N_DATA = 10_000
N_EPOCHS = 3_000
LR = 1e-3
LOG_EVERY = 100


def sample_batch(solver, n, x_range=3.0):
    """Draw uniform (t, x) and compute analytical v as target."""
    t = torch.rand(n, device=DEVICE) * T
    x = (2 * torch.rand(n, DIM, device=DEVICE) - 1) * x_range

    with torch.no_grad():
        v = solver.value_function(t, x)  # (n,)

    return t, x, v.unsqueeze(1)  # target shape (n, 1)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Analytical solution ─────────────────────────────────────────────
    solver = LQRSolver(H, M, C, D, R, SIGMA, T)
    solver.solve_riccati(n_grid=2000)
    print("Riccati ODE solved.")

    # ── Network + optimiser ─────────────────────────────────────────────
    torch.manual_seed(42)
    net = Net_DGM(input_dim=1 + DIM, hidden_size=HIDDEN_SIZE,
                  output_dim=1, n_layers=N_DGM_LAYERS).to(DEVICE)
    optimiser = torch.optim.Adam(net.parameters(), lr=LR)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Net_DGM: {N_DGM_LAYERS} layers × {HIDDEN_SIZE} hidden, "
          f"{n_params:,} parameters")

    # ── Training loop ───────────────────────────────────────────────────
    loss_history = []
    print(f"\nTraining for {N_EPOCHS} epochs ({N_DATA} samples/epoch)...")

    for epoch in range(1, N_EPOCHS + 1):
        t_b, x_b, v_target = sample_batch(solver, N_DATA)

        optimiser.zero_grad()
        v_pred = net(t_b, x_b)
        loss = torch.mean((v_pred - v_target) ** 2)
        loss.backward()
        optimiser.step()

        loss_history.append(loss.item())
        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d}/{N_EPOCHS}  MSE = {loss.item():.4e}")

    print("Training complete.")

    # ── Test evaluation ─────────────────────────────────────────────────
    net.eval()
    N_TEST = 50_000
    t_test, x_test, v_true = sample_batch(solver, N_TEST)

    with torch.no_grad():
        v_pred = net(t_test, x_test)

    mse = torch.mean((v_pred - v_true) ** 2).item()
    rel_err = (mse ** 0.5) / (torch.mean(v_true ** 2) ** 0.5).item()
    print(f"\nTest set ({N_TEST:,} points):")
    print(f"  MSE           = {mse:.4e}")
    print(f"  RMSE          = {mse**0.5:.4e}")
    print(f"  Relative L2   = {rel_err*100:.3f}%")

    # ── Plot 1: Training loss ───────────────────────────────────────────
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.semilogy(range(1, N_EPOCHS + 1), loss_history, linewidth=1, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Exercise 2.1 — Supervised learning of v(t, x): training loss")
    ax.grid(True, which="both", alpha=0.4)
    fig.tight_layout(pad=1.1)
    path = os.path.join(PLOT_DIR, "exercise_2_1_loss.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 2: Analytical vs NN surfaces ───────────────────────────────
    x_range = np.linspace(-2.5, 2.5, 50)
    X1, X2 = np.meshgrid(x_range, x_range)
    x_flat = torch.tensor(
        np.stack([X1.ravel(), X2.ravel()], axis=1),
        dtype=DTYPE, device=DEVICE,
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 11.0), subplot_kw={"projection": "3d"})

    for col, t_val in enumerate([0.0, 0.5]):
        t_flat = torch.full((x_flat.shape[0],), t_val, device=DEVICE)

        with torch.no_grad():
            V_true = solver.value_function(t_flat, x_flat).cpu().numpy().reshape(X1.shape)
            V_nn = net(t_flat, x_flat).squeeze(1).cpu().numpy().reshape(X1.shape)

        for row, (V, label, cmap) in enumerate([
            (V_true, "Analytical", "viridis"),
            (V_nn, "Net_DGM", "plasma"),
        ]):
            ax = axes[row, col]
            ax.plot_surface(X1, X2, V, cmap=cmap, alpha=0.85)
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_zlabel("v", labelpad=8)
            ax.set_title(f"{label}  v(t={t_val}, x)")
            ax.view_init(elev=28, azim=-58)
            ax.set_box_aspect((1.0, 1.0, 0.8))

    fig.suptitle("Exercise 2.1 — Analytical vs NN value function surfaces", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=1.2, w_pad=2.2, h_pad=2.0)
    path = os.path.join(PLOT_DIR, "exercise_2_1_surfaces.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nExercise 2.1 complete.")


if __name__ == "__main__":
    main()
