"""
exercise_2_2.py — Supervised learning of a*(t, x) with an FFN.

Trains FFN([3, 100, 100, 2]) to approximate the optimal Markov control
from Exercise 1.1 via MSE regression on uniform (t, x) samples.

Generates:
  - plots/exercise_2_2_loss.png            (training loss, joint + per-component)
  - plots/exercise_2_2_vector_fields.png   (analytical vs FFN quiver plots)
  - plots/exercise_2_2_component_error.png (absolute vs relative error bars)

Usage:
    python exercise_2_2.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DIM, DEVICE, DTYPE, PLOT_DIR, PLOT_DPI
from lqr_solver import LQRSolver
from networks import FFN


# ── Hyperparameters ─────────────────────────────────────────────────────
SIZES = [1 + DIM, 100, 100, DIM]   # [3, 100, 100, 2]
N_DATA = 10_000
N_ITER = 3_000
LR = 1e-3
LR_MILESTONES = [1000, 2000]
LR_GAMMA = 0.1
LOG_EVERY = 100


def sample_batch(solver, n, x_range=3.0):
    """Draw uniform (t, x) and compute analytical a* as target."""
    t = torch.rand(n, device=DEVICE) * T
    x = (2 * torch.rand(n, DIM, device=DEVICE) - 1) * x_range

    with torch.no_grad():
        a = solver.optimal_control(t, x)  # (n, d)

    # FFN takes concatenated [t, x]
    tx = torch.cat([t.unsqueeze(1), x], dim=1)
    return tx, a


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Analytical solution ─────────────────────────────────────────────
    solver = LQRSolver(H, M, C, D, R, SIGMA, T)
    solver.solve_riccati(n_grid=2000)
    print("Riccati ODE solved.")

    # ── Network + optimiser ─────────────────────────────────────────────
    torch.manual_seed(42)
    net = FFN(sizes=SIZES).to(DEVICE)
    optimiser = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, milestones=LR_MILESTONES, gamma=LR_GAMMA
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"FFN architecture: {SIZES},  {n_params:,} parameters")

    # ── Training loop ───────────────────────────────────────────────────
    loss_hist, loss_a1, loss_a2 = [], [], []
    print(f"\nTraining for {N_ITER} iterations ({N_DATA} samples/iter)...")

    for it in range(1, N_ITER + 1):
        tx_b, a_target = sample_batch(solver, N_DATA)

        optimiser.zero_grad()
        a_pred = net(tx_b)
        loss = torch.mean((a_pred - a_target) ** 2)
        loss.backward()
        optimiser.step()
        scheduler.step()

        loss_hist.append(loss.item())
        loss_a1.append(torch.mean((a_pred[:, 0] - a_target[:, 0]) ** 2).item())
        loss_a2.append(torch.mean((a_pred[:, 1] - a_target[:, 1]) ** 2).item())

        if it % LOG_EVERY == 0 or it == 1:
            print(f"  Iter {it:5d}/{N_ITER}  MSE = {loss.item():.4e}")

    print("Training complete.")

    # ── Test evaluation ─────────────────────────────────────────────────
    net.eval()
    N_TEST = 50_000
    tx_test, a_true = sample_batch(solver, N_TEST)

    with torch.no_grad():
        a_pred = net(tx_test)

    mse = torch.mean((a_pred - a_true) ** 2).item()
    mse_comp = torch.mean((a_pred - a_true) ** 2, dim=0)
    rel_err = (mse ** 0.5) / (torch.mean(a_true ** 2) ** 0.5).item()

    print(f"\nTest set ({N_TEST:,} points):")
    print(f"  MSE  (joint) = {mse:.4e}")
    print(f"  MSE  (a₁)    = {mse_comp[0].item():.4e}")
    print(f"  MSE  (a₂)    = {mse_comp[1].item():.4e}")
    print(f"  Relative L2  = {rel_err*100:.3f}%")

    # ── Plot 1: Training loss ───────────────────────────────────────────
    print("\nGenerating plots...")
    iters = np.arange(1, N_ITER + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(iters, loss_hist, lw=1.2, color="darkorange", label="Joint MSE")
    ax.semilogy(iters, loss_a1, lw=0.9, color="steelblue", ls="--", label="MSE a₁")
    ax.semilogy(iters, loss_a2, lw=0.9, color="seagreen", ls="--", label="MSE a₂")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("Exercise 2.2 — Supervised learning of a*(t, x): training loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_2_2_loss.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 2: Vector fields ───────────────────────────────────────────
    x_range = np.linspace(-2.5, 2.5, 15)
    X1, X2 = np.meshgrid(x_range, x_range)
    x_np = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for col, t_val in enumerate([0.0, 0.5]):
        n_pts = x_np.shape[0]
        t_1d = torch.full((n_pts,), t_val, device=DEVICE)
        x_t = torch.as_tensor(x_np, device=DEVICE)

        with torch.no_grad():
            a_true_q = solver.optimal_control(t_1d, x_t).cpu().numpy()
            tx = torch.cat([t_1d.unsqueeze(1), x_t], dim=1)
            a_nn_q = net(tx).cpu().numpy()

        for row, (a_vals, label, color) in enumerate([
            (a_true_q, "Analytical", "steelblue"),
            (a_nn_q, "FFN", "darkorange"),
        ]):
            ax = axes[row, col]
            ax.quiver(X1.ravel(), X2.ravel(), a_vals[:, 0], a_vals[:, 1],
                      color=color, alpha=0.8, scale=30, scale_units="inches")
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")
            ax.set_title(f"{label}  a*(t={t_val}, x)")
            ax.set_xlim(-2.8, 2.8)
            ax.set_ylim(-2.8, 2.8)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Exercise 2.2 — Analytical vs FFN control vector fields", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_2_2_vector_fields.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 3: Component error bars ────────────────────────────────────
    a_true_np = a_true.cpu().numpy()
    a_pred_np = a_pred.cpu().numpy()
    sq_err = (a_pred_np - a_true_np) ** 2
    var_a = np.var(a_true_np, axis=0)
    rel_sq_err = sq_err / var_a[np.newaxis, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    mse_vals = [np.mean(sq_err[:, 0]), np.mean(sq_err[:, 1])]
    ax1.bar(["a₁", "a₂"], mse_vals, color=["steelblue", "seagreen"], alpha=0.85, width=0.4)
    ax1.set_ylabel("MSE")
    ax1.set_title("Absolute MSE per component")
    ax1.grid(True, axis="y", alpha=0.4)
    for i, v in enumerate(mse_vals):
        ax1.text(i, v * 1.02, f"{v:.2e}", ha="center", fontsize=10)

    rel_vals = [np.mean(rel_sq_err[:, 0]), np.mean(rel_sq_err[:, 1])]
    ax2.bar(["a₁", "a₂"], rel_vals, color=["steelblue", "seagreen"], alpha=0.85, width=0.4)
    ax2.set_ylabel("MSE / Var(aᵢ)")
    ax2.set_title("Relative MSE per component")
    ax2.grid(True, axis="y", alpha=0.4)
    for i, v in enumerate(rel_vals):
        ax2.text(i, v * 1.02, f"{v:.2e}", ha="center", fontsize=10)

    fig.suptitle(
        "Exercise 2.2 — Absolute vs relative error per control component\n"
        "a₂ has larger absolute MSE due to scale (D⁻¹ weights: 0.5 vs 1.0), "
        "not harder learning",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_2_2_component_error.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print(f"\n  Var(a₁) = {var_a[0]:.4f},  Var(a₂) = {var_a[1]:.4f}")
    print(f"  Relative MSE a₁ = {np.mean(rel_sq_err[:,0]):.4e}")
    print(f"  Relative MSE a₂ = {np.mean(rel_sq_err[:,1]):.4e}")

    print("\nExercise 2.2 complete.")


if __name__ == "__main__":
    main()
