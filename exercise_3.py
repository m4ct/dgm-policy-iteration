"""
exercise_3.py — Deep Galerkin Method for the HJB PDE (constant control).

Solves the HJB PDE for a fixed constant control α = [1, 1]:
    ∂v/∂t + ½ tr(σσ' ∇²v) + (Hx + Mα)·∇v + x'Cx + α'Dα = 0
    v(T, x) = x' R x

The network is trained by minimising the sum of the PDE residual and
terminal condition errors, then validated against a Monte Carlo estimate.

Generates:
  - plots/exercise_3_loss.png
  - plots/exercise_3_error_vs_mc.png

Usage:
    python exercise_3.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DIM, DEVICE, DTYPE, PLOT_DIR, PLOT_DPI
from networks import SimpleDGMNet
from monte_carlo import ConstantControlMC


# ── Hyperparameters ─────────────────────────────────────────────────────
HIDDEN_DIM = 64
N_LAYERS = 4
EPOCHS = 10_000
BATCH_SIZE = 512
LR = 1e-3
ALPHA = np.array([1.0, 1.0])
LOG_EVERY = 500
EVAL_INTERVAL = 10  # evaluate error every N epochs (saves time)

# ── Test point ──────────────────────────────────────────────────────────
T0_TEST = 0.0
X0_TEST = np.array([0.0, 0.0])


def compute_pde_residual(net, t, x, H_t, M_t, C_t, D_t, sigma_t, alpha_t):
    """
    Evaluate the HJB PDE residual at (t, x) for constant control α.

    Returns shape (B,) residual.
    """
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    u = net(t, x)                              # (B, 1)

    # ∂u/∂t
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

    # ∇u  (B, d)
    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    # Hessian  (B, d, d)
    hess_rows = []
    for i in range(x.shape[1]):
        gi = torch.autograd.grad(grad_u[:, i], x, torch.ones_like(grad_u[:, i]),
                                 create_graph=True)[0]
        hess_rows.append(gi)
    hessian = torch.stack(hess_rows, dim=2)

    # Diffusion: ½ tr(σσ' H)
    ssT = sigma_t @ sigma_t.T
    diffusion = 0.5 * torch.einsum("ij,bji->b", ssT, hessian)

    # Drift: ∇u · (Hx + Mα)
    drift = (grad_u * (x @ H_t.T + (M_t @ alpha_t).unsqueeze(0))).sum(dim=1)

    # Source: x'Cx + α'Dα
    quad_x = torch.einsum("bi,ij,bj->b", x, C_t, x)
    quad_a = alpha_t @ D_t @ alpha_t

    return u_t.squeeze(1) + diffusion + drift + quad_x + quad_a


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Monte Carlo reference ───────────────────────────────────────────
    print("Computing Monte Carlo reference value...")
    mc = ConstantControlMC(H, M, C, D, R, SIGMA, T, ALPHA)
    mc_mean, mc_se = mc.estimate_value(
        T0_TEST, X0_TEST, n_steps=10_000, n_mc=100_000, seed=0
    )
    print(f"  MC mean = {mc_mean:.6f},  std error = {mc_se:.2e}")

    # ── Convert matrices to tensors ─────────────────────────────────────
    H_t = torch.as_tensor(H, dtype=DTYPE, device=DEVICE)
    M_t = torch.as_tensor(M, dtype=DTYPE, device=DEVICE)
    C_t = torch.as_tensor(C, dtype=DTYPE, device=DEVICE)
    D_t = torch.as_tensor(D, dtype=DTYPE, device=DEVICE)
    R_t = torch.as_tensor(R, dtype=DTYPE, device=DEVICE)
    sigma_t = torch.as_tensor(SIGMA, dtype=DTYPE, device=DEVICE)
    alpha_t = torch.as_tensor(ALPHA, dtype=DTYPE, device=DEVICE)

    # ── Network + optimiser ─────────────────────────────────────────────
    torch.manual_seed(42)
    net = SimpleDGMNet(DIM + 1, HIDDEN_DIM, N_LAYERS, output_dim=1).to(DEVICE)
    optimiser = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3000, gamma=0.5)

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"SimpleDGMNet: {N_LAYERS} layers × {HIDDEN_DIM} hidden, "
          f"{n_params:,} parameters")

    # ── Test-point tensors ──────────────────────────────────────────────
    t0_t = torch.tensor([[T0_TEST]], dtype=DTYPE, device=DEVICE)
    x0_t = torch.as_tensor(X0_TEST, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    # ── Training loop ───────────────────────────────────────────────────
    loss_history = []
    error_history = []
    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE})...")

    for epoch in range(1, EPOCHS + 1):
        optimiser.zero_grad()

        # Interior samples
        t_int = torch.rand(BATCH_SIZE, 1, device=DEVICE) * T
        x_int = torch.randn(BATCH_SIZE, DIM, device=DEVICE)

        res = compute_pde_residual(net, t_int, x_int, H_t, M_t, C_t, D_t, sigma_t, alpha_t)
        pde_loss = torch.mean(res ** 2)

        # Terminal condition
        x_T = torch.randn(BATCH_SIZE, DIM, device=DEVICE)
        t_T = torch.full((BATCH_SIZE, 1), T, device=DEVICE)
        u_T = net(t_T, x_T).squeeze(1)
        target_T = torch.einsum("bi,ij,bj->b", x_T, R_t, x_T)
        terminal_loss = torch.mean((u_T - target_T) ** 2)

        loss = pde_loss + terminal_loss
        loss.backward()
        optimiser.step()
        scheduler.step()

        loss_history.append(loss.item())

        # Evaluate error periodically (not every epoch — saves time)
        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            with torch.no_grad():
                val = net(t0_t, x0_t).item()
            error_history.append((epoch, abs(val - mc_mean)))

        if epoch % LOG_EVERY == 0 or epoch == 1:
            with torch.no_grad():
                val = net(t0_t, x0_t).item()
            print(f"  Epoch {epoch:6d}/{EPOCHS}  loss = {loss.item():.6f}  "
                  f"err_vs_MC = {abs(val - mc_mean):.6f}")

    print("Training complete.")

    # ── Final evaluation ────────────────────────────────────────────────
    with torch.no_grad():
        v_dgm = net(t0_t, x0_t).item()
    print(f"\n  DGM value  = {v_dgm:.6f}")
    print(f"  MC value   = {mc_mean:.6f}")
    print(f"  |error|    = {abs(v_dgm - mc_mean):.6f}")
    print(f"  MC 95% CI  = ±{1.96 * mc_se:.6f}")

    # ── Plot 1: Training loss ───────────────────────────────────────────
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, EPOCHS + 1), np.log(loss_history), linewidth=0.4, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log(Loss)")
    ax.set_title("Exercise 3 — DGM training loss (constant control α = [1,1])")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_3_loss.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 2: Error vs MC with rolling average ────────────────────────
    err_epochs = np.array([e for e, _ in error_history])
    err_vals = np.array([v for _, v in error_history])
    log_err = np.log(err_vals + 1e-15)  # avoid log(0)

    # Rolling average (window of 50 evaluations)
    window = 50
    if len(log_err) >= window:
        rolling = np.convolve(log_err, np.ones(window) / window, mode="valid")
        rolling_epochs = err_epochs[window - 1:]
    else:
        rolling = log_err
        rolling_epochs = err_epochs

    threshold = 1.96 * mc_se

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(err_epochs, log_err, linewidth=0.3, color="steelblue", alpha=0.4,
            label="Raw error")
    ax.plot(rolling_epochs, rolling, linewidth=1.5, color="darkblue",
            label=f"Rolling avg (window={window})")
    ax.axhline(np.log(threshold), ls="--", color="red", linewidth=1.5,
               label="95% MC noise level")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log |v_DGM − v_MC|")
    ax.set_title("Exercise 3 — Error vs Monte Carlo reference")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_3_error_vs_mc.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nExercise 3 complete.")


if __name__ == "__main__":
    main()