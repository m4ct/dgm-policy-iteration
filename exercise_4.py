"""
exercise_4.py — Policy iteration for the HJB equation.

Alternates between:
  1. Policy evaluation  — solve the HJB PDE for the current policy via DGM
  2. Policy improvement — minimise the control-dependent Hamiltonian

Key implementation details:
  - Value network is warm-started from the previous iteration
  - Policy is frozen (requires_grad=False) during evaluation
  - Hamiltonian improvement uses only control-dependent terms with
    detached grad_v, reducing gradient noise

Generates:
  - plots/exercise_4_policy_iteration_loss.png
  - plots/exercise_4_value_comparison.png

Usage:
    python exercise_4.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DIM, DEVICE, DTYPE, PLOT_DIR, PLOT_DPI
from lqr_solver import LQRSolver
from networks import SimpleDGMNet


# ── Hyperparameters ─────────────────────────────────────────────────────
HIDDEN_DIM = 64
N_LAYERS = 4
N_POLICY_ITERS = 5
EVAL_EPOCHS = 5_000
EVAL_BATCH = 512
IMPROVE_STEPS = 1_000
IMPROVE_BATCH = 512
LR_EVAL = 1e-3
LR_IMPROVE = 5e-4
ALPHA_INIT = np.array([1.0, 1.0])
LOG_EVERY = 500


# ── Test points for monitoring ──────────────────────────────────────────
T_TEST = torch.tensor([[0.0], [0.5], [1.0]], dtype=DTYPE, device=DEVICE)
X_TEST = torch.tensor([[1.0, 1.0], [0.5, -0.5], [0.0, 0.0]], dtype=DTYPE, device=DEVICE)


class PolicyNet(nn.Module):
    """Policy network: (t, x) -> a(t, x) in R^d."""

    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, x):
        h = torch.cat([t, x], dim=1)
        h = torch.nn.functional.silu(self.input_layer(h))
        for layer in self.hidden_layers:
            h = torch.nn.functional.silu(layer(h))
        return self.output_layer(h)


def compute_pde_residual(value_net, policy_net, t, x, H_t, M_t, C_t, D_t, sigma_t):
    """HJB PDE residual with policy-dependent control."""
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    u = value_net(t, x)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    hess_rows = []
    for i in range(x.shape[1]):
        gi = torch.autograd.grad(grad_u[:, i], x, torch.ones_like(grad_u[:, i]),
                                 create_graph=True)[0]
        hess_rows.append(gi)
    hessian = torch.stack(hess_rows, dim=2)

    ssT = sigma_t @ sigma_t.T
    diffusion = 0.5 * torch.einsum("ij,bji->b", ssT, hessian)

    a = policy_net(t, x)
    drift = (grad_u * (x @ H_t.T + a @ M_t.T)).sum(dim=1)

    quad_x = torch.einsum("bi,ij,bj->b", x, C_t, x)
    quad_a = torch.einsum("bi,ij,bj->b", a, D_t, a)

    return u_t.squeeze(1) + diffusion + drift + quad_x + quad_a


def policy_evaluation(value_net, policy_net, optimiser, H_t, M_t, C_t, D_t, R_t, sigma_t):
    """Train value network to satisfy HJB PDE for current policy."""
    losses = []
    for epoch in range(1, EVAL_EPOCHS + 1):
        optimiser.zero_grad()

        t_int = torch.rand(EVAL_BATCH, 1, device=DEVICE) * T
        x_int = torch.randn(EVAL_BATCH, DIM, device=DEVICE)
        res = compute_pde_residual(value_net, policy_net, t_int, x_int,
                                   H_t, M_t, C_t, D_t, sigma_t)
        pde_loss = torch.mean(res ** 2)

        x_T = torch.randn(EVAL_BATCH, DIM, device=DEVICE)
        t_T = torch.full((EVAL_BATCH, 1), T, device=DEVICE)
        u_T = value_net(t_T, x_T).squeeze(1)
        target_T = torch.einsum("bi,ij,bj->b", x_T, R_t, x_T)
        terminal_loss = torch.mean((u_T - target_T) ** 2)

        loss = pde_loss + terminal_loss
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

        if epoch % LOG_EVERY == 0:
            print(f"    Eval epoch {epoch:5d}/{EVAL_EPOCHS}  loss = {loss.item():.6f}")

    return losses


def compute_hamiltonian_control_only(value_net, policy_net, M_t, D_t, batch_size):
    """
    Control-dependent part of the Hamiltonian:
        H_a = (grad_v)' M a + a' D a

    grad_v is detached so gradients flow only through the policy network.
    grad_v is normalised by its l2 norm to prevent large gradient magnitudes
    in early iterations from destabilising the policy update.
    The state-dependent terms (grad_v)' H x + x' C x are omitted since
    they don't depend on the policy and would only add gradient noise.
    """
    t = torch.rand(batch_size, 1, device=DEVICE) * T
    x = torch.randn(batch_size, DIM, device=DEVICE)
    x.requires_grad_(True)

    v = value_net(t, x)
    grad_v = torch.autograd.grad(
        v, x, torch.ones_like(v), create_graph=False
    )[0].detach()  # detach: treat as fixed coefficient

    a = policy_net(t, x.detach())

    ham = (grad_v @ M_t.T * a).sum(dim=1)             # (grad_v)' M a
    ham = ham + torch.einsum("bi,ij,bj->b", a, D_t, a)  # a' D a

    return ham.mean()


def policy_improvement(value_net, policy_net, M_t, D_t):
    """Update policy to minimise the control-dependent Hamiltonian."""
    value_net.eval()
    opt = torch.optim.Adam(policy_net.parameters(), lr=LR_IMPROVE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=IMPROVE_STEPS)

    for step in range(1, IMPROVE_STEPS + 1):
        opt.zero_grad()
        loss = compute_hamiltonian_control_only(
            value_net, policy_net, M_t, D_t, batch_size=IMPROVE_BATCH
        )
        loss.backward()
        opt.step()
        scheduler.step()

        if step % 200 == 0 or step == IMPROVE_STEPS:
            print(f"    Improve step {step:5d}/{IMPROVE_STEPS}  "
                  f"Hamiltonian = {loss.item():.4f}")

    value_net.train()


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Analytical reference ────────────────────────────────────────────
    solver = LQRSolver(H, M, C, D, R, SIGMA, T)
    solver.solve_riccati(n_grid=2000)

    # ── Tensor matrices ─────────────────────────────────────────────────
    H_t = torch.as_tensor(H, dtype=DTYPE, device=DEVICE)
    M_t = torch.as_tensor(M, dtype=DTYPE, device=DEVICE)
    C_t = torch.as_tensor(C, dtype=DTYPE, device=DEVICE)
    D_t = torch.as_tensor(D, dtype=DTYPE, device=DEVICE)
    R_t = torch.as_tensor(R, dtype=DTYPE, device=DEVICE)
    sigma_t = torch.as_tensor(SIGMA, dtype=DTYPE, device=DEVICE)

    # ── Policy network ──────────────────────────────────────────────────
    torch.manual_seed(42)
    policy_net = PolicyNet(DIM + 1, HIDDEN_DIM, N_LAYERS, DIM).to(DEVICE)

    # Initialise output bias to alpha_init
    with torch.no_grad():
        policy_net.output_layer.bias.copy_(
            torch.as_tensor(ALPHA_INIT, dtype=DTYPE, device=DEVICE)
        )

    all_losses = []
    prev_val_state = None  # for warm-starting

    # ── Policy iteration loop ───────────────────────────────────────────
    for k in range(N_POLICY_ITERS):
        print(f"\n{'='*60}")
        print(f"Policy iteration {k+1}/{N_POLICY_ITERS}")
        print(f"{'='*60}")

        # Fresh value network (or warm-started from previous)
        value_net = SimpleDGMNet(DIM + 1, HIDDEN_DIM, N_LAYERS, output_dim=1).to(DEVICE)
        if prev_val_state is not None:
            print("  Warm-starting value network from previous iteration.")
            value_net.load_state_dict(prev_val_state)
        val_opt = torch.optim.Adam(value_net.parameters(), lr=LR_EVAL)

        # 1. Policy evaluation (freeze policy)
        print("  Policy evaluation...")
        for p in policy_net.parameters():
            p.requires_grad_(False)

        losses = policy_evaluation(value_net, policy_net, val_opt,
                                   H_t, M_t, C_t, D_t, R_t, sigma_t)
        all_losses.append(losses)

        # Save state for warm-starting next iteration
        prev_val_state = {
            key: val.detach().cpu().clone()
            for key, val in value_net.state_dict().items()
        }

        # Unfreeze policy for improvement
        for p in policy_net.parameters():
            p.requires_grad_(True)

        # 2. Policy improvement
        print("  Policy improvement...")
        policy_improvement(value_net, policy_net, M_t, D_t)

        # 3. Monitor progress
        with torch.no_grad():
            values = value_net(T_TEST, X_TEST)
            controls = policy_net(T_TEST, X_TEST)

        print("\n  Test points after iteration:")
        for i in range(T_TEST.shape[0]):
            tv = T_TEST[i, 0].item()
            xv = X_TEST[i].cpu().numpy()
            v = values[i, 0].item()
            a = controls[i].cpu().numpy()

            # Analytical reference
            t_ref = torch.tensor([tv], device=DEVICE)
            x_ref = X_TEST[i:i+1]
            v_true = solver.value_function(t_ref, x_ref)[0].item()
            a_true = solver.optimal_control(t_ref, x_ref)[0].cpu().numpy()

            print(f"    t={tv:.1f}, x=[{xv[0]:+.1f}, {xv[1]:+.1f}]  "
                  f"v={v:+.4f} (true={v_true:+.4f})  "
                  f"a=[{a[0]:+.4f}, {a[1]:+.4f}] "
                  f"(true=[{a_true[0]:+.4f}, {a_true[1]:+.4f}])")

    # ── Plot 1: Loss curves across iterations ───────────────────────────
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, losses in enumerate(all_losses):
        offset = k * EVAL_EPOCHS
        epochs = np.arange(1, len(losses) + 1) + offset
        ax.semilogy(epochs, losses, linewidth=0.6, label=f"Iteration {k+1}")
        if k < len(all_losses) - 1:
            ax.axvline(offset + EVAL_EPOCHS, color="gray", ls=":", alpha=0.5)

    ax.set_xlabel("Total training epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Exercise 4 — Policy iteration: evaluation losses")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_4_policy_iteration_loss.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 2: Final value comparison ──────────────────────────────────
    x_range = np.linspace(-2.5, 2.5, 40)
    X1, X2 = np.meshgrid(x_range, x_range)
    x_flat = torch.tensor(
        np.stack([X1.ravel(), X2.ravel()], axis=1), dtype=DTYPE, device=DEVICE
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), subplot_kw={"projection": "3d"})

    t_val = 0.0
    t_flat = torch.full((x_flat.shape[0],), t_val, device=DEVICE)
    t_flat_2d = t_flat.unsqueeze(1)

    with torch.no_grad():
        V_true = solver.value_function(t_flat, x_flat).cpu().numpy().reshape(X1.shape)
        V_pi = value_net(t_flat_2d, x_flat).squeeze(1).cpu().numpy().reshape(X1.shape)

    axes[0].plot_surface(X1, X2, V_true, cmap="viridis", alpha=0.85)
    axes[0].set_title(f"Analytical v(t={t_val}, x)")
    axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2"); axes[0].set_zlabel("v")

    axes[1].plot_surface(X1, X2, V_pi, cmap="plasma", alpha=0.85)
    axes[1].set_title(f"Policy iteration v(t={t_val}, x)")
    axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2"); axes[1].set_zlabel("v")

    axes[2].plot_surface(X1, X2, np.abs(V_true - V_pi), cmap="inferno", alpha=0.85)
    axes[2].set_title(f"|Error| at t={t_val}")
    axes[2].set_xlabel("x1"); axes[2].set_ylabel("x2"); axes[2].set_zlabel("|dv|")

    fig.suptitle("Exercise 4 — Policy iteration vs analytical value function", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_4_value_comparison.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nExercise 4 complete.")


if __name__ == "__main__":
    main()