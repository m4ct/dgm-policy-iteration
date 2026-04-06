"""
exercise_3.py — Deep Galerkin Method for the linear PDE (constant control).

Solves the linear PDE obtained by fixing α = [1, 1]:
    ∂u/∂t + ½ tr(σσ' ∂²u) + (∂u)ᵀHx + (∂u)ᵀMα + xᵀCx + αᵀDα = 0
    u(T, x) = xᵀRx

The network is trained by minimising the composite loss
    R(θ) = R_interior(θ) + λ R_boundary(θ),   λ = 5,
then validated against replicated Monte Carlo benchmarks.

Generates:
  - plots/exercise_3_loss.png
  - plots/exercise_3_loss_components.png
  - plots/exercise_3_error_vs_mc.png
  - plots/exercise_3_surfaces.png

Usage:
    python exercise_3.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DIM, DEVICE, DTYPE, PLOT_DIR, PLOT_DPI
from networks import Net_DGM
from monte_carlo import ConstantControlMC
from lqr_solver import LQRSolver


# ── Hyperparameters ─────────────────────────────────────────────────────
HIDDEN_SIZE = 100
N_DGM_LAYERS = 3
EPOCHS = 10_000
BATCH_SIZE = 512
LR = 1e-3
ALPHA = np.array([1.0, 1.0])
LOG_EVERY = 500
EVAL_INTERVAL = 10
X_RANGE = 3.0
TERMINAL_WEIGHT = 5.0   # upweight boundary to anchor the solution
MC_REPS = 12
MC_STEPS_GRID = (5_000, 10_000, 20_000)

# ── Test point ──────────────────────────────────────────────────────────
T0_TEST = 0.0
X0_TEST = np.array([0.0, 0.0]) 
DIAG_POINTS = [
    # (0.0, np.array([0.0, 0.0])), # removed to avoid redundancy
    (0.0, np.array([1.0, 1.0])),
    (0.5, np.array([0.0, 0.0])),
    (0.5, np.array([1.0, -1.0])),
]
DIAG_N_MC = 50_000
DIAG_REPS = 6


# ── Exact solution via Riccati-type ODE ─────────────────────────────────
class ConstantControlSolver:
    """
    Solve the linear PDE for constant control α analytically.

    Under constant α, the value function has the form
        u(t,x) = xᵀP(t)x + qᵀ(t)x + r(t)
    where P, q, r satisfy coupled ODEs derived from the PDE.

    For this problem (Mα is a constant vector), it simplifies to
        u(t,x) = xᵀP(t)x + 2 pᵀ(t)x + r(t)
    with:
        P' = -(HᵀP + PH) - C,                          P(T) = R
        p' = -(H + PMα)ᵀ ... but since drift is Hx + Mα (constant part),
    we just solve it numerically via scipy.
    """

    def __init__(self, H, M, C, D, R, sigma, T, alpha):
        from scipy.integrate import solve_ivp
        from scipy.interpolate import interp1d

        self.d = H.shape[0]
        self.T = T
        self.alpha = alpha
        self.Ma = M @ alpha
        self.aDa = float(alpha @ D @ alpha)
        ssT = sigma @ sigma.T

        # State: [P11, P12, P22, p1, p2, r] — P symmetric so 3 entries
        def rhs(tau, y):
            P = np.array([[y[0], y[1]], [y[1], y[2]]])
            p = np.array([y[3], y[4]])
            r_val = y[5]

            # dP/dτ = HᵀP + PH + C  (forward in τ = T-t)
            dP = H.T @ P + P @ H + C
            # dp/dτ = Hᵀp + P @ Mα
            dp = H.T @ p + P @ self.Ma
            # dr/dτ = tr(σσᵀP) + 2 pᵀMα + αᵀDα
            dr = np.trace(ssT @ P) + 2.0 * p @ self.Ma + self.aDa

            return [dP[0, 0], dP[0, 1], dP[1, 1], dp[0], dp[1], dr]

        y0 = [R[0, 0], R[0, 1], R[1, 1], 0.0, 0.0, 0.0]
        n_grid = 2000
        tau_span = (0.0, T)
        tau_eval = np.linspace(0, T, n_grid)

        sol = solve_ivp(rhs, tau_span, y0, t_eval=tau_eval,
                        method="RK45", rtol=1e-10, atol=1e-12)
        tau = sol.t
        t_grid = T - tau[::-1]  # back to original time

        P_vals = np.zeros((n_grid, 2, 2))
        p_vals = np.zeros((n_grid, 2))
        r_vals = np.zeros(n_grid)
        for i in range(n_grid):
            j = n_grid - 1 - i  # reverse index
            P_vals[i] = [[sol.y[0, j], sol.y[1, j]], [sol.y[1, j], sol.y[2, j]]]
            p_vals[i] = [sol.y[3, j], sol.y[4, j]]
            r_vals[i] = sol.y[5, j]

        self._P_interp = [
            interp1d(t_grid, P_vals[:, 0, 0], kind="cubic"),
            interp1d(t_grid, P_vals[:, 0, 1], kind="cubic"),
            interp1d(t_grid, P_vals[:, 1, 1], kind="cubic"),
        ]
        self._p_interp = [
            interp1d(t_grid, p_vals[:, 0], kind="cubic"),
            interp1d(t_grid, p_vals[:, 1], kind="cubic"),
        ]
        self._r_interp = interp1d(t_grid, r_vals, kind="cubic")

    def value(self, t_np, x_np):
        """Evaluate u(t, x) for arrays t (N,) and x (N, 2)."""
        t_np = np.clip(t_np, 0.0, self.T)
        P00 = self._P_interp[0](t_np)
        P01 = self._P_interp[1](t_np)
        P11 = self._P_interp[2](t_np)
        p0 = self._p_interp[0](t_np)
        p1 = self._p_interp[1](t_np)
        r = self._r_interp(t_np)

        x0, x1 = x_np[:, 0], x_np[:, 1]
        quad = P00 * x0 ** 2 + 2 * P01 * x0 * x1 + P11 * x1 ** 2
        lin = 2 * (p0 * x0 + p1 * x1)
        return quad + lin + r

    def value_torch(self, t_tensor, x_tensor, device):
        """Torch-compatible wrapper."""
        t_np = t_tensor.detach().cpu().numpy().ravel()
        x_np = x_tensor.detach().cpu().numpy()
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
        vals = self.value(t_np, x_np)
        return torch.as_tensor(vals, dtype=DTYPE, device=device)


def compute_pde_residual(net, t, x, H_t, M_t, C_t, D_t, sigma_t, alpha_t):
    """
    Evaluate the PDE residual at (t, x) for constant control α.
    u(t,x) = net(t,x) directly (no ansatz).
    """
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    u = net(t, x)  # (B, 1)

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


def replicated_mc_benchmark(mc, t0, x0, n_steps, n_rep=MC_REPS, n_mc=100_000, seed0=0):
    """Replicated MC estimate for one time-step count."""
    means = []
    ses = []
    for r in range(n_rep):
        mean_r, se_r = mc.estimate_value(t0, x0, n_steps=n_steps, n_mc=n_mc, seed=seed0 + r)
        means.append(mean_r)
        ses.append(se_r)
    means = np.asarray(means, dtype=float)
    ses = np.asarray(ses, dtype=float)
    rep_mean = float(np.mean(means))
    rep_std = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
    rep_se = rep_std / np.sqrt(len(means)) if len(means) > 1 else 0.0
    mean_path_se = float(np.mean(ses))
    return rep_mean, rep_se, mean_path_se


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Exact reference via ODE ──────────────────────────────────────────
    print("Computing exact ODE reference for constant control...")
    exact_solver = ConstantControlSolver(H, M, C, D, R, SIGMA, T, ALPHA)
    exact_val_primary = exact_solver.value(
        np.array([T0_TEST]), np.array([X0_TEST])
    )[0]
    print(f"  Exact u(0, [1,1]) = {exact_val_primary:.6f}")

    diag_exact = []
    for td, xd in DIAG_POINTS:
        ev = exact_solver.value(np.array([td]), xd.reshape(1, -1))[0]
        diag_exact.append((td, xd, ev))
        print(f"  Exact u({td}, {xd.tolist()}) = {ev:.6f}")

    # ── Monte Carlo reference (for independent cross-check) ─────────────
    print("\nComputing replicated Monte Carlo reference values...")
    mc = ConstantControlMC(H, M, C, D, R, SIGMA, T, ALPHA)
    step_results = []
    for n_steps in MC_STEPS_GRID:
        mean_s, rep_se_s, path_se_s = replicated_mc_benchmark(
            mc, T0_TEST, X0_TEST, n_steps=n_steps, n_rep=MC_REPS, n_mc=100_000,
            seed0=1_000 + n_steps
        )
        step_results.append((n_steps, mean_s, rep_se_s, path_se_s))
        print(f"  n_steps={n_steps:>5d}  mean={mean_s:.6f}  "
              f"rep_se={rep_se_s:.2e}  path_se={path_se_s:.2e}")

    mc_mean = step_results[-1][1]
    rep_se = step_results[-1][2]
    disc_spread = max(m for _, m, _, _ in step_results) - min(m for _, m, _, _ in step_results)
    mc_threshold = max(1.96 * rep_se, 0.5 * disc_spread)
    print(f"  MC reference (finest grid) = {mc_mean:.6f}")
    print(f"  |MC − exact| = {abs(mc_mean - exact_val_primary):.6f}")
    print(f"  MC 95% CI half-width = {1.96 * rep_se:.6f}")

    # Use exact solution as primary reference
    ref_val = exact_val_primary

    # Diagnostic references at additional points
    diag_refs = []
    print("\nDiagnostic-point exact references:")
    for td, xd, ev in diag_exact:
        diag_refs.append((td, xd, ev))
        print(f"  t={td:.2f}, x={xd.tolist()}  exact={ev:.6f}")

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
    net = Net_DGM(
        input_dim=DIM + 1, hidden_size=HIDDEN_SIZE, output_dim=1, n_layers=N_DGM_LAYERS
    ).to(DEVICE)
    optimiser = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS, eta_min=LR * 1e-2
    )

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"\nNet_DGM: {N_DGM_LAYERS} layers × {HIDDEN_SIZE} hidden, "
          f"{n_params:,} parameters")

    # ── Test-point tensors ──────────────────────────────────────────────
    t0_t = torch.tensor([[T0_TEST]], dtype=DTYPE, device=DEVICE)
    x0_t = torch.as_tensor(X0_TEST, dtype=DTYPE, device=DEVICE).unsqueeze(0)

    # ── Training loop ───────────────────────────────────────────────────
    loss_history = []
    pde_history = []
    terminal_history = []
    error_vs_exact = []       # (epoch, |DGM − exact| at primary)
    error_vs_mc = []          # (epoch, |DGM − MC| at primary)
    multi_error_history = []  # (epoch, mean |DGM − exact| over diag points)

    print(f"\nTraining for {EPOCHS} epochs (batch_size={BATCH_SIZE}, λ={TERMINAL_WEIGHT})...")

    for epoch in range(1, EPOCHS + 1):
        optimiser.zero_grad()

        # ── Interior samples ────────────────────────────────────────────
        t_int = torch.rand(BATCH_SIZE, 1, device=DEVICE) * T
        x_int = (2 * torch.rand(BATCH_SIZE, DIM, device=DEVICE) - 1) * X_RANGE

        res = compute_pde_residual(net, t_int, x_int,
                                   H_t, M_t, C_t, D_t, sigma_t, alpha_t)
        pde_loss = torch.mean(res ** 2)

        # ── Terminal condition ──────────────────────────────────────────
        x_T = (2 * torch.rand(BATCH_SIZE, DIM, device=DEVICE) - 1) * X_RANGE
        t_T = torch.full((BATCH_SIZE, 1), T, device=DEVICE)
        u_T = net(t_T, x_T).squeeze(1)
        target_T = torch.einsum("bi,ij,bj->b", x_T, R_t, x_T)
        terminal_loss = torch.mean((u_T - target_T) ** 2)

        loss = pde_loss + TERMINAL_WEIGHT * terminal_loss
        loss.backward()
        optimiser.step()
        scheduler.step()

        loss_history.append(loss.item())
        pde_history.append(pde_loss.item())
        terminal_history.append(terminal_loss.item())

        # ── Periodic diagnostics ────────────────────────────────────────
        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            with torch.no_grad():
                val = net(t0_t, x0_t).item()
                mean_abs = 0.0
                for td, xd, e_ref in diag_refs:
                    t_d = torch.tensor([[td]], dtype=DTYPE, device=DEVICE)
                    x_d = torch.as_tensor(xd, dtype=DTYPE, device=DEVICE).unsqueeze(0)
                    mean_abs += abs(net(t_d, x_d).item() - e_ref)
                mean_abs /= len(diag_refs)

            error_vs_exact.append((epoch, abs(val - ref_val)))
            error_vs_mc.append((epoch, abs(val - mc_mean)))
            multi_error_history.append((epoch, mean_abs))

        if epoch % LOG_EVERY == 0 or epoch == 1:
            with torch.no_grad():
                val = net(t0_t, x0_t).item()
            print(f"  Epoch {epoch:6d}/{EPOCHS}  loss={loss.item():.4e}  "
                  f"pde={pde_loss.item():.4e}  term={terminal_loss.item():.4e}  "
                  f"|err_exact|={abs(val - ref_val):.4e}")

    print("Training complete.")

    # ── Final evaluation ────────────────────────────────────────────────
    with torch.no_grad():
        v_dgm = net(t0_t, x0_t).item()
    print(f"\n  DGM value      = {v_dgm:.6f}")
    print(f"  Exact value    = {ref_val:.6f}")
    print(f"  MC value       = {mc_mean:.6f}")
    print(f"  |DGM − exact|  = {abs(v_dgm - ref_val):.6f}")
    print(f"  |DGM − MC|     = {abs(v_dgm - mc_mean):.6f}")
    print(f"  |MC − exact|   = {abs(mc_mean - ref_val):.6f}")
    print(f"  MC threshold   = ±{mc_threshold:.6f}")

    # ── Diagnostics at all points ───────────────────────────────────────
    print("\n  Final diagnostics at all points:")
    for td, xd, e_ref in diag_refs:
        t_d = torch.tensor([[td]], dtype=DTYPE, device=DEVICE)
        x_d = torch.as_tensor(xd, dtype=DTYPE, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            v_d = net(t_d, x_d).item()
        print(f"    t={td:.2f}, x={xd.tolist()}  DGM={v_d:.4f}  "
            f"exact={e_ref:.4f}  |err|={abs(v_d - e_ref):.4f}")

    # ── Plot 1: Training loss ───────────────────────────────────────────
    print("\nGenerating plots...")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(range(1, EPOCHS + 1), np.log(loss_history), linewidth=0.4, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("log(Loss)")
    ax.set_title(r"Exercise 3 — DGM training loss (constant control $\alpha = [1,1]^\top$)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=1.1)
    path = os.path.join(PLOT_DIR, "exercise_3_loss.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 1b: PDE vs terminal losses ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(range(1, EPOCHS + 1), pde_history, linewidth=0.5,
                label="PDE residual MSE", alpha=0.8)
    ax.semilogy(range(1, EPOCHS + 1), terminal_history, linewidth=0.5,
                label="Terminal-condition MSE", alpha=0.8)
    ax.semilogy(range(1, EPOCHS + 1), loss_history, linewidth=0.5, color="black",
                label=f"Total loss (λ={TERMINAL_WEIGHT})", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss term (log scale)")
    ax.set_title("Exercise 3 — Loss components")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_3_loss_components.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 2: Error vs exact and MC ───────────────────────────────────
    err_epochs = np.array([e for e, _ in error_vs_exact])
    err_exact_vals = np.array([v for _, v in error_vs_exact])
    err_mc_vals = np.array([v for _, v in error_vs_mc])
    multi_err_vals = np.array([v for _, v in multi_error_history])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.semilogy(err_epochs, err_exact_vals, linewidth=0.8, color="steelblue",
                alpha=0.7, label="|DGM − exact| (primary)")
    ax.semilogy(err_epochs, err_mc_vals, linewidth=0.6, color="lightblue",
                alpha=0.5, label="|DGM − MC| (primary)")
    ax.semilogy(err_epochs, multi_err_vals, linewidth=0.9, color="purple",
                alpha=0.7, label="Mean |DGM − exact| (diag pts)")
    ax.axhline(mc_threshold, ls="--", color="red", linewidth=1.5,
               label="MC 95% CI ±")
    ax.axhline(abs(mc_mean - ref_val), ls=":", color="orange", linewidth=1.2,
               label="|MC − exact|")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Absolute error")
    ax.set_title("Exercise 3 — Error vs exact and MC")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_3_error_vs_reference.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Plot 3: Surface comparison ──────────────────────────────────────
    x_range = np.linspace(-2.5, 2.5, 50)
    X1, X2 = np.meshgrid(x_range, x_range)
    x_flat_np = np.stack([X1.ravel(), X2.ravel()], axis=1).astype(np.float64)
    x_flat = torch.as_tensor(x_flat_np, dtype=DTYPE, device=DEVICE)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={"projection": "3d"})

    for col, t_val in enumerate([0.0, 0.5]):
        t_flat = torch.full((x_flat.shape[0], 1), t_val, dtype=DTYPE, device=DEVICE)
        t_np = np.full(x_flat_np.shape[0], t_val)

        with torch.no_grad():
            V_dgm = net(t_flat, x_flat).squeeze(1).cpu().numpy().reshape(X1.shape)
        V_exact = exact_solver.value(t_np, x_flat_np).reshape(X1.shape)

        for row, (V, label, cmap) in enumerate([
            (V_exact, "Exact", "viridis"),
            (V_dgm, "DGM", "plasma"),
        ]):
            ax = axes[row, col]
            ax.plot_surface(X1, X2, V, cmap=cmap, alpha=0.85)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("u")
            ax.set_title(f"{label}  u(t={t_val}, x)")

    fig.suptitle("Exercise 3 — Exact vs DGM value surfaces", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_3_surfaces.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    print("\nExercise 3 complete.")


if __name__ == "__main__":
    main()