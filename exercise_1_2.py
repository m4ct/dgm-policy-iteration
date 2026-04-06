"""
exercise_1_2.py — Monte Carlo validation of the LQR solution.

Verifies the analytical value function from Exercise 1.1 using MC simulation:
  1. Time-step convergence: fix N_MC, vary Δt  → expect O(Δt)
  2. Sample convergence:    fix Δt,  vary N_MC → expect O(1/√N)

Improvements over the basic version:
  - repeated independent runs for each configuration,
  - RMSE / mean-absolute-error diagnostics rather than a single noisy error,
  - 95% confidence intervals for replicated mean estimates,
  - exact coursework grids for N_steps and N_MC,
  - higher-resolution Riccati interpolation to reduce reference-solver error.

Generates:
  - plots/exercise_1_2_timestep_convergence.png
  - plots/exercise_1_2_timestep_estimates.png
  - plots/exercise_1_2_mc_convergence.png

Usage:
    python exercise_1_2.py
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import H, M, C, D, R, SIGMA, T, DEVICE, PLOT_DIR, PLOT_DPI
from lqr_solver import LQRSolver
from monte_carlo import LQRMonteCarlo


# ── Reproducibility / experiment controls ──────────────────────────────
MASTER_SEED = 12345
RICCATI_N_GRID = 5000
TIMESTEP_N_MC = 100_000
TIMESTEP_REPS = 8
MC_REPS = 20
CI_Z = 1.96  # 95% Gaussian confidence interval multiplier


def _true_value(solver, t0, x0):
    """Analytical reference value v(t0, x0)."""
    return solver.value_function(
        torch.tensor([t0], device=DEVICE),
        torch.as_tensor(x0, dtype=torch.float32, device=DEVICE).unsqueeze(0),
    )[0].item()


def _seed_sequence(n, offset=0):
    """Deterministic list of independent integer seeds."""
    rng = np.random.default_rng(MASTER_SEED + offset)
    return rng.integers(0, 2**31 - 1, size=n, dtype=np.int64).tolist()


def _simulate_costs(mc, t0, x0, n_steps, n_mc, method, seed):
    """Return pathwise costs for one Monte Carlo run."""
    if method == "explicit":
        return mc.simulate_explicit(t0, x0, n_steps, n_mc, seed=seed)
    if method == "implicit":
        return mc.simulate_implicit(t0, x0, n_steps, n_mc, seed=seed)
    raise ValueError(f"Unknown method: {method}")


def _replicated_statistics(mc, t0, x0, n_steps, n_mc, v_true, method, n_rep, seed_offset=0):
    """
    Run repeated independent Monte Carlo estimates and return stable diagnostics.

    Returns a dictionary with:
      - estimates         : repeated MC estimates
      - std_errors        : repeated per-run standard errors
      - mean_estimate     : average of the repeated MC estimates
      - mean_abs_error    : average absolute error across repeated runs
      - rmse              : root-mean-square error across repeated runs
      - mean_std_error    : average standard error across repeated runs
      - ci_half_width     : 95% CI half-width for the replicated mean estimate
    """
    estimates = []
    std_errors = []

    for seed in _seed_sequence(n_rep, offset=seed_offset):
        costs = _simulate_costs(mc, t0, x0, n_steps, n_mc, method=method, seed=int(seed))
        estimate = float(np.mean(costs))
        std_error = float(np.std(costs, ddof=1) / np.sqrt(n_mc))
        estimates.append(estimate)
        std_errors.append(std_error)

    estimates = np.asarray(estimates, dtype=float)
    std_errors = np.asarray(std_errors, dtype=float)
    errors = estimates - v_true

    mean_estimate = float(np.mean(estimates))
    mean_abs_error = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_std_error = float(np.mean(std_errors))
    rep_std = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
    ci_half_width = float(CI_Z * rep_std / np.sqrt(len(estimates))) if len(estimates) > 1 else 0.0

    return {
        "estimates": estimates,
        "std_errors": std_errors,
        "mean_estimate": mean_estimate,
        "mean_abs_error": mean_abs_error,
        "rmse": rmse,
        "mean_std_error": mean_std_error,
        "ci_half_width": ci_half_width,
    }


def run_timestep_convergence(solver, mc, n_mc=TIMESTEP_N_MC, n_rep=TIMESTEP_REPS):
    """
    Fix N_MC and vary the number of time steps.

    We report replicated RMSE and replicated mean absolute error rather than a
    single realised absolute error, which is much noisier on a log-log plot.
    """
    t0, x0 = 0.0, np.array([1.0, 1.0], dtype=float)
    v_true = _true_value(solver, t0, x0)
    print(f"  True value v(0, [1,1]) = {v_true:.6f}")
    print("  Error metric in plot: replicated RMSE over independent MC runs")

    # Match the coursework grid exactly.
    n_steps_list = [1, 10, 50, 100, 500, 1000, 5000]
    tau_vals = np.array([T / N for N in n_steps_list], dtype=float)

    exp_rmse, imp_rmse = [], []
    exp_mean, imp_mean = [], []
    exp_ci, imp_ci = [], []

    print(
        f"\n  {'N':>6}  {'τ':>8}  {'RMSE expl':>12}  {'RMSE impl':>12}"
        f"  {'mean expl':>12}  {'mean impl':>12}"
    )
    for idx, N in enumerate(n_steps_list):
        tau = T / N
        stats_exp = _replicated_statistics(
            mc, t0, x0, N, n_mc, v_true, method="explicit", n_rep=n_rep, seed_offset=10_000 + idx
        )
        stats_imp = _replicated_statistics(
            mc, t0, x0, N, n_mc, v_true, method="implicit", n_rep=n_rep, seed_offset=20_000 + idx
        )

        exp_rmse.append(stats_exp["rmse"])
        imp_rmse.append(stats_imp["rmse"])
        exp_mean.append(stats_exp["mean_estimate"])
        imp_mean.append(stats_imp["mean_estimate"])
        exp_ci.append(stats_exp["ci_half_width"])
        imp_ci.append(stats_imp["ci_half_width"])

        print(
            f"  {N:>6}  {tau:>8.4f}  {stats_exp['rmse']:>12.4e}  {stats_imp['rmse']:>12.4e}"
            f"  {stats_exp['mean_estimate']:>12.6f}  {stats_imp['mean_estimate']:>12.6f}"
        )

    exp_rmse = np.asarray(exp_rmse)
    imp_rmse = np.asarray(imp_rmse)
    exp_mean = np.asarray(exp_mean)
    imp_mean = np.asarray(imp_mean)
    exp_ci = np.asarray(exp_ci)
    imp_ci = np.asarray(imp_ci)

    fig, ax = plt.subplots(figsize=(8.0, 5.4))

    # Main report plot: keep only the two RMSE curves and a first-order guide.
    ax.loglog(
        tau_vals,
        exp_rmse,
        "o-",
        color="#1f77b4",
        linewidth=2.2,
        markersize=7,
        label="Explicit Euler RMSE",
    )
    ax.loglog(
        tau_vals,
        imp_rmse,
        "s-",
        color="#ff7f0e",
        linewidth=2.2,
        markersize=6.5,
        label="Implicit Euler RMSE",
    )

    ref_idx = min(3, len(tau_vals) - 1)
    ref = exp_rmse[ref_idx] / tau_vals[ref_idx]
    ax.loglog(
        tau_vals,
        ref * tau_vals,
        "--",
        color="0.35",
        linewidth=1.8,
        label="First-order reference O(tau)",
    )

    ax.set_xlabel("Time step tau")
    ax.set_ylabel("Replicated RMSE")
    ax.set_title(f"Time-step convergence (N_MC = {n_mc:,}, reps = {n_rep})")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    fig.tight_layout(pad=1.1)
    path = os.path.join(PLOT_DIR, "exercise_1_2_timestep_convergence.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Keep the estimate-against-truth picture as a separate diagnostic figure.
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.errorbar(
        tau_vals,
        exp_mean,
        yerr=exp_ci,
        fmt="o-",
        capsize=4,
        color="#1f77b4",
        linewidth=2.0,
        markersize=7,
        label="Explicit Euler mean +- 95% CI",
    )
    ax.errorbar(
        tau_vals,
        imp_mean,
        yerr=imp_ci,
        fmt="s-",
        capsize=4,
        color="#ff7f0e",
        linewidth=2.0,
        markersize=6.5,
        label="Implicit Euler mean +- 95% CI",
    )
    ax.axhline(v_true, color="0.35", linestyle="--", linewidth=1.8, label="True value")
    ax.set_xscale("log")
    ax.set_xlabel("Time step tau")
    ax.set_ylabel("Replicated mean estimate")
    ax.set_title("Timestep bias diagnostic")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.10)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout(pad=1.1)
    path = os.path.join(PLOT_DIR, "exercise_1_2_timestep_estimates.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_mc_convergence(solver, mc, n_steps=5000, n_rep=MC_REPS):
    """
    Fix the time step and vary N_MC.

    Uses repeated independent runs for each N_MC so that the convergence plot is
    statistically much cleaner than plotting one realised absolute error.
    """
    t0, x0 = 0.0, np.array([1.0, 1.0], dtype=float)
    v_true = _true_value(solver, t0, x0)
    print(f"  True value v(0, [1,1]) = {v_true:.6f}")
    print("  Error metric in left plot: replicated RMSE over independent MC runs")

    # Match the coursework grid exactly.
    n_mc_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

    rmse_list, mae_list = [], []
    mean_std_errors = []
    mean_estimates, ci_half_widths = [], []

    print(
        f"\n  {'N_MC':>8}  {'mean(v_MC)':>12}  {'RMSE':>12}  {'mean |err|':>12}"
        f"  {'mean SE':>12}  {'95% CI half':>12}"
    )
    for idx, n_mc in enumerate(n_mc_list):
        stats = _replicated_statistics(
            mc, t0, x0, n_steps, n_mc, v_true, method="explicit", n_rep=n_rep, seed_offset=30_000 + idx
        )
        rmse_list.append(stats["rmse"])
        mae_list.append(stats["mean_abs_error"])
        mean_std_errors.append(stats["mean_std_error"])
        mean_estimates.append(stats["mean_estimate"])
        ci_half_widths.append(stats["ci_half_width"])

        print(
            f"  {n_mc:>8}  {stats['mean_estimate']:>12.6f}  {stats['rmse']:>12.4e}"
            f"  {stats['mean_abs_error']:>12.4e}  {stats['mean_std_error']:>12.4e}"
            f"  {stats['ci_half_width']:>12.4e}"
        )

    n_arr = np.asarray(n_mc_list, dtype=float)
    rmse_arr = np.asarray(rmse_list, dtype=float)
    mae_arr = np.asarray(mae_list, dtype=float)
    se_arr = np.asarray(mean_std_errors, dtype=float)
    mean_arr = np.asarray(mean_estimates, dtype=float)
    ci_arr = np.asarray(ci_half_widths, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.8))

    # Left: log-log replicated error.
    ax1.loglog(n_mc_list, rmse_arr, "o-", label="Replicated RMSE")
    ax1.loglog(n_mc_list, mae_arr, "s-", label="Replicated mean abs error")
    ax1.loglog(n_mc_list, se_arr, "^-", label="Average MC std error")

    ref_idx = min(2, len(n_arr) - 1)
    scale = rmse_arr[ref_idx] * np.sqrt(n_arr[ref_idx])
    ax1.loglog(n_arr, scale / np.sqrt(n_arr), "k--", alpha=0.45, label="O(1/√N)")
    ax1.set_xlabel("N_MC")
    ax1.set_ylabel("Error")
    ax1.set_title(f"MC convergence  (N_steps = {n_steps}, reps = {n_rep})")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    # Right: replicated mean estimate with 95% CI.
    ax2.errorbar(
        n_mc_list,
        mean_arr,
        yerr=ci_arr,
        fmt="o-",
        capsize=4,
        label="Replicated mean ± 95% CI",
    )
    ax2.axhline(v_true, color="r", linestyle="--", label="True value")
    ax2.set_xscale("log")
    ax2.set_xlabel("N_MC")
    ax2.set_ylabel("Value estimate")
    ax2.set_title("MC estimate converging to the analytical value")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout(pad=1.1, w_pad=2.0)
    path = os.path.join(PLOT_DIR, "exercise_1_2_mc_convergence.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    solver = LQRSolver(H, M, C, D, R, SIGMA, T)
    solver.solve_riccati(n_grid=RICCATI_N_GRID)
    mc = LQRMonteCarlo(solver)

    print("\n── Test 1: Time-step convergence ──")
    run_timestep_convergence(solver, mc, n_mc=TIMESTEP_N_MC, n_rep=TIMESTEP_REPS)

    print("\n── Test 2: MC sample convergence ──")
    run_mc_convergence(solver, mc, n_steps=5000, n_rep=MC_REPS)

    print("\nExercise 1.2 complete.")


if __name__ == "__main__":
    main()
