"""
exercise_1_2.py — Monte Carlo validation of the LQR solution.

Verifies the analytical value function from Exercise 1.1 using MC simulation:
  1. Time-step convergence: fix N_MC, vary Δt  → expect O(Δt)
  2. Sample convergence:    fix Δt,  vary N_MC → expect O(1/√N)

Generates:
  - plots/exercise_1_2_timestep_convergence.png
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


def run_timestep_convergence(solver, mc, n_mc=100_000):
    """Fix N_MC and vary the number of time steps."""
    t0, x0 = 0.0, np.array([1.0, 1.0])
    v_true = solver.value_function(
        torch.tensor([t0], device=DEVICE),
        torch.tensor([[1.0, 1.0]], device=DEVICE),
    )[0].item()
    print(f"  True value v(0, [1,1]) = {v_true:.6f}")

    n_steps_list = [1, 5, 10, 50, 100, 500, 1000, 5000]
    errs_exp, errs_imp = [], []

    print(f"\n  {'N':>6}  {'τ':>8}  {'err_expl':>12}  {'err_impl':>12}")
    for N in n_steps_list:
        tau = T / N
        v_exp, _ = mc.estimate_value(t0, x0, N, n_mc, "explicit", seed=0)
        v_imp, _ = mc.estimate_value(t0, x0, N, n_mc, "implicit", seed=0)
        errs_exp.append(abs(v_exp - v_true))
        errs_imp.append(abs(v_imp - v_true))
        print(f"  {N:>6}  {tau:>8.4f}  {errs_exp[-1]:>12.4e}  {errs_imp[-1]:>12.4e}")

    # ── Plot ────────────────────────────────────────────────────────────
    tau_vals = np.array([T / N for N in n_steps_list])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(tau_vals, errs_exp, "o-", label="Explicit Euler")
    ax.loglog(tau_vals, errs_imp, "s-", label="Implicit Euler")

    # Reference slopes
    ref = errs_exp[2] / tau_vals[2]
    ax.loglog(tau_vals, ref * tau_vals, "k--", alpha=0.4, label="O(τ)")
    ax.loglog(tau_vals, ref * tau_vals**2 / tau_vals[2], "k:", alpha=0.4, label="O(τ²)")

    ax.set_xlabel("Time step τ")
    ax.set_ylabel("|v_MC − v_true|")
    ax.set_title(f"Time discretisation convergence  (N_MC = {n_mc:,})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "exercise_1_2_timestep_convergence.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run_mc_convergence(solver, mc, n_steps=5000):
    """Fix the time step and vary N_MC."""
    t0, x0 = 0.0, np.array([1.0, 1.0])
    v_true = solver.value_function(
        torch.tensor([t0], device=DEVICE),
        torch.tensor([[1.0, 1.0]], device=DEVICE),
    )[0].item()
    print(f"  True value v(0, [1,1]) = {v_true:.6f}")

    n_mc_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    errors, std_errors, means = [], [], []

    print(f"\n  {'N_MC':>8}  {'v_MC':>10}  {'std_err':>10}  {'|err|':>10}")
    for n_mc in n_mc_list:
        v_mc, se = mc.estimate_value(t0, x0, n_steps, n_mc, "explicit", seed=0)
        errors.append(abs(v_mc - v_true))
        std_errors.append(se)
        means.append(v_mc)
        print(f"  {n_mc:>8}  {v_mc:>10.5f}  {se:>10.2e}  {errors[-1]:>10.2e}")

    n_arr = np.array(n_mc_list, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log-log error
    ax1.loglog(n_mc_list, errors, "o-", label="|v_MC − v_true|")
    ax1.loglog(n_mc_list, std_errors, "s-", label="Std error")
    scale = errors[2] * np.sqrt(n_mc_list[2])
    ax1.loglog(n_arr, scale / np.sqrt(n_arr), "k--", alpha=0.4, label="O(1/√N)")
    ax1.set_xlabel("N_MC")
    ax1.set_ylabel("Error")
    ax1.set_title(f"MC convergence  (N_steps = {n_steps})")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # Right: estimate with error bars
    ax2.errorbar(n_mc_list, means, yerr=std_errors, fmt="o-", capsize=4,
                 label="MC estimate ± std error")
    ax2.axhline(v_true, color="r", linestyle="--", label="True value")
    ax2.set_xscale("log")
    ax2.set_xlabel("N_MC")
    ax2.set_ylabel("Value estimate")
    ax2.set_title("MC estimate converging to true value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "exercise_1_2_mc_convergence.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    solver = LQRSolver(H, M, C, D, R, SIGMA, T)
    solver.solve_riccati(n_grid=2000)
    mc = LQRMonteCarlo(solver)

    print("\n── Test 1: Time-step convergence ──")
    run_timestep_convergence(solver, mc, n_mc=100_000)

    print("\n── Test 2: MC sample convergence ──")
    run_mc_convergence(solver, mc, n_steps=5000)

    print("\nExercise 1.2 complete.")


if __name__ == "__main__":
    main()
