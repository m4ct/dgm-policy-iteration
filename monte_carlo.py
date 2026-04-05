"""
monte_carlo.py — Monte Carlo simulation of the controlled LQR SDE.

Provides explicit and implicit Euler schemes for:
  1. Optimally-controlled SDE  (control from Riccati solution)
  2. Constant-control SDE      (fixed α, used in Exercise 3)
"""

import numpy as np
import torch

from config import DEVICE, DTYPE


class LQRMonteCarlo:
    """
    Monte Carlo estimator for the LQR value function under optimal control.

    With a*(t,x) = −D⁻¹ M' S(t) x plugged in, the closed-loop SDE is
        dX = (H − M D⁻¹ M' S(t)) X dt + σ dW.
    """

    def __init__(self, solver):
        self.solver = solver
        self.d = solver.d

        # Pre-convert matrices to tensors once
        self._H = torch.as_tensor(solver.H, dtype=DTYPE, device=DEVICE)
        self._M = torch.as_tensor(solver.M, dtype=DTYPE, device=DEVICE)
        self._C = torch.as_tensor(solver.C, dtype=DTYPE, device=DEVICE)
        self._D = torch.as_tensor(solver.D, dtype=DTYPE, device=DEVICE)
        self._R = torch.as_tensor(solver.R, dtype=DTYPE, device=DEVICE)
        self._sig = torch.as_tensor(solver.sigma, dtype=DTYPE, device=DEVICE)
        self._DinvMT = torch.as_tensor(
            solver.D_inv @ solver.M.T, dtype=DTYPE, device=DEVICE
        )

    def _get_S_all(self, times):
        """Evaluate S at all grid points, return as Tensor."""
        S_np = self.solver.get_S(times)
        return torch.as_tensor(S_np, dtype=DTYPE, device=DEVICE)

    def simulate_explicit(self, t0, x0, n_steps, n_mc, seed=None):
        """Explicit Euler with left-endpoint running cost."""
        if seed is not None:
            torch.manual_seed(seed)

        tau = (self.solver.T - t0) / n_steps
        times = np.linspace(t0, self.solver.T, n_steps + 1)
        S_all = self._get_S_all(times)

        X = torch.as_tensor(x0, dtype=DTYPE, device=DEVICE).expand(n_mc, -1).clone()
        cost = torch.zeros(n_mc, device=DEVICE)
        sqrt_tau = np.sqrt(tau)

        for n in range(n_steps):
            S_n = S_all[n]
            A = self._H - self._M @ self._DinvMT @ S_n

            a_n = -(self._DinvMT @ S_n @ X.T).T
            cost += tau * (
                torch.einsum("bi,ij,bj->b", X, self._C, X) +
                torch.einsum("bi,ij,bj->b", a_n, self._D, a_n)
            )

            dW = torch.randn(n_mc, self.d, device=DEVICE) * sqrt_tau
            X = X + tau * (X @ A.T) + (self._sig @ dW.T).T

        cost += torch.einsum("bi,ij,bj->b", X, self._R, X)
        return cost.cpu().numpy()

    def simulate_implicit(self, t0, x0, n_steps, n_mc, seed=None):
        """Implicit Euler with right-endpoint running cost."""
        if seed is not None:
            torch.manual_seed(seed)

        tau = (self.solver.T - t0) / n_steps
        times = np.linspace(t0, self.solver.T, n_steps + 1)
        S_all = self._get_S_all(times)

        X = torch.as_tensor(x0, dtype=DTYPE, device=DEVICE).expand(n_mc, -1).clone()
        cost = torch.zeros(n_mc, device=DEVICE)
        I = torch.eye(self.d, device=DEVICE)
        sqrt_tau = np.sqrt(tau)

        for n in range(n_steps):
            S_np1 = S_all[n + 1]
            A = self._H - self._M @ self._DinvMT @ S_np1
            sys_mat = I - tau * A

            dW = torch.randn(n_mc, self.d, device=DEVICE) * sqrt_tau
            rhs = X + (self._sig @ dW.T).T
            X = torch.linalg.solve(sys_mat, rhs.T).T

            a_np1 = -(self._DinvMT @ S_np1 @ X.T).T
            cost += tau * (
                torch.einsum("bi,ij,bj->b", X, self._C, X) +
                torch.einsum("bi,ij,bj->b", a_np1, self._D, a_np1)
            )

        cost += torch.einsum("bi,ij,bj->b", X, self._R, X)
        return cost.cpu().numpy()

    def estimate_value(self, t0, x0, n_steps, n_mc, method="explicit", seed=None):
        """Return (mean, standard_error)."""
        sim = self.simulate_explicit if method == "explicit" else self.simulate_implicit
        costs = sim(t0, x0, n_steps, n_mc, seed)
        return float(np.mean(costs)), float(np.std(costs) / np.sqrt(n_mc))


class ConstantControlMC:
    """
    Monte Carlo for constant control α (used in Exercise 3).

    dX = (HX + Mα) dt + σ dW
    J  = E[∫ (X'CX + α'Dα) dt + X_T' R X_T]
    """

    def __init__(self, H, M, C, D, R, sigma, T, alpha):
        self._H = torch.as_tensor(H, dtype=DTYPE, device=DEVICE)
        self._M = torch.as_tensor(M, dtype=DTYPE, device=DEVICE)
        self._C = torch.as_tensor(C, dtype=DTYPE, device=DEVICE)
        self._D = torch.as_tensor(D, dtype=DTYPE, device=DEVICE)
        self._R = torch.as_tensor(R, dtype=DTYPE, device=DEVICE)
        self._sig = torch.as_tensor(sigma, dtype=DTYPE, device=DEVICE)
        self._alpha = torch.as_tensor(alpha, dtype=DTYPE, device=DEVICE)
        self.T = T
        self.d = H.shape[0]

    def estimate_value(self, t0, x0, n_steps, n_mc, method="explicit", seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        tau = (self.T - t0) / n_steps
        sqrt_tau = np.sqrt(tau)
        Ma = (self._M @ self._alpha).unsqueeze(0)  # (1, d)
        aDa = self._alpha @ self._D @ self._alpha   # scalar

        X = torch.as_tensor(x0, dtype=DTYPE, device=DEVICE).expand(n_mc, -1).clone()
        cost = torch.zeros(n_mc, device=DEVICE)

        if method == "explicit":
            for _ in range(n_steps):
                cost += tau * (torch.einsum("bi,ij,bj->b", X, self._C, X) + aDa)
                dW = torch.randn(n_mc, self.d, device=DEVICE) * sqrt_tau
                X = X + tau * (X @ self._H.T + Ma) + (self._sig @ dW.T).T
        else:
            I = torch.eye(self.d, device=DEVICE)
            sys_mat = I - tau * self._H
            for _ in range(n_steps):
                dW = torch.randn(n_mc, self.d, device=DEVICE) * sqrt_tau
                rhs = X + tau * Ma + (self._sig @ dW.T).T
                X = torch.linalg.solve(sys_mat, rhs.T).T
                cost += tau * (torch.einsum("bi,ij,bj->b", X, self._C, X) + aDa)

        cost += torch.einsum("bi,ij,bj->b", X, self._R, X)
        costs = cost.cpu().numpy()
        return float(np.mean(costs)), float(np.std(costs) / np.sqrt(n_mc))
