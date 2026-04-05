"""
lqr_solver.py — Analytical LQR solution via Riccati ODE.

Solves
    min  J(t,x) = E[ ∫_t^T (X'CX + a'Da) ds + X_T' R X_T ]
    s.t. dX = (HX + Ma) dt + σ dW,  X_t = x.

Value function:  v(t,x) = x' S(t) x + ∫_t^T tr(σσ' S(r)) dr
Optimal control: a*(t,x) = -D⁻¹ M' S(t) x

where S solves the matrix Riccati ODE backwards from S(T) = R.
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class LQRSolver:
    """Solve the LQR problem and provide value / control evaluators."""

    def __init__(self, H, M, C, D, R, sigma, T):
        self.H = np.asarray(H, dtype=np.float64)
        self.M = np.asarray(M, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        self.D = np.asarray(D, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.T = float(T)
        self.d = self.H.shape[0]

        self.D_inv = np.linalg.inv(self.D)
        self.MD_invMT = self.M @ self.D_inv @ self.M.T

        self._S_interp = None
        self._integral_interp = None

    # ── Riccati ODE ─────────────────────────────────────────────────────

    def _riccati_rhs(self, _tau, S_flat):
        """RHS in forward-time variable τ = T − t."""
        S = S_flat.reshape(self.d, self.d)
        dS = 2.0 * self.H.T @ S - S @ self.MD_invMT @ S + self.C
        return dS.ravel()

    def solve_riccati(self, n_grid=2000):
        """
        Solve the Riccati ODE on a uniform grid of `n_grid` points and
        build cubic interpolators for S(t) and the integral correction.

        Returns
        -------
        S_values : ndarray, shape (n_grid, d, d)
        """
        time_grid = np.linspace(0.0, self.T, n_grid)
        tau_grid = self.T - time_grid  # decreasing → flip for solver

        sol = solve_ivp(
            self._riccati_rhs,
            (0.0, self.T),
            self.R.ravel(),
            t_eval=tau_grid[::-1],
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"Riccati solve failed: {sol.message}")

        S_values = sol.y[:, ::-1].T.reshape(-1, self.d, self.d)
        S_values = 0.5 * (S_values + np.transpose(S_values, (0, 2, 1)))

        # Interpolators for each S_{ij}(t)
        kind = "cubic" if n_grid >= 4 else "linear"
        self._S_interp = [
            [
                interp1d(time_grid, S_values[:, i, j], kind=kind,
                         bounds_error=False, fill_value="extrapolate")
                for j in range(self.d)
            ]
            for i in range(self.d)
        ]

        # Integral correction ∫_t^T tr(σσ' S(r)) dr  (backward trapezoid)
        ss = self.sigma @ self.sigma.T
        trace_vals = np.array([np.trace(ss @ S) for S in S_values])
        integral = np.zeros(n_grid)
        for i in range(n_grid - 2, -1, -1):
            dt = time_grid[i + 1] - time_grid[i]
            integral[i] = integral[i + 1] + 0.5 * dt * (trace_vals[i] + trace_vals[i + 1])

        self._integral_interp = interp1d(
            time_grid, integral, kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )
        return S_values

    # ── Evaluators ──────────────────────────────────────────────────────

    def get_S(self, t):
        """
        Evaluate S(t).

        Parameters
        ----------
        t : ndarray or Tensor, any shape

        Returns
        -------
        S : same type as input, shape (*t.shape, d, d)
        """
        if self._S_interp is None:
            raise RuntimeError("Call solve_riccati() first.")

        is_torch = isinstance(t, torch.Tensor)
        t_np = t.detach().cpu().numpy() if is_torch else np.asarray(t, dtype=np.float64)
        t_flat = t_np.ravel()

        S = np.zeros((len(t_flat), self.d, self.d), dtype=np.float64)
        for i in range(self.d):
            for j in range(self.d):
                S[:, i, j] = self._S_interp[i][j](t_flat)
        S = S.reshape(*t_np.shape, self.d, self.d)

        if is_torch:
            return torch.as_tensor(S, dtype=t.dtype, device=t.device)
        return S

    def value_function(self, t, x):
        """
        v(t, x) = x' S(t) x + ∫_t^T tr(σσ' S(r)) dr.

        Parameters
        ----------
        t : Tensor, shape (B,)
        x : Tensor, shape (B, d)

        Returns
        -------
        v : Tensor, shape (B,)
        """
        S = self.get_S(t)                          # (B, d, d)
        xSx = torch.einsum("bi,bij,bj->b", x, S, x)

        integ = self._integral_interp(t.detach().cpu().numpy())
        integ = torch.as_tensor(integ, dtype=t.dtype, device=t.device)
        return xSx + integ

    def optimal_control(self, t, x):
        """
        a*(t, x) = −D⁻¹ M' S(t) x.

        Parameters
        ----------
        t : Tensor, shape (B,)
        x : Tensor, shape (B, d)

        Returns
        -------
        a : Tensor, shape (B, d)
        """
        S = self.get_S(t)                          # (B, d, d)
        neg_DinvMT = torch.as_tensor(
            -self.D_inv @ self.M.T, dtype=t.dtype, device=t.device
        )
        Sx = torch.einsum("bij,bj->bi", S, x)     # (B, d)
        return torch.einsum("ij,bj->bi", neg_DinvMT, Sx)
