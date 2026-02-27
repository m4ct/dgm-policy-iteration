"""
Optimiser comparison for Exercise 2.1
Supervised learning of LQR value function using Net_DGM.

Tests:
    - Adam
    - AdamW
    - AMSGrad
    - SGD + Momentum
    - RMSprop

Produces:
    - optimiser_comparison.png
    - optimiser_test_mse.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# LQRSolver
# ============================================================

class LQRSolver:

    def __init__(self, H, M, C, D, R, sigma, T):
        self.H = np.array(H, dtype=float)
        self.M = np.array(M, dtype=float)
        self.C = np.array(C, dtype=float)
        self.D = np.array(D, dtype=float)
        self.R = np.array(R, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.T = float(T)

        self.d = self.H.shape[0]
        self.D_inv = np.linalg.inv(self.D)
        self.MD_invMT = self.M @ self.D_inv @ self.M.T

    def _riccati_rhs(self, tau, S_flat):
        S = S_flat.reshape(self.d, self.d)
        dS = 2.0 * self.H.T @ S - S @ self.MD_invMT @ S + self.C
        return dS.reshape(-1)

    def solve_riccati(self, time_grid):

        tau_grid = self.T - time_grid

        sol = solve_ivp(
            self._riccati_rhs,
            (0.0, self.T),
            self.R.reshape(-1),
            t_eval=tau_grid[::-1],
            rtol=1e-8,
            atol=1e-10
        )

        S_values = sol.y[:, ::-1].T.reshape(-1, self.d, self.d)
        S_values = 0.5 * (S_values + np.transpose(S_values, (0, 2, 1)))

        self.S_interp = [
            [
                interp1d(time_grid, S_values[:, i, j], kind="cubic",
                         bounds_error=False, fill_value="extrapolate")
                for j in range(self.d)
            ]
            for i in range(self.d)
        ]

        ss = self.sigma @ self.sigma.T
        trace_vals = np.array([np.trace(ss @ S) for S in S_values])

        integral = np.zeros(len(time_grid))
        for i in range(len(time_grid) - 2, -1, -1):
            dt = time_grid[i + 1] - time_grid[i]
            integral[i] = integral[i + 1] + 0.5 * dt * (trace_vals[i] + trace_vals[i + 1])

        self.integral_interp = interp1d(
            time_grid, integral,
            bounds_error=False,
            fill_value="extrapolate"
        )

    def _get_S(self, t):
        t_np = t.detach().cpu().numpy()
        S_out = np.zeros((len(t_np), self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                S_out[:, i, j] = self.S_interp[i][j](t_np)
        return torch.tensor(S_out, dtype=t.dtype, device=t.device)

    def value_function(self, t_batch, x_batch):

        S = self._get_S(t_batch)
        x = x_batch.squeeze(1)

        xSx = torch.sum((x.unsqueeze(1) @ S).squeeze(1) * x, dim=1)

        integral = torch.tensor(
            self.integral_interp(t_batch.detach().cpu().numpy()),
            dtype=t_batch.dtype,
            device=t_batch.device
        )

        return (xSx + integral).unsqueeze(1)


# ============================================================
# Net_DGM
# ============================================================

class Net_DGM(nn.Module):

    def __init__(self, input_dim, hidden_size, output_dim=1, num_layers=3):
        super().__init__()

        self.initial = nn.Linear(input_dim, hidden_size)
        self.num_layers = num_layers

        for l in range(num_layers):
            setattr(self, f'Ug_{l}', nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f'Wg_{l}', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'Uz_{l}', nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f'Wz_{l}', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'Ur_{l}', nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f'Wr_{l}', nn.Linear(hidden_size, hidden_size))
            setattr(self, f'Uh_{l}', nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f'Wh_{l}', nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, output_dim)

    def forward(self, t, x):

        if t.dim() == 1:
            t = t.unsqueeze(1)

        z = torch.cat([t, x], dim=1)
        S = torch.tanh(self.initial(z))

        for l in range(self.num_layers):
            G = torch.sigmoid(getattr(self, f'Ug_{l}')(z) + getattr(self, f'Wg_{l}')(S))
            Z = torch.sigmoid(getattr(self, f'Uz_{l}')(z) + getattr(self, f'Wz_{l}')(S))
            R = torch.sigmoid(getattr(self, f'Ur_{l}')(z) + getattr(self, f'Wr_{l}')(S))
            H = torch.tanh(getattr(self, f'Uh_{l}')(z) + getattr(self, f'Wh_{l}')(S * R))
            S = (1 - G) * H + Z * S

        return self.output(S)


# ============================================================
# Problem Setup
# ============================================================

H     = np.array([[0.5, 0.1], [0.1, 0.3]])
M     = np.array([[1.0, 0.5], [0.0, 1.0]])
C     = np.array([[2.0, 0.5], [0.5, 1.0]])
D     = np.array([[2.0, 0.0], [0.0, 1.0]])
R_mat = np.array([[1.0, 0.0], [0.0, 2.0]])
sigma = np.array([[0.3, 0.1], [0.0, 0.2]])
T     = 1.0
d     = 2

solver = LQRSolver(H, M, C, D, R_mat, sigma, T)
time_grid = np.linspace(0.0, T, 2000)
solver.solve_riccati(time_grid)

# ============================================================
# Training Setup
# ============================================================

HIDDEN_SIZE = 100
N_DATA = 5000
N_EPOCHS = 2000
LR = 1e-3
N_TEST = 30000


def sample_batch(n):

    t_np = np.random.uniform(0.0, T, size=(n,)).astype(np.float32)
    x_np = np.random.uniform(-3.0, 3.0, size=(n, d)).astype(np.float32)

    t_torch = torch.from_numpy(t_np)
    x_torch = torch.from_numpy(x_np).unsqueeze(1)

    with torch.no_grad():
        v_torch = solver.value_function(t_torch, x_torch)

    return (
        t_torch.to(device),
        torch.from_numpy(x_np).to(device),
        v_torch.to(device)
    )


# ============================================================
# Training Loop
# ============================================================

def train(optimiser_name):

    net = Net_DGM(1 + d, HIDDEN_SIZE).to(device)

    if optimiser_name == "Adam":
        optimiser = torch.optim.Adam(net.parameters(), lr=LR)
    elif optimiser_name == "AdamW":
        optimiser = torch.optim.AdamW(net.parameters(), lr=LR)
    elif optimiser_name == "AMSGrad":
        optimiser = torch.optim.Adam(net.parameters(), lr=LR, amsgrad=True)
    elif optimiser_name == "SGD":
        optimiser = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    elif optimiser_name == "RMSprop":
        optimiser = torch.optim.RMSprop(net.parameters(), lr=LR)

    losses = []

    for epoch in range(N_EPOCHS):

        t_b, x_b, v_target = sample_batch(N_DATA)

        optimiser.zero_grad()
        v_pred = net(t_b, x_b)
        loss = torch.mean((v_pred - v_target) ** 2)
        loss.backward()
        optimiser.step()

        losses.append(loss.item())

    # Test error
    net.eval()
    t_test, x_test, v_true = sample_batch(N_TEST)

    with torch.no_grad():
        v_pred = net(t_test, x_test)

    mse = torch.mean((v_pred - v_true) ** 2).item()

    return losses, mse


# ============================================================
# Run Experiment
# ============================================================

optimisers = ["Adam", "AdamW", "AMSGrad", "SGD", "RMSprop"]

all_losses = {}
final_mse = {}

for opt in optimisers:
    print("Training with", opt)
    losses, mse = train(opt)
    all_losses[opt] = losses
    final_mse[opt] = mse
    print(f"{opt} final test MSE: {mse:.4e}")

# ============================================================
# Plot Convergence
# ============================================================

plt.figure(figsize=(9,6))
for opt in optimisers:
    plt.semilogy(all_losses[opt], label=opt)

plt.xlabel("Epoch")
plt.ylabel("Training MSE (log scale)")
plt.title("Optimiser Comparison — LQR Value Function Learning")
plt.legend()
plt.grid(True, which="both", alpha=0.4)
plt.tight_layout()
plt.savefig("optimiser_comparison.png", dpi=150)
plt.close()

# ============================================================
# Plot Final Test Error
# ============================================================

plt.figure(figsize=(8,5))
plt.bar(optimisers, [final_mse[o] for o in optimisers])
plt.yscale("log")
plt.ylabel("Test MSE (log scale)")
plt.title("Final Test Error Comparison")
plt.tight_layout()
plt.savefig("optimiser_test_mse.png", dpi=150)
plt.close()