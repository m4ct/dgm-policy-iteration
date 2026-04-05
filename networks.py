"""
networks.py — Neural network architectures for deep PDE solving.

Contains:
  - Net_DGM  : DGM-style recurrent network  [Sirignano & Spiliopoulos 2018]
  - FFN      : Standard feed-forward network [Sabate-Vidales 2021]
"""

import torch
import torch.nn as nn


class Net_DGM(nn.Module):
    """
    DGM network for approximating PDE solutions.

    Architecture (per DGM layer l):
        G = σ(U_g z + W_g S + b_g)       gate
        Z = σ(U_z z + W_z S + b_z)       gate
        R = σ(U_r z + W_r S + b_r)       gate
        H = tanh(U_h z + W_h (S⊙R) + b_h)
        S ← (1−G)⊙H + Z⊙S

    Input  z = (t, x) ∈ ℝ^{1+d}
    Output ∈ ℝ^{output_dim}
    """

    def __init__(self, input_dim, hidden_size, output_dim=1, n_layers=3):
        super().__init__()
        self.n_layers = n_layers

        self.initial = nn.Linear(input_dim, hidden_size)

        # DGM recurrent layers
        for l in range(n_layers):
            setattr(self, f"Ug_{l}", nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f"Wg_{l}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"Uz_{l}", nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f"Wz_{l}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"Ur_{l}", nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f"Wr_{l}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"Uh_{l}", nn.Linear(input_dim, hidden_size, bias=False))
            setattr(self, f"Wh_{l}", nn.Linear(hidden_size, hidden_size))

        self.output = nn.Linear(hidden_size, output_dim)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : Tensor, shape (B,) or (B, 1)
        x : Tensor, shape (B, d)

        Returns
        -------
        out : Tensor, shape (B, output_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        z = torch.cat([t, x], dim=1)
        S = torch.tanh(self.initial(z))

        for l in range(self.n_layers):
            G = torch.sigmoid(getattr(self, f"Ug_{l}")(z) + getattr(self, f"Wg_{l}")(S))
            Z = torch.sigmoid(getattr(self, f"Uz_{l}")(z) + getattr(self, f"Wz_{l}")(S))
            R = torch.sigmoid(getattr(self, f"Ur_{l}")(z) + getattr(self, f"Wr_{l}")(S))
            H = torch.tanh(getattr(self, f"Uh_{l}")(z) + getattr(self, f"Wh_{l}")(S * R))
            S = (1.0 - G) * H + Z * S

        return self.output(S)


class FFN(nn.Module):
    """
    Standard feed-forward network.

    Architecture defined by `sizes` list, e.g. [3, 100, 100, 2]:
      Linear(3→100) → Tanh → Linear(100→100) → Tanh → Linear(100→2)
    """

    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1]))
            if j < len(sizes) - 2:
                layers.append(activation())
            else:
                layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x : Tensor, shape (B, sizes[0]) → (B, sizes[-1])"""
        return self.net(x)


class SimpleDGMNet(nn.Module):
    """
    Simpler DGM-style network used in Exercises 3 & 4.

    Uses SiLU (swish) activation and plain hidden layers (no gating).
    """

    def __init__(self, input_dim, hidden_dim, n_layers, output_dim=1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : Tensor, shape (B, 1)
        x : Tensor, shape (B, d)

        Returns
        -------
        out : Tensor, shape (B, output_dim)
        """
        h = torch.cat([t, x], dim=1)
        h = torch.nn.functional.silu(self.input_layer(h))
        for layer in self.hidden_layers:
            h = torch.nn.functional.silu(layer(h))
        return self.output_layer(h)
