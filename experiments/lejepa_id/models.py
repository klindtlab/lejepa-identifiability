"""Encoder architectures."""

import torch
import torch.nn as nn
import numpy as np


def make_mlp_encoder(N, hidden=256, n_layers=4, device="cuda"):
    """MLP encoder."""
    layers = [nn.Linear(N, hidden), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    layers.append(nn.Linear(hidden, N))
    return nn.Sequential(*layers).to(device)


class MatchedEncoder(nn.Module):
    """Inverse coupling-layer encoder matched to NVP mixing architecture."""

    def __init__(self, N, n_layers=4, device="cuda"):
        super().__init__()
        half = N // 2
        self.half = half
        self.n_layers = n_layers
        self.Ws = nn.ParameterList([
            nn.Parameter(torch.randn(half, half, device=device) / np.sqrt(half))
            for _ in range(n_layers)
        ])

    def forward(self, x):
        for i, W in reversed(list(enumerate(self.Ws))):
            z1, z2 = x[..., :self.half], x[..., self.half:]
            if i % 2 == 0:
                z2 = z2 - torch.tanh(z1 @ W)
            else:
                z1 = z1 - torch.tanh(z2 @ W)
            x = torch.cat([z1, z2], dim=-1)
        return x


def make_matched_encoder(N, n_layers=4, seed=42, device="cuda"):
    torch.manual_seed(seed)
    return MatchedEncoder(N, n_layers=n_layers, device=device).to(device)
