"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sliced characteristic function regularizer (Balestriero & LeCun 2025)."""

    def __init__(self, knots=17, n_slices=256, t_max=3.0):
        super().__init__()
        self.n_slices = n_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-t**2 / 2))
        self.register_buffer("weights", w * torch.exp(-t**2 / 2))

    def forward(self, h):
        """h: (V, B, N) -> scalar."""
        flat = h.flatten(0, 1)
        A = F.normalize(torch.randn(flat.size(-1), self.n_slices, device=flat.device), dim=0)
        xt = (flat @ A).unsqueeze(-1) * self.t
        err = (xt.cos().mean(0) - self.phi) ** 2 + xt.sin().mean(0) ** 2
        return (err @ self.weights).mean() * flat.size(0)


def whitening_loss(h):
    """||Cov(h) - I||²_F.  h: (V, B, N) -> scalar."""
    flat = h.flatten(0, 1)
    flat = flat - flat.mean(dim=0)
    cov = (flat.T @ flat) / (flat.shape[0] - 1)
    return (cov - torch.eye(flat.shape[1], device=h.device)).square().mean()


def alignment_loss(h):
    """Pull positive-pair views together. h: (V, B, N) -> scalar."""
    return (h.mean(0) - h).square().mean()
