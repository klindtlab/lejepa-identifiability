"""Data generation: latent sources and OU augmentation."""

import torch


def sample_latents(D, N, dist="gaussian", device="cuda"):
    """Sample D points in R^N (unit variance)."""
    if dist == "gaussian":
        return torch.randn(D, N, device=device)
    elif dist == "laplace":
        return torch.distributions.Laplace(0, 1 / (2 ** 0.5)).sample((D, N)).to(device)
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def ou_augment(z, rho, n_views=2):
    """OU channel: z' = ρz + √(1-ρ²)η.  Returns (V, B, N)."""
    fac = (1 - rho ** 2) ** 0.5
    eta = torch.randn(n_views, *z.shape, device=z.device)
    return rho * z.unsqueeze(0) + fac * eta
