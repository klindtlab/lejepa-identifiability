# %% [markdown]
# # LeJEPA 2D Identifiability Demo
#
# ### (runs in 30s on GPU T4, 2min on CPU)
# ### Trains an encoder on a nonlinear spiral mixing problem.
# ### Supports two modes:
# ### - **lejepa**: alignment loss + SIGReg (Gaussianity)
# ### - **whiten**: alignment loss + covariance whitening (no full Gaussianity, just Cov → I)

# %% Parameters
SEED = 1337
D = 100000             # number of datapoints
N = 2                  # latent dimension
HIDDEN = 256           # encoder hidden dimension
RHO = 0.95             # OU correlation
BATCH_SIZE = 256
STEPS = 3000
LR = 1e-3
LOG_EVERY = 100

MODE = "lejepa"        # "lejepa" or "whiten"

# regularization weight depends on loss type
if MODE == "lejepa":
    LAMB = 5e-3
elif MODE == "whiten":
    LAMB = 0.5

# SIGReg defaults (from Balestriero & LeCun 2025)
SIGREG_KNOTS = 17      # quadrature knots for characteristic function
SIGREG_SLICES = 256    # number of random 1D projections
SIGREG_TMAX = 3.0      # max frequency

# %% Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from scipy.optimize import linear_sum_assignment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %% Color map (angle → hue, radius → lightness)

def make_colors(z):
    """Polar color map: hue = angle, lightness = radius."""
    x, y = z[:, 0].cpu().numpy(), z[:, 1].cpu().numpy()
    angles = np.arctan2(y, x)
    radii = np.sqrt(x**2 + y**2)
    hue = (angles + np.pi) / (2 * np.pi)
    lightness = 0.3 + 0.4 * (radii / (radii.max() + 1e-8))
    saturation = np.full_like(hue, 0.85)
    return np.array([
        colorsys.hls_to_rgb(h, l, s)
        for h, l, s in zip(hue, lightness, saturation)
    ])

# %% Mixing function

def g(z):
    """g(z) = R(π‖z‖) z — measure-preserving spiral diffeomorphism."""
    norms = z.norm(dim=-1) * torch.pi
    c, s = norms.cos(), norms.sin()
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)
    return (R @ z.unsqueeze(-1)).squeeze(-1)

# %% MCC metric

def compute_mcc(z_true, z_learned):
    """Mean absolute correlation, maximized over permutations (Hungarian)."""
    a = z_true - z_true.mean(dim=0)
    b = z_learned - z_learned.mean(dim=0)
    a = a / (a.norm(dim=0, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=0, keepdim=True) + 1e-9)
    C = (a.T @ b).abs().cpu().numpy()
    row, col = linear_sum_assignment(-C)
    return C[row, col].mean()

# %% SIGReg loss

class SIGReg(nn.Module):
    """Sliced characteristic function regularizer (Balestriero & LeCun 2025)."""
    def __init__(self, knots=SIGREG_KNOTS, n_slices=SIGREG_SLICES, t_max=SIGREG_TMAX):
        super().__init__()
        self.n_slices = n_slices
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt); w[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", torch.exp(-t**2 / 2))
        self.register_buffer("weights", w * torch.exp(-t**2 / 2))

    def forward(self, h):
        """h: (V, B, N) → scalar loss."""
        flat = h.flatten(0, 1)
        A = F.normalize(torch.randn(flat.size(-1), self.n_slices, device=flat.device), dim=0)
        xt = (flat @ A).unsqueeze(-1) * self.t
        err = (xt.cos().mean(0) - self.phi)**2 + xt.sin().mean(0)**2
        return (err @ self.weights).mean() * flat.size(0)

# %% Whitening loss

def whitening_loss(h):
    """Penalize deviation of covariance from identity: ‖Cov(h) - I‖²_F.
    This enforces both unit variance and decorrelation.
    h: (V, B, N) → scalar loss."""
    flat = h.flatten(0, 1)
    flat = flat - flat.mean(dim=0)
    cov = (flat.T @ flat) / (flat.shape[0] - 1)
    return (cov - torch.eye(flat.shape[1], device=h.device)).square().mean()

# %% Data

torch.manual_seed(SEED)
z = torch.randn(D, N, device=device)
x = g(z)
colors = make_colors(z)

# %% Model

torch.manual_seed(SEED)
encoder = nn.Sequential(
    nn.Linear(N, HIDDEN), nn.GELU(),
    nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
    nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
    nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
    nn.Linear(HIDDEN, N),
).to(device)

sigreg = SIGReg().to(device)
opt = torch.optim.AdamW(encoder.parameters(), lr=LR)

# %% Training

fac = (1 - RHO**2) ** 0.5
log = {"step": [], "align": [], "sigreg": [], "whiten": [], "mcc": []}

for step in range(STEPS + 1):
    idx = torch.randperm(D, device=device)[:BATCH_SIZE]
    z_batch = z[idx]

    # OU channel: z' = ρz + √(1-ρ²)η  →  two views (positive pairs)
    eta = torch.randn(2, BATCH_SIZE, N, device=device)
    z_pos = RHO * z_batch.unsqueeze(0) + fac * eta            # (2, B, N)

    # encode: h = encoder ∘ g
    h = encoder(g(z_pos).flatten(0, 1)).reshape(2, BATCH_SIZE, N)

    # alignment: pull positive pairs together
    align_loss = (h.mean(0) - h).square().mean()

    # regularization: track both, train with one
    sig_loss = sigreg(h)
    wht_loss = whitening_loss(h)

    if MODE == "lejepa":
        loss = LAMB * sig_loss + (1 - LAMB) * align_loss
    else:
        loss = LAMB * wht_loss + (1 - LAMB) * align_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % LOG_EVERY == 0:
        with torch.no_grad():
            hz = encoder(g(z))
        mcc = compute_mcc(z, hz)
        log["step"].append(step)
        log["align"].append(align_loss.item())
        log["sigreg"].append(sig_loss.item())
        log["whiten"].append(wht_loss.item())
        log["mcc"].append(mcc)
        print(f"step {step:5d} | align={align_loss.item():.4f} | "
              f"sigreg={sig_loss.item():.4f} | whiten={wht_loss.item():.4f} | "
              f"MCC={mcc:.4f}")

# %% Training curves

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
steps = log["step"]

axes[0].plot(steps, log["align"])
axes[0].set_ylabel("Alignment loss")

axes[1].plot(steps, log["sigreg"])
axes[1].set_ylabel("SIGReg loss")

axes[2].plot(steps, log["whiten"])
axes[2].set_ylabel("Whitening loss")

axes[3].plot(steps, log["mcc"])
axes[3].set_ylabel("MCC")
axes[3].set_ylim(0, 1.05)

for ax in axes:
    ax.set_xlabel("Step")
    ax.grid(alpha=0.3)

fig.suptitle(f"Training with: {MODE} | λ={LAMB} | ρ={RHO}", fontsize=12)
fig.tight_layout()
plt.show()

# %% Final evaluation

encoder.eval()
with torch.no_grad():
    hz = encoder(g(z)).cpu().numpy()
z_np = z.cpu().numpy()
x_np = x.cpu().numpy()

# %% Main figure (Figure 2 in paper)

s = 5
lim = 4

fig, axes = plt.subplots(1, 3, figsize=(9, 3))

for ax, data, labels in zip(
    axes,
    [z_np, x_np, hz],
    [("True Latent 0", "True Latent 1"),
     ("Observation 0", "Observation 1"),
     ("Learned Latent 0", "Learned Latent 1")],
):
    ax.scatter(data[:, 0], data[:, 1], c=colors, s=s, linewidths=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.3)

fig.tight_layout()
# fig.savefig("fig_lejepa_demo.pdf", bbox_inches="tight")
plt.show()
