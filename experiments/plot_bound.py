"""
Bound verification plots.

bound_verification.pdf  — 1×2, main paper (clean, simple labels)
bound_decomposition.pdf — 2×2, appendix (technical)

Usage:
    python plot_bound.py --results_dir results/ --out figures/
"""

import argparse, glob, os, math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 200, "font.family": "serif",
})


# ═══════════════════════════════════════════════════════════════════════════════
# Model / evaluation
# ═══════════════════════════════════════════════════════════════════════════════
def g_torch(z):
    norms = z.norm(dim=-1) * torch.pi
    c, s = norms.cos(), norms.sin()
    R = torch.stack([torch.stack([c, -s], dim=-1),
                     torch.stack([s,  c], dim=-1)], dim=-2)
    return (R @ z.unsqueeze(-1)).squeeze(-1)


def make_encoder(N, hidden=256):
    return nn.Sequential(
        nn.Linear(N, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, N),
    )


def evaluate_bound(r):
    device = torch.device("cpu")
    rho, N, D_data = r["rho"], r["N"], r["D"]
    hidden = r.get("hidden", 256)
    f = make_encoder(N, hidden)
    if "model_state_dict" not in r:
        return None
    f.load_state_dict(r["model_state_dict"])
    f.eval()

    np.random.seed(r["seed"])
    z_np = np.random.normal(0, 1, (D_data, N))
    z_t = torch.tensor(z_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        fgz_t = f(g_torch(z_t))
        fgz = fgz_t.numpy()

    cov_hz = np.cov(fgz, rowvar=False)
    epsilon = np.linalg.norm(cov_hz - np.eye(N), 'fro')
    trace_cov = np.trace(cov_hz)

    np.random.seed(r["seed"] + 99999)
    eta = torch.randn_like(z_t)
    z_prime = rho * z_t + math.sqrt(1 - rho**2) * eta
    with torch.no_grad():
        hz_prime = f(g_torch(z_prime))
    L_h = (hz_prime - fgz_t).square().sum(dim=-1).mean().item()

    delta = max(L_h - 2 * (1 - rho) * trace_cov, 0.0)
    spectral_gap = 2 * rho * (1 - rho)
    D_val = delta / spectral_gap if spectral_gap > 0 else float('inf')
    bound = D_val + (epsilon + D_val) ** 2

    M = fgz.T @ z_np / len(z_np)
    U, S, Vt = np.linalg.svd(M)
    Q = U @ Vt
    procrustes_mse = np.mean(np.sum((fgz - z_np @ Q.T)**2, axis=1))

    return {
        "epsilon": epsilon, "delta": delta, "D_val": D_val,
        "bound": bound, "procrustes_mse": procrustes_mse,
        "lamb": r["lamb"], "rho": rho, "seed": r["seed"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Scatter helpers
# ═══════════════════════════════════════════════════════════════════════════════
LAMB_MARKERS = {1e-3: 'o', 5e-3: 's', 1e-2: 'D', 5e-2: '^', 1e-1: 'v', 5e-1: 'P'}
RHO_MARKERS  = {0.3: 'o', 0.5: 's', 0.7: 'D', 0.8: '^', 0.9: 'v', 0.95: 'P', 0.99: 'X'}


def scatter_marker_lamb_color_rho(ax, xvals, yvals, lambs, rhos,
                                   add_legend=True, loc='upper left'):
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0.3, vmax=0.99)
    for lamb in sorted(set(lambs)):
        mask = lambs == lamb
        ax.scatter(xvals[mask], yvals[mask],
                   c=rhos[mask], cmap=cmap, norm=norm,
                   marker=LAMB_MARKERS.get(lamb, 'o'),
                   s=30, alpha=0.8, edgecolors='k', linewidths=0.3,
                   label=f"$\\lambda$={lamb:.0e}" if add_legend else None)
    if add_legend:
        ax.legend(fontsize=6, loc=loc, framealpha=0.9, handletextpad=0.3)
    return cmap, norm, r"Correlation [$\rho$]"


def scatter_marker_rho_color_lamb(ax, xvals, yvals, lambs, rhos,
                                   add_legend=True, loc='upper left'):
    log_lambs = np.log10(lambs)
    cmap = plt.cm.plasma
    norm = mpl.colors.Normalize(vmin=log_lambs.min(), vmax=log_lambs.max())
    for rho_val in sorted(set(rhos)):
        mask = rhos == rho_val
        ax.scatter(xvals[mask], yvals[mask],
                   c=log_lambs[mask], cmap=cmap, norm=norm,
                   marker=RHO_MARKERS.get(rho_val, 'o'),
                   s=30, alpha=0.8, edgecolors='k', linewidths=0.3,
                   label=f"$\\rho$={rho_val:.2f}" if add_legend else None)
    if add_legend:
        ax.legend(fontsize=6, loc=loc, framealpha=0.9, handletextpad=0.3)
    return cmap, norm, r"Regularization [$\log_{10}\lambda$]"


# ═══════════════════════════════════════════════════════════════════════════════
# Main paper figure
# ═══════════════════════════════════════════════════════════════════════════════
def plot_verification(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    errors = np.array([d["procrustes_mse"] for d in data])
    bounds = np.array([d["bound"] for d in data])
    rhos   = np.array([d["rho"] for d in data])
    lambs  = np.array([d["lamb"] for d in data])

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    scatter_fns = [scatter_marker_lamb_color_rho, scatter_marker_rho_color_lamb]

    for ax, scatter_fn in zip(axes, scatter_fns):
        # Shaded regions
        lo = min(errors[errors > 0].min(), bounds[bounds > 0].min()) * 0.3
        hi = max(errors.max(), bounds.max()) * 3

        # Green below diagonal, red above
        pts = np.logspace(np.log10(lo), np.log10(hi), 200)
        ax.fill_between(pts, lo * 0.1, pts, color='#c8e6c9', alpha=0.35, zorder=0)
        ax.fill_between(pts, pts, hi * 10, color='#ffcdd2', alpha=0.35, zorder=0)

        # Labels
        ax.text(0.06, 0.92, "unfeasible", transform=ax.transAxes,
                fontsize=7, color='#c62828', fontstyle='normal', alpha=0.8)
        ax.text(0.65, 0.92, "feasible", transform=ax.transAxes,
                fontsize=7, color='#2e7d32', fontstyle='normal', alpha=0.8)

        # Data
        cmap, norm, cbar_label = scatter_fn(ax, bounds, errors, lambs, rhos,
                                             loc='lower right')

        # Diagonal
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=0.8,
                label="error = bound")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo * 0.5, errors.max() * 3)
        ax.set_xlabel("Recovery error bound")
        ax.set_ylabel("Actual recovery error")

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=cbar_label, shrink=0.85, pad=0.02)

        # Re-add bound line to legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=5.5, loc='lower right',
                  framealpha=0.9, handletextpad=0.3)

        ax.grid()

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(out_dir, "bound_verification.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bound_verification.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved bound_verification.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Appendix figure
# ═══════════════════════════════════════════════════════════════════════════════
YLABEL_TECH = r"$\min_{Q \in O(n)} \mathbb{E}[\|h(z) - Qz\|^2]$"

def plot_decomposition(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    errors   = np.array([d["procrustes_mse"] for d in data])
    epsilons = np.array([d["epsilon"] for d in data])
    deltas   = np.array([d["delta"] for d in data])
    rhos     = np.array([d["rho"] for d in data])
    lambs    = np.array([d["lamb"] for d in data])

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    x_configs = [
        (epsilons,
         r"Covariance error $\varepsilon = \|\mathrm{Cov}(h(z)) - I\|_F$",
         r"Error vs $\varepsilon$"),
        (deltas,
         r"Alignment gap $\delta = \mathcal{L}(h) - 2(1{-}\rho)\,\mathrm{tr}(\Sigma)$",
         r"Error vs $\delta$"),
    ]

    scatter_fns = [scatter_marker_lamb_color_rho, scatter_marker_rho_color_lamb]

    for row, scatter_fn in enumerate(scatter_fns):
        for col, (xvals, xlabel, title) in enumerate(x_configs):
            ax = axes[row, col]
            add_legend = (col == 0)
            cmap, norm, cbar_label = scatter_fn(
                ax, xvals, errors, lambs, rhos,
                add_legend=add_legend, loc='upper right')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(YLABEL_TECH)
            ax.set_title(title)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=cbar_label, shrink=0.85)

            ax.grid()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bound_decomposition.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "bound_decomposition.png"), bbox_inches="tight")
    plt.close(fig)
    print("  Saved bound_decomposition.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/")
    p.add_argument("--out", type=str, default="figures/")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, "*.pt")))
    print(f"Found {len(files)} result files.")

    data = []
    for fpath in files:
        try:
            r = torch.load(fpath, map_location="cpu", weights_only=False)
            d = evaluate_bound(r)
            if d is not None:
                data.append(d)
        except Exception as e:
            print(f"  ERROR {fpath}: {e}")

    if not data:
        print("No evaluable results.")
        exit(1)

    print(f"{len(data)} runs loaded.")
    plot_verification(data, args.out)
    plot_decomposition(data, args.out)
    print("Done!")