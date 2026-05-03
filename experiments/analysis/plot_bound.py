"""
Bound verification and grid search plots.

bound_verification.pdf  — pooled across experiments (main paper)
bound_decomposition.pdf — grid search only (appendix)
heatmap_*.pdf           — grid search only (appendix)

Usage:
    python analysis/plot_bound.py \
    --results_dirs results/grid results/2d results/scaling results/gennorm \
    --out figures/
"""

import argparse, os, glob, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.rcParams.update({
#     "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
#     "figure.dpi": 200, "font.family": "serif",
# })

EXPERIMENT_COLORS = {
    "2d":      "tab:blue",
    "grid":    "tab:red",
    "scaling": "tab:green",
    "reacher": "tab:purple",
    "gennorm": "tab:orange",
}
EXPERIMENT_MARKERS = {
    "2d":      "o",
    "grid":    "D",
    "scaling": "s",
    "reacher": "v",
    "gennorm": "^",
}
EXPERIMENT_ORDER = ["grid", "scaling", "2d", "reacher", "gennorm"]


def is_valid_run(r, path=""):
    """SIGReg + Gaussian source + non-degenerate (encoder actually learned)."""
    if r.get("mode") != "lejepa":
        return False
    sd = r.get("source_dist", "gaussian")
    if sd == "gennorm" and abs(r.get("source_alpha", 0) - 2.0) > 1e-6:
        return False
    if sd not in ("gaussian", "gennorm"):
        return False
    # Drop degenerate runs where the encoder failed to learn
    if r.get("r2_hz", 0) < 0.5:
        return False
    return True


def load_all(dirs):
    data = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            with open(path) as f:
                r = json.load(f)
            if not isinstance(r, dict):
                continue
            if r.get("approx_bound") is None or r.get("procrustes_mse") is None:
                continue
            if not is_valid_run(r, path):
                continue
            data.append(r)
    print(f"Loaded {len(data)} Gaussian-source runs")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+", required=True)
    p.add_argument("--out", default="figures/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = load_all(args.results_dirs)
    if not data:
        print("No results."); return

    # ══════════════════════════════════════════════════════════════════════
    # Bound verification (main paper, single panel)
    # ══════════════════════════════════════════════════════════════════════
    errors = np.array([d["procrustes_mse"] for d in data])
    bounds = np.array([d["approx_bound"] for d in data])
    experiments = [d["experiment"] for d in data]

    fig, ax = plt.subplots(figsize=0.8 * np.array((3, 3)))

    pos = (errors > 0) & (bounds > 0)
    lo = min(errors[pos].min(), bounds[pos].min()) * 0.3
    hi = max(errors.max(), bounds.max()) * 3
    pts = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.fill_between(pts, lo * 0.1, pts, color='#c8e6c9', alpha=0.35, zorder=0)
    ax.fill_between(pts, pts, hi * 10, color='#ffcdd2', alpha=0.35, zorder=0)
    
    for exp in EXPERIMENT_ORDER:
        mask = np.array([e == exp for e in experiments])
        if not mask.any():
            continue
        ax.scatter(bounds[mask], errors[mask],
                   c=EXPERIMENT_COLORS[exp],
                   marker=EXPERIMENT_MARKERS[exp],
                   s=32, alpha=0.7, edgecolors='k', linewidths=0.3,
                   label=exp, zorder=3)
    
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=0.8)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Recovery error bound")
    ax.set_ylabel("Recovery error")
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bound_verification.pdf"), bbox_inches="tight")
    print("Saved bound_verification.pdf")
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Grid-specific plots (appendix)
    # ══════════════════════════════════════════════════════════════════════
    grid_dir = None
    for d in args.results_dirs:
        if "grid" in d:
            grid_dir = d
            break
    if grid_dir is None:
        print("No grid dir found, skipping decomposition and heatmaps.")
        return

    grid_data = []
    for path in sorted(glob.glob(os.path.join(grid_dir, "*.json"))):
        with open(path) as f:
            grid_data.append(json.load(f))
    if not grid_data:
        print("No grid results."); return

    errors_g = np.array([d["procrustes_mse"] for d in grid_data])
    epsilons = np.array([d["epsilon"] for d in grid_data])
    deltas = np.array([d["delta"] for d in grid_data])
    rhos = np.array([d["rho"] for d in grid_data])
    lambs = np.array([d["lamb"] for d in grid_data])

    LAMB_MARKERS = {1e-6: 'h', 1e-5: 'H', 1e-4: 'p',
                    1e-3: 'o', 5e-3: 's', 1e-2: 'D', 5e-2: '^', 1e-1: 'v', 5e-1: 'P'}
    RHO_MARKERS = {0.3: 'o', 0.5: 's', 0.7: 'D', 0.8: '^', 0.9: 'v', 0.95: 'P', 0.99: 'X'}

    def scatter_by_lamb(ax, xvals, yvals):
        cmap = plt.cm.viridis
        norm = mpl.colors.Normalize(vmin=min(rhos), vmax=max(rhos))
        for lamb in sorted(set(lambs)):
            mask = lambs == lamb
            ax.scatter(xvals[mask], yvals[mask], c=rhos[mask], cmap=cmap, norm=norm,
                       marker=LAMB_MARKERS.get(lamb, 'o'), s=30, alpha=0.8,
                       edgecolors='k', linewidths=0.3, label=f"$\\lambda$={lamb:.0e}")
        return cmap, norm, r"Correlation [$\rho$]"

    def scatter_by_rho(ax, xvals, yvals):
        log_lambs = np.log10(lambs)
        cmap = plt.cm.plasma
        norm = mpl.colors.Normalize(vmin=log_lambs.min(), vmax=log_lambs.max())
        for rho_val in sorted(set(rhos)):
            mask = rhos == rho_val
            ax.scatter(xvals[mask], yvals[mask], c=log_lambs[mask], cmap=cmap, norm=norm,
                       marker=RHO_MARKERS.get(rho_val, 'o'), s=30, alpha=0.8,
                       edgecolors='k', linewidths=0.3, label=f"$\\rho$={rho_val:.2f}")
        return cmap, norm, r"Regularization [$\log_{10}\lambda$]"

    # ── Decomposition ──
    ylabel = r"$\min_{Q \in O(n)} \mathbb{E}[\|h(z) - Qz\|^2]$"
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    x_configs = [
        (epsilons, r"$\varepsilon = \|\mathrm{Cov}(h(z)) - I\|_F$", r"Error vs $\varepsilon$"),
        (deltas, r"$\delta = \mathcal{L}(h) - 2(1{-}\rho)\,\mathrm{tr}(\Sigma)$", r"Error vs $\delta$"),
    ]
    for row, scatter_fn in enumerate([scatter_by_lamb, scatter_by_rho]):
        for col, (xvals, xlabel, title) in enumerate(x_configs):
            ax = axes[row, col]
            cmap, norm, cbar_label = scatter_fn(ax, xvals, errors_g)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            plt.colorbar(sm, ax=ax, label=cbar_label, shrink=0.85)
            ax.legend(fontsize=5.5, loc='upper right', framealpha=0.9)
            ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "bound_decomposition.pdf"), bbox_inches="tight")
    print("Saved bound_decomposition.pdf")
    plt.close()

    # ── Heatmaps ──
    unique_lambs = sorted(set(lambs))
    unique_rhos = sorted(set(rhos))
    for metric_key, title, cmap_name in [
        ("r2_hz", "Linear $R^2$ (h -> z)", "viridis"),
        ("orth_err_normalized", "Orth. error normalized", "viridis_r"),
    ]:
        grid = np.full((len(unique_lambs), len(unique_rhos)), np.nan)
        counts = np.zeros_like(grid)
        for r in grid_data:
            li = unique_lambs.index(r["lamb"])
            ri = unique_rhos.index(r["rho"])
            val = r.get(metric_key)
            if val is not None:
                if np.isnan(grid[li, ri]):
                    grid[li, ri] = 0
                grid[li, ri] += val
                counts[li, ri] += 1
        grid = np.where(counts > 0, grid / counts, np.nan)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap_name)
        plt.colorbar(im, ax=ax, label=title)
        ax.set_xticks(range(len(unique_rhos)))
        ax.set_xticklabels([f"{r:.2f}" for r in unique_rhos])
        ax.set_yticks(range(len(unique_lambs)))
        ax.set_yticklabels([f"{l:.0e}" for l in unique_lambs])
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\lambda$")
        ax.set_title(title)
        for i in range(len(unique_lambs)):
            for j in range(len(unique_rhos)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=6)
        fig.tight_layout()
        safe = metric_key.replace(".", "_")
        fig.savefig(os.path.join(args.out, f"heatmap_{safe}.pdf"), bbox_inches="tight")
        print(f"Saved heatmap_{safe}.pdf")
        plt.close()


if __name__ == "__main__":
    main()