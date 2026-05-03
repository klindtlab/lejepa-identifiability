"""
Cross-experiment scatter plots (2x2 panel).

Usage:
    python analysis/plot_scatter.py --results_dirs results/2d results/scaling results/grid results/ablation --out figures/
"""

import argparse, os, glob, json
import numpy as np
import matplotlib.pyplot as plt

EXPERIMENT_COLORS = {"2d": "tab:blue", "grid": "tab:red", "scaling": "tab:green", "ablation": "tab:orange"}
EXPERIMENT_ORDER = ["grid", "scaling", "2d", "ablation"]


def load_all(dirs):
    rows = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                with open(path) as f:
                    rows.append(json.load(f))
            except Exception:
                pass
    return rows


def scatter_by_experiment(ax, rows, x_key, y_key):
    for exp in EXPERIMENT_ORDER:
        pts = [r for r in rows if r.get("experiment") == exp
               and r.get(x_key) is not None and r.get(y_key) is not None]
        if not pts:
            continue
        ax.scatter([r[x_key] for r in pts],
                   [r[y_key] for r in pts],
                   c=EXPERIMENT_COLORS[exp],
                   s=25, alpha=0.7, edgecolors='k', linewidths=0.3,
                   label=exp, zorder=3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dirs", nargs="+", required=True)
    p.add_argument("--out", default="figures/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_all(args.results_dirs)
    if not rows:
        print("No results."); return

    xlim = (5e-3, 1e0)
    ylim = (0.9, 1.01)

    fig = plt.figure(figsize=0.65 * np.array((8, 7)))

    # ── Total loss vs R² ──
    ax = plt.subplot(2, 2, 1)
    scatter_by_experiment(ax, rows, "final_loss", "r2_hz")
    ax.set_xlabel("Total loss")
    ax.set_ylabel("Linear Identifiability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xscale("log")

    # ── Alignment vs R² ──
    ax = plt.subplot(2, 2, 2)
    scatter_by_experiment(ax, rows, "final_align", "r2_hz")
    ax.set_xlabel("Alignment loss")
    ax.set_ylabel("Linear Identifiability")
    ax.grid(alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xscale("log")

    # ── SIGReg vs R² ──
    ax = plt.subplot(2, 2, 3)
    scatter_by_experiment(ax, rows, "final_sigreg", "r2_hz")
    ax.set_xlabel("SIGReg loss")
    ax.set_ylabel("Linear Identifiability")
    ax.grid(alpha=0.3)
    ax.set_ylim(*ylim)
    ax.set_xscale("log")

    # ── SIGReg vs whitening ──
    ax = plt.subplot(2, 2, 4)
    scatter_by_experiment(ax, rows, "final_sigreg", "final_whiten")
    ax.set_xlabel("SIGReg loss")
    ax.set_ylabel("Whitening loss")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "scatter_plots.pdf"), bbox_inches="tight")
    print("Saved scatter_plots.pdf")
    plt.close()


if __name__ == "__main__":
    main()