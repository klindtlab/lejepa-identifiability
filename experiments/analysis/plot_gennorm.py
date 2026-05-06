"""
Generalized normal sweep across 4 mixings.

Emits three figures:
  1. fig_gennorm.pdf       — 4-panel R^2(h -> z) vs alpha for SIGReg/VICReg/InfoNCE
  2. fig_gennorm_orth.pdf  — 4-panel orthogonality error vs alpha (unconstrained ylim
                             to show InfoNCE excursions off the chart)
  3. fig_gennorm_main.pdf  — single-panel spiral-only headline figure for main text,
                             matching the Fig.~4b style of the paper

Usage:
    python analysis/plot_gennorm.py --results_dir results/gennorm/ --out figures/
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

MIXINGS = [("spiral", "Spiral"), ("banana", "Banana"),
           ("sinusoid", "Sinusoid"), ("nvp", "NVP")]

MODES = ("lejepa", "whiten", "infonce")
COLORS = {"lejepa": "#d62728", "whiten": "#1f77b4", "infonce": "#2ca02c"}
LABELS = {"lejepa": "SIGReg", "whiten": "VICReg", "infonce": "InfoNCE"}

YLABELS = {
    "r2_hz_grid":               r"Linear identifiability  $R^2(h \to z)$",
    "r2_hz":                    r"Linear identifiability  $R^2(h \to z)$",
    "orth_err_normalized_grid": r"Orthogonality error  $\|\hat Q^\top \hat Q - I\|_F / \sqrt{n}$",
    "orth_err_normalized":      r"Orthogonality error  $\|\hat Q^\top \hat Q - I\|_F / \sqrt{n}$",
}

# Sensible y-limits per metric. R^2 is bounded in [0,1] so we clip there.
# Orthogonality error is unbounded above (Whitening/InfoNCE off-Gaussian can spike
# into the tens), so we use log scale and let matplotlib autoscale.
YLIMS = {
    "r2_hz_grid":               (-0.05, 1.05),
    "r2_hz":                    (-0.05, 1.05),
    "orth_err_normalized_grid": None,  # autoscale; log scale (see YSCALES) handles outliers
    "orth_err_normalized":      None,
}

# Y-axis scale per metric. Linear by default; log for orth error to compress
# off-Gaussian excursions while still showing structure near zero.
YSCALES = {
    "r2_hz_grid":               "linear",
    "r2_hz":                    "linear",
    "orth_err_normalized_grid": "log",
    "orth_err_normalized":      "log",
}


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
def load_groups(results_dir, metric):
    """groups[(mixing, mode, alpha)] -> list of seed values for `metric`."""
    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("experiment") != "gennorm":
            continue
        alpha = r.get("source_alpha")
        if alpha is None:
            m = re.search(r"alpha=([\d.]+)", r.get("run_name", ""))
            if m:
                alpha = float(m.group(1))
        if alpha is None or metric not in r:
            continue
        groups[(r["mixing"], r["mode"], alpha)].append(r[metric])
    return groups


def curve(groups, mixing, mode):
    alphas = sorted({a for (mx, m, a) in groups if mx == mixing and m == mode})
    mu = np.array([np.mean(groups[(mixing, mode, a)]) for a in alphas])
    sd = np.array([np.std (groups[(mixing, mode, a)]) for a in alphas])
    return np.array(alphas), mu, sd


# ──────────────────────────────────────────────────────────────────────────
# Figure 1 & 2: 4-panel grids (one per metric)
# ──────────────────────────────────────────────────────────────────────────
def plot_grid(groups, metric, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
    use_log = YSCALES[metric] == "log"

    for ax, (mix_key, mix_name) in zip(axes, MIXINGS):
        for mode in MODES:
            alphas, mu, sd = curve(groups, mix_key, mode)
            if len(alphas) == 0:
                continue
            ax.plot(alphas, mu, marker="o", ms=5, lw=1.8,
                    color=COLORS[mode], label=LABELS[mode], zorder=3)
            # On log axes, clip the lower edge of the band away from zero
            # so fill_between doesn't disappear / warn.
            lower = mu - sd
            if use_log:
                lower = np.maximum(lower, 1e-3)
            ax.fill_between(alphas, lower, mu + sd,
                            color=COLORS[mode], alpha=0.2, zorder=2)
        ax.set_xscale("log", base=2)
        if use_log:
            ax.set_yscale("log")
        ax.axvline(2.0, color="black", lw=0.7, ls="--", alpha=0.6, zorder=1)
        ax.set_xlabel(r"Source shape $\alpha$")
        ax.set_title(mix_name)
        if YLIMS[metric] is not None:
            ax.set_ylim(*YLIMS[metric])
        ax.grid(alpha=0.3, which="both" if use_log else "major")

    axes[0].set_ylabel(YLABELS[metric])
    axes[-1].legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Figure 3: main-text single-panel headline (spiral mixing, all three methods)
# ──────────────────────────────────────────────────────────────────────────
def plot_main_panel(groups, out_path):
    """Single-panel spiral-only figure to sit next to the bound-verification panel
    in the main-text composite figure (Fig.~4b in the paper)."""
    FIGSIZE  = 0.8 * np.array((4.0, 3.0))
    LW       = 2.0
    MS       = 6
    FONTSIZE = 11

    rc_saved = plt.rcParams.copy()
    plt.rcParams.update({
        "font.size":         FONTSIZE,
        "axes.labelsize":    FONTSIZE,
        "xtick.labelsize":   FONTSIZE - 1,
        "ytick.labelsize":   FONTSIZE - 1,
        "legend.fontsize":   FONTSIZE - 1,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for mode in MODES:
        alphas, mu, sd = curve(groups, "spiral", mode)
        if len(alphas) == 0:
            continue
        ax.plot(alphas, mu, marker="o", ms=MS, lw=LW,
                color=COLORS[mode], label=LABELS[mode], zorder=3)
        ax.fill_between(alphas, mu - sd, mu + sd,
                        color=COLORS[mode], alpha=0.2, zorder=2)

    # Reference lines for canonical distributions
    ax.axvline(1.0, 0, 0.95, color="gray",  lw=0.8, ls=":",  alpha=0.7, zorder=1)
    ax.axvline(2.0, 0, 0.90, color="black", lw=0.8, ls="--", alpha=0.7, zorder=1)
    ax.text(1.0, 1.04, "Laplace", ha="center", va="bottom",
            fontsize=FONTSIZE - 1, color="gray")

    ax.set_xscale("log", base=2)
    ax.set_ylabel(r"Linearity")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xticks([2**(-2), 2, 16])
    ax.set_xticklabels([r"$\leftarrow$ sparse", "Gaussian", r"uniform $\rightarrow$"])

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=500)
    print(f"Saved {out_path}")
    plt.close(fig)
    plt.rcParams.update(rc_saved)


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/gennorm/")
    p.add_argument("--out",         default="figures/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 4-panel R^2 grid (appendix)
    groups_r2 = load_groups(args.results_dir, "r2_hz_grid")
    plot_grid(groups_r2, "r2_hz_grid",
              os.path.join(args.out, "fig_gennorm.pdf"))

    # 4-panel orthogonality grid (appendix, autoscaled)
    groups_orth = load_groups(args.results_dir, "orth_err_normalized_grid")
    plot_grid(groups_orth, "orth_err_normalized_grid",
              os.path.join(args.out, "fig_gennorm_orth.pdf"))

    # Single-panel main-text headline (spiral, all three methods)
    plot_main_panel(groups_r2,
                    os.path.join(args.out, "fig_gennorm_main.pdf"))


if __name__ == "__main__":
    main()