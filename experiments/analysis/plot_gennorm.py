"""
Generalized normal sweep across 4 mixings.
1 row × 4 panels, each showing recovery metric vs α for SIGReg and Whitening.

Usage:
    python analysis/plot_gennorm.py --results_dir results/gennorm/ --out figures/
    python analysis/plot_gennorm.py --metric orth_err_normalized_grid --suffix _orth
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

MIXINGS = [("spiral", "Spiral"), ("banana", "Banana"),
           ("sinusoid", "Sinusoid"), ("nvp", "NVP")]

YLABELS = {
    "r2_hz_grid":               r"Linear identifiability  $R^2(h \to z)$",
    "r2_hz":                    r"Linear identifiability  $R^2(h \to z)$",
    "orth_err_normalized_grid": r"Orthogonality error  $\|\hat Q^\top \hat Q - I\|_F / \sqrt{n}$",
    "orth_err_normalized":      r"Orthogonality error  $\|\hat Q^\top \hat Q - I\|_F / \sqrt{n}$",
}

# Sensible y-limits per metric (R² in [0,1]; orth_err non-negative, ~1.4 max)
YLIMS = {
    "r2_hz_grid":               (-0.05, 1.05),
    "r2_hz":                    (-0.05, 1.05),
    "orth_err_normalized_grid": (-0.05, 1.55),
    "orth_err_normalized":      (-0.05, 1.55),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/gennorm/")
    p.add_argument("--out", default="figures/")
    p.add_argument("--metric", default="r2_hz_grid",
                   choices=list(YLABELS.keys()))
    p.add_argument("--suffix", default="",
                   help="Filename suffix (e.g. _orth)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    groups = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        if r.get("experiment") != "gennorm":
            continue
        alpha = r.get("source_alpha")
        if alpha is None:
            m = re.search(r"alpha=([\d.]+)", r.get("run_name", ""))
            if m:
                alpha = float(m.group(1))
        if alpha is None or args.metric not in r:
            continue
        groups[(r["mixing"], r["mode"], alpha)].append(r[args.metric])

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)

    colors = {"lejepa": "#d62728", "whiten": "#1f77b4", "infonce": "#2ca02c"}
    labels = {"lejepa": "SIGReg", "whiten": "Whitening", "infonce": "InfoNCE"}

    for ax, (mix_key, mix_name) in zip(axes, MIXINGS):
        for mode in ("lejepa", "whiten", "infonce"):
            alphas = sorted({a for (mx, m, a) in groups
                             if mx == mix_key and m == mode})
            mu = np.array([np.mean(groups[(mix_key, mode, a)]) for a in alphas])
            sd = np.array([np.std (groups[(mix_key, mode, a)]) for a in alphas])
            ax.plot(alphas, mu, marker="o", ms=5, lw=1.8,
                    color=colors[mode], label=labels[mode], zorder=3)
            ax.fill_between(alphas, mu - sd, mu + sd,
                            color=colors[mode], alpha=0.2, zorder=2)
        ax.set_xscale("log", base=2)
        ax.axvline(2.0, color="black", lw=0.7, ls="--", alpha=0.6, zorder=1)
        ax.set_xlabel(r"Source shape $\alpha$")
        ax.set_title(mix_name)
        ax.set_ylim(*YLIMS[args.metric])
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(YLABELS[args.metric])
    axes[-1].legend(frameon=False, loc="best")

    fig.tight_layout()
    out_path = os.path.join(args.out, f"fig_gennorm{args.suffix}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()