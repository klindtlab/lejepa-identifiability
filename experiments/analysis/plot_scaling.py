"""
Scaling plots: R² and orthogonality vs latent dimension N.

Usage:
    python analysis/plot_scaling.py --results_dir results/scaling/ --out figures/
"""

import argparse, os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        rows.append({k: r.get(k) for k in [
            "N", "seed", "r2_zx", "r2_xz", "r2_zh", "r2_hz",
            "orth_err", "orth_err_normalized", "final_loss",
            "final_align", "final_sigreg", "final_whiten",
            "epsilon", "delta", "approx_bound", "procrustes_mse",
        ]})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/scaling/")
    p.add_argument("--out", default="figures/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_results(args.results_dir)
    if len(df) == 0:
        print("No results."); return

    summary = df.groupby("N").agg(
        r2_xz_mean=("r2_xz", "mean"), r2_xz_std=("r2_xz", "std"),
        r2_hz_mean=("r2_hz", "mean"), r2_hz_std=("r2_hz", "std"),
        orth_mean=("orth_err_normalized", "mean"), orth_std=("orth_err_normalized", "std"),
    ).reset_index()
    dims = summary["N"].values

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    ax = axes[0]
    ax.errorbar(dims, summary["r2_xz_mean"], yerr=summary["r2_xz_std"],
                fmt="o-", capsize=3, color="gray", label=r"Probe: $g(z) \to z$")
    ax.errorbar(dims, summary["r2_hz_mean"], yerr=summary["r2_hz_std"],
                fmt="s-", capsize=3, label=r"Probe: $f \circ g(z) \to z$")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent dimension $N$"); ax.set_ylabel(r"Linearity [$R^2$]")
    ax.set_title("Latent Recovery"); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(dims); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(dims, summary["orth_mean"], yerr=summary["orth_std"],
                fmt="D-", capsize=3, color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent dimension $N$")
    ax.set_ylabel(r"$\|A^\top A - I\|_F / \sqrt{N}$")
    ax.set_title("Orthogonality Error"); ax.set_xticks(dims); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_scaling.pdf"), bbox_inches="tight")
    print("Saved fig_scaling.pdf")
    plt.close()


if __name__ == "__main__":
    main()
