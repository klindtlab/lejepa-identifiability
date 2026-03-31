"""
Aggregate grid search results and produce summary plots.

Usage:
    python aggregate_results.py --results_dir results/ --out figures/
"""

import argparse, glob, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})


def load_all(results_dir):
    """Load all .pt result files into a list of dicts."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.pt")))
    print(f"Found {len(files)} result files.")
    records = []
    for f in files:
        try:
            r = torch.load(f, map_location="cpu", weights_only=False)
            records.append(r)
        except Exception as e:
            print(f"  SKIP {f}: {e}")
    return records


def build_grid(records):
    """Pivot records into structured arrays indexed by (lamb, rho)."""
    lambs = sorted(set(r["lamb"] for r in records))
    rhos  = sorted(set(r["rho"]  for r in records))
    seeds = sorted(set(r["seed"] for r in records))

    metrics = ["mcc", "linear_r2", "orth_err", "orth_err_normalized"]
    grid = {m: np.full((len(lambs), len(rhos), len(seeds)), np.nan) for m in metrics}

    lamb_idx = {v: i for i, v in enumerate(lambs)}
    rho_idx  = {v: i for i, v in enumerate(rhos)}
    seed_idx = {v: i for i, v in enumerate(seeds)}

    for r in records:
        li, ri, si = lamb_idx[r["lamb"]], rho_idx[r["rho"]], seed_idx[r["seed"]]
        for m in metrics:
            grid[m][li, ri, si] = r[m]

    return grid, lambs, rhos, seeds


def plot_heatmaps(grid, lambs, rhos, seeds, out_dir):
    """2D heatmaps: mean over seeds for each metric."""
    os.makedirs(out_dir, exist_ok=True)

    metric_titles = {
        "mcc":                  "MCC (↑ better)",
        "linear_r2":            "Linear R² (↑ better)",
        "orth_err_normalized":  "Orthogonality error (↓ better)",
    }
    cmaps = {
        "mcc":                  "viridis",
        "linear_r2":            "viridis",
        "orth_err_normalized":  "viridis_r",
    }

    lamb_labels = [f"{l:.0e}" for l in lambs]
    rho_labels  = [f"{r:.2f}" for r in rhos]

    for metric, title in metric_titles.items():
        mean = np.nanmean(grid[metric], axis=2)  # (n_lamb, n_rho)
        std  = np.nanstd(grid[metric],  axis=2)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(mean, aspect="auto", origin="lower", cmap=cmaps[metric])
        plt.colorbar(im, ax=ax, label=title)

        # Annotate cells with mean ± std
        for i in range(len(lambs)):
            for j in range(len(rhos)):
                val = mean[i, j]
                s   = std[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}\n±{s:.3f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if val < (mean[~np.isnan(mean)].mean()) else "black")

        ax.set_xticks(range(len(rhos)))
        ax.set_xticklabels(rho_labels)
        ax.set_yticks(range(len(lambs)))
        ax.set_yticklabels(lamb_labels)
        ax.set_xlabel("ρ (OU correlation)")
        ax.set_ylabel("λ (SIGReg weight)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"heatmap_{metric}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"heatmap_{metric}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap_{metric}.pdf")


def plot_training_curves(records, out_dir):
    """Training curves: one subplot per (lamb, rho), overlaying seeds."""
    os.makedirs(out_dir, exist_ok=True)
    from collections import defaultdict

    by_config = defaultdict(list)
    for r in records:
        by_config[(r["lamb"], r["rho"])].append(r)

    lambs = sorted(set(k[0] for k in by_config))
    rhos  = sorted(set(k[1] for k in by_config))

    for loss_key, loss_label in [("log_inv_loss", "Invariance loss"),
                                  ("log_sig_loss", "SIGReg loss")]:
        fig, axes = plt.subplots(len(lambs), len(rhos),
                                  figsize=(3 * len(rhos), 2.5 * len(lambs)),
                                  sharex=True, sharey="row", squeeze=False)
        for i, lamb in enumerate(lambs):
            for j, rho in enumerate(rhos):
                ax = axes[i][j]
                runs = by_config.get((lamb, rho), [])
                for r in runs:
                    ax.plot(r["log_steps"], r[loss_key], alpha=0.5, linewidth=0.8)
                if i == 0:
                    ax.set_title(f"ρ={rho:.2f}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"λ={lamb:.0e}", fontsize=9)
                ax.tick_params(labelsize=7)

        fig.suptitle(loss_label, fontsize=13, y=1.01)
        fig.tight_layout()
        safe = loss_key.replace("log_", "")
        fig.savefig(os.path.join(out_dir, f"curves_{safe}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"curves_{safe}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved curves_{safe}.pdf")


def print_summary_table(grid, lambs, rhos, seeds):
    """Print a quick text summary to stdout."""
    print("\n" + "=" * 70)
    print("SUMMARY: MCC (mean ± std over seeds)")
    print("=" * 70)
    header = f"{'λ \\ ρ':>10s}" + "".join(f"  {r:>8.2f}" for r in rhos)
    print(header)
    print("-" * len(header))
    for i, lamb in enumerate(lambs):
        row = f"{lamb:>10.0e}"
        for j in range(len(rhos)):
            m = np.nanmean(grid["mcc"][i, j, :])
            s = np.nanstd(grid["mcc"][i, j, :])
            row += f"  {m:>5.3f}±{s:<4.3f}"
        print(row)

    print("\n" + "=" * 70)
    print("SUMMARY: Linear R² (mean ± std)")
    print("=" * 70)
    header = f"{'λ \\ ρ':>10s}" + "".join(f"  {r:>8.2f}" for r in rhos)
    print(header)
    print("-" * len(header))
    for i, lamb in enumerate(lambs):
        row = f"{lamb:>10.0e}"
        for j in range(len(rhos)):
            m = np.nanmean(grid["linear_r2"][i, j, :])
            s = np.nanstd(grid["linear_r2"][i, j, :])
            row += f"  {m:>5.3f}±{s:<4.3f}"
        print(row)

    print("\n" + "=" * 70)
    print("SUMMARY: Orthogonality error (mean ± std)")
    print("=" * 70)
    header = f"{'λ \\ ρ':>10s}" + "".join(f"  {r:>8.2f}" for r in rhos)
    print(header)
    print("-" * len(header))
    for i, lamb in enumerate(lambs):
        row = f"{lamb:>10.0e}"
        for j in range(len(rhos)):
            m = np.nanmean(grid["orth_err_normalized"][i, j, :])
            s = np.nanstd(grid["orth_err_normalized"][i, j, :])
            row += f"  {m:>5.3f}±{s:<4.3f}"
        print(row)


def save_flat_csv(records, out_dir):
    """Save a flat CSV of all runs for easy downstream analysis."""
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "all_results.csv")
    cols = ["lamb", "rho", "seed", "D", "steps", "mcc", "linear_r2",
            "orth_err", "orth_err_normalized"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({c: r[c] for c in cols})
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="results/")
    p.add_argument("--out",         type=str, default="figures/")
    args = p.parse_args()

    records = load_all(args.results_dir)
    if not records:
        print("No results found!")
        exit(1)

    grid, lambs, rhos, seeds = build_grid(records)
    print_summary_table(grid, lambs, rhos, seeds)
    plot_heatmaps(grid, lambs, rhos, seeds, args.out)
    plot_training_curves(records, args.out)
    save_flat_csv(records, args.out)
    print("\nDone!")