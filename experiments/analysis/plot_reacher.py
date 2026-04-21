"""
Aggregate and plot Reacher sweep results.

Generates:
  1. OU: R² vs ρ (per lambda)
  2. OU: R² vs ρ (per dimension)
  3. Traj: per-dim R² vs δ (with per-dim ρ annotations)
  4. OU vs Traj on same axes (ρ on x-axis, traj uses measured ρ_mean)
  5. Lambda robustness panel (OU, R² vs λ for each ρ)
  6. Traj: R² vs measured ρ
  7. Orthogonality error plots

Usage:
    python analysis/plot_reacher.py --results_dir results/reacher
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_summaries(results_dir):
    """Load all summary_*.json files, split into OU and traj."""
    ou_results, traj_results = [], []
    for p in sorted(Path(results_dir).glob("summary_*.json")):
        with open(p) as f:
            summary = json.load(f)
        for run_name, r in summary.items():
            if r.get("r2_hz", -999) <= -1:
                continue
            if "delta" in r:
                traj_results.append(r)
            else:
                ou_results.append(r)
    return ou_results, traj_results


def _group_by(results, x_key):
    """Group results by (x_key, lamb) → list of result dicts."""
    grouped = defaultdict(list)
    for r in results:
        grouped[(r[x_key], r["lamb"])].append(r)
    return grouped


def _get_sorted(results, key):
    return sorted(set(r[key] for r in results))


# ═════════════════════════════════════════════════════════════════════════════
# 1. OU: R² vs ρ (one curve per lambda)
# ═════════════════════════════════════════════════════════════════════════════

def plot_ou_r2_vs_rho(results, save_path):
    grouped = _group_by(results, "rho")
    lambs = _get_sorted(results, "lamb")
    rhos = _get_sorted(results, "rho")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lamb in lambs:
        medians, q25, q75, xs = [], [], [], []
        for rho in rhos:
            vals = [r["r2_hz"] for r in grouped.get((rho, lamb), [])]
            if vals:
                medians.append(np.median(vals))
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
                xs.append(rho)
        medians, q25, q75 = np.array(medians), np.array(q25), np.array(q75)
        ax.plot(xs, medians, "o-", label=f"λ={lamb:.0e}", markersize=5)
        ax.fill_between(xs, q25, q75, alpha=0.15)

    ax.set_xlabel("ρ (OU autocorrelation)", fontsize=12)
    ax.set_ylabel("R² (embed → true state)", fontsize=12)
    ax.set_title("OU: Linear identifiability vs. ρ", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. OU: per-dimension R² vs ρ
# ═════════════════════════════════════════════════════════════════════════════

def plot_ou_perdim_r2(results, save_path):
    """Two curves: shoulder vs wrist, averaged over lambda and seeds."""
    by_rho = defaultdict(list)
    for r in results:
        if "r2_hz_per_dim" in r:
            by_rho[r["rho"]].append(r["r2_hz_per_dim"])

    rhos = sorted(by_rho.keys())
    dim0_med, dim1_med = [], []
    for rho in rhos:
        vals = np.array(by_rho[rho])
        dim0_med.append(np.median(vals[:, 0]))
        dim1_med.append(np.median(vals[:, 1]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rhos, dim0_med, "o-", label="Shoulder (dim 0)", markersize=5)
    ax.plot(rhos, dim1_med, "s-", label="Wrist (dim 1)", markersize=5)

    ax.set_xlabel("ρ (OU autocorrelation)", fontsize=12)
    ax.set_ylabel("R² per dimension", fontsize=12)
    ax.set_title("OU: Per-dimension identifiability", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 3. Traj: per-dimension R² vs δ (with ρ annotations)
# ═════════════════════════════════════════════════════════════════════════════

def plot_traj_perdim_r2(results, save_path):
    """Two curves: shoulder vs wrist, annotated with per-dim ρ."""
    by_delta = defaultdict(list)
    for r in results:
        if "r2_hz_per_dim" in r:
            by_delta[r["delta"]].append(r)

    deltas = sorted(by_delta.keys())
    dim0_med, dim1_med = [], []
    rho0_vals, rho1_vals = [], []

    for delta in deltas:
        runs = by_delta[delta]
        vals = np.array([r["r2_hz_per_dim"] for r in runs])
        dim0_med.append(np.median(vals[:, 0]))
        dim1_med.append(np.median(vals[:, 1]))
        rho0 = [r.get("rho_shoulder", None) for r in runs]
        rho1 = [r.get("rho_wrist", None) for r in runs]
        rho0_vals.append(rho0[0] if rho0[0] is not None else None)
        rho1_vals.append(rho1[0] if rho1[0] is not None else None)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(deltas, dim0_med, "o-", label="Shoulder (dim 0)",
            markersize=5, color="C0")
    ax.plot(deltas, dim1_med, "s-", label="Wrist (dim 1)",
            markersize=5, color="C1")

    # Annotate with ρ values
    for i, delta in enumerate(deltas):
        if rho0_vals[i] is not None:
            y_pos = max(dim0_med[i], dim1_med[i]) + 0.03
            ax.annotate(f"ρ₀={rho0_vals[i]:.3f}\nρ₁={rho1_vals[i]:.3f}",
                        (delta, y_pos),
                        fontsize=7, ha="center", alpha=0.7)

    ax.set_xlabel("δ (temporal stride)", fontsize=12)
    ax.set_ylabel("R² per dimension", fontsize=12)
    ax.set_title("Trajectory: Per-dimension identifiability vs. δ", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.15, 1.15)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. OU vs Traj on same axes (ρ on x-axis)
# ═════════════════════════════════════════════════════════════════════════════

def plot_ou_vs_traj(ou_results, traj_results, save_path):
    """Both conditions on one plot, using ρ as common x-axis."""
    # OU: group by rho, average over lambda and seeds
    ou_by_rho = defaultdict(list)
    for r in ou_results:
        ou_by_rho[r["rho"]].append(r["r2_hz"])

    # Traj: use rho_mean, group by delta
    traj_by_rho = defaultdict(list)
    for r in traj_results:
        rho_mean = r.get("rho_mean", None)
        if rho_mean is not None:
            traj_by_rho[rho_mean].append(r["r2_hz"])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # OU
    rhos_ou = sorted(ou_by_rho.keys())
    med_ou = [np.median(ou_by_rho[rho]) for rho in rhos_ou]
    q25_ou = [np.percentile(ou_by_rho[rho], 25) for rho in rhos_ou]
    q75_ou = [np.percentile(ou_by_rho[rho], 75) for rho in rhos_ou]
    ax.plot(rhos_ou, med_ou, "o-", label="OU (Gaussian)", markersize=6,
            color="C0", linewidth=2)
    ax.fill_between(rhos_ou, q25_ou, q75_ou, alpha=0.15, color="C0")

    # Traj
    rhos_traj = sorted(traj_by_rho.keys())
    med_traj = [np.median(traj_by_rho[rho]) for rho in rhos_traj]
    q25_traj = [np.percentile(traj_by_rho[rho], 25) for rho in rhos_traj]
    q75_traj = [np.percentile(traj_by_rho[rho], 75) for rho in rhos_traj]
    ax.plot(rhos_traj, med_traj, "s-", label="Trajectory (non-Gaussian)",
            markersize=6, color="C3", linewidth=2)
    ax.fill_between(rhos_traj, q25_traj, q75_traj, alpha=0.15, color="C3")

    ax.set_xlabel("ρ (autocorrelation)", fontsize=12)
    ax.set_ylabel("R² (embed → true state)", fontsize=12)
    ax.set_title("Gaussian vs. non-Gaussian latents", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 5. Lambda robustness (OU)
# ═════════════════════════════════════════════════════════════════════════════

def plot_lambda_robustness(results, save_path):
    """R² vs lambda for each rho."""
    grouped = _group_by(results, "rho")
    rhos = _get_sorted(results, "rho")
    lambs = _get_sorted(results, "lamb")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rho in rhos:
        medians, xs = [], []
        for lamb in lambs:
            vals = [r["r2_hz"] for r in grouped.get((rho, lamb), [])]
            if vals:
                medians.append(np.median(vals))
                xs.append(lamb)
        if medians:
            ax.plot(xs, medians, "o-", label=f"ρ={rho}", markersize=4)

    ax.set_xscale("log")
    ax.set_xlabel("λ (SIGReg weight)", fontsize=12)
    ax.set_ylabel("R² (embed → true state)", fontsize=12)
    ax.set_title("OU: Robustness to λ", fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 6. Traj: R² vs measured ρ (per lambda)
# ═════════════════════════════════════════════════════════════════════════════

def plot_traj_r2_vs_rho(results, save_path):
    """Traj R² plotted against measured autocorrelation, per lambda."""
    grouped = defaultdict(list)
    for r in results:
        rho_mean = r.get("rho_mean", None)
        if rho_mean is not None:
            grouped[(rho_mean, r["lamb"])].append(r["r2_hz"])

    lambs = _get_sorted(results, "lamb")
    rhos = sorted(set(k[0] for k in grouped.keys()))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lamb in lambs:
        medians, xs = [], []
        for rho in rhos:
            vals = grouped.get((rho, lamb), [])
            if vals:
                medians.append(np.median(vals))
                xs.append(rho)
        if medians:
            ax.plot(xs, medians, "o-", label=f"λ={lamb:.0e}", markersize=5)

    ax.set_xlabel("ρ (measured autocorrelation)", fontsize=12)
    ax.set_ylabel("R² (embed → true state)", fontsize=12)
    ax.set_title("Trajectory: Identifiability vs. measured ρ", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. Orthogonality error
# ═════════════════════════════════════════════════════════════════════════════

def plot_orth_err(results, x_key, x_label, title, save_path):
    grouped = _group_by(results, x_key)
    lambs = _get_sorted(results, "lamb")
    x_vals = _get_sorted(results, x_key)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lamb in lambs:
        medians, xs = [], []
        for x in x_vals:
            vals = [r.get("orth_error", r.get("procrustes_error", 1.0))
                    for r in grouped.get((x, lamb), [])]
            if vals:
                medians.append(np.median(vals))
                xs.append(x)
        ax.plot(xs, medians, "o-", label=f"λ={lamb:.0e}", markersize=4)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Orthogonality error", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_table(results, x_key, label):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r[x_key], r["lamb"])].append(r)

    print(f"\n=== {label} ===")
    print(f"{x_key:>6s} {'lamb':>8s} {'R²(h→z)':>10s} "
          f"{'R²(dim0)':>10s} {'R²(dim1)':>10s} "
          f"{'R²(sincos)':>10s} {'orth_err':>10s} {'n':>4s}")
    print("-" * 76)
    for (x, lamb), runs in sorted(grouped.items()):
        r2s = [r["r2_hz"] for r in runs]
        errs = [r.get("orth_error", r.get("procrustes_error", 1.0))
                for r in runs]

        dim0 = [r["r2_hz_per_dim"][0] for r in runs
                if "r2_hz_per_dim" in r]
        dim1 = [r["r2_hz_per_dim"][1] for r in runs
                if "r2_hz_per_dim" in r]
        sincos = [r["r2_sincos"] for r in runs if "r2_sincos" in r]

        d0 = f"{np.median(dim0):10.4f}" if dim0 else f"{'n/a':>10s}"
        d1 = f"{np.median(dim1):10.4f}" if dim1 else f"{'n/a':>10s}"
        sc = f"{np.median(sincos):10.4f}" if sincos else f"{'n/a':>10s}"

        print(f"{x:6g} {lamb:8.1e} "
              f"{np.median(r2s):10.4f} "
              f"{d0} {d1} "
              f"{sc} "
              f"{np.median(errs):10.4f} "
              f"{len(runs):4d}")


def plot_traj_distributions(h5_path, results_dir, save_path):
    """Marginal + transition distributions for trajectory data, annotated with R²."""
    import h5py
    from scipy.stats import pearsonr

    with h5py.File(h5_path, "r") as f:
        qpos = np.array(f["qpos"])
        ep_len = np.array(f["ep_len"])
    T = ep_len[0]
    episodes = qpos.reshape(-1, T, 2)

    # Load R² per delta
    grouped = defaultdict(list)
    for p in Path(results_dir).rglob("result.json"):
        r = json.load(open(p))
        if "delta" not in r or r.get("rho") is not None:
            continue
        grouped[(r["delta"], r["lamb"])].append(r)

    r2_dict = {}
    for delta in set(k[0] for k in grouped):
        best_mean, best_lamb = -np.inf, None
        for lamb in set(k[1] for k in grouped if k[0] == delta):
            m = np.mean([r["r2_hz"] for r in grouped[(delta, lamb)]])
            if m > best_mean:
                best_mean, best_lamb = m, lamb
        runs = grouped[(delta, best_lamb)]
        r2_dict[delta] = {
            "r2_dim0": np.mean([r["r2_hz_per_dim"][0] for r in runs]),
            "r2_dim1": np.mean([r["r2_hz_per_dim"][1] for r in runs]),
        }

    DELTAS = [1, 2, 4, 8, 16, 32, 64]
    s = 0.0001
    sub = 1

    fig = plt.figure(figsize=0.85 * np.array((1 + 3 * len(DELTAS), 5)))
    gs = fig.add_gridspec(2, 2 + len(DELTAS))

    ax = fig.add_subplot(gs[:2, :2])
    ax.scatter(*episodes.reshape(-1, 2)[::sub].T, s=s * 10)
    ax.set_title("Marginal")
    ax.grid()
    ax.set_xlabel(r"$z_0$ (shoulder)")
    ax.set_ylabel(r"$z_1$ (wrist)")

    for i, delta in enumerate(DELTAS):
        # Transition scatter
        ax = fig.add_subplot(gs[0, 2 + i])
        transitions = (episodes[:, delta:] - episodes[:, :-delta]).reshape(-1, 2)
        ax.scatter(*transitions[::sub].T, s=s)
        ax.set_title(
            r"$\Delta=%d$" % delta + "\n"
            + r"$R^2=(%.2f,\,%.2f)$"
            % (r2_dict[delta]["r2_dim0"], r2_dict[delta]["r2_dim1"])
        )
        ax.grid()

        # Autocorrelation scatter
        ax = fig.add_subplot(gs[1, 2 + i])
        a = episodes[:, delta:, 0].flatten()[::sub]
        b = episodes[:, :-delta, 0].flatten()[::sub]
        rho0 = pearsonr(a, b)[0]
        ax.scatter(a, b, s=s)

        c = episodes[:, delta:, 1].flatten()[::sub]
        d = episodes[:, :-delta, 1].flatten()[::sub]
        rho1 = pearsonr(c, d)[0]
        ax.scatter(c, d, s=s)

        ax.set_title(r"$\rho=(%.2f,\,%.2f)$" % (rho0, rho1))
        ax.grid()
        if i == 0:
            ax.legend([r"$z_0$ (shoulder)", r"$z_1$ (wrist)"], loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight", format="jpg")
    plt.close()
    print(f"Saved {save_path}")
    

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/reacher")
    parser.add_argument("--out_dir", type=str, default="figures/reacher")
    parser.add_argument("--h5_path", type=str, default="data/reacher.h5")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ou_results, traj_results = load_summaries(args.results_dir)
    print(f"Loaded {len(ou_results)} OU runs, {len(traj_results)} traj runs")

    # Tables
    if ou_results:
        print_table(ou_results, "rho", "OU")
    if traj_results:
        print_table(traj_results, "delta", "Trajectory")

    # OU plots
    if ou_results:
        plot_ou_r2_vs_rho(ou_results, out_dir / "ou_r2_vs_rho.png")
        plot_ou_perdim_r2(ou_results, out_dir / "ou_perdim_r2.png")
        plot_lambda_robustness(ou_results, out_dir / "ou_lambda_robustness.png")
        plot_orth_err(ou_results, "rho", "ρ",
                      "OU: Orthogonality error vs. ρ",
                      out_dir / "ou_orth_err.png")

    # Traj plots
    if traj_results:
        plot_traj_perdim_r2(traj_results, out_dir / "traj_perdim_r2.png")
        plot_traj_r2_vs_rho(traj_results, out_dir / "traj_r2_vs_rho.png")
        plot_orth_err(traj_results, "delta", "δ",
                      "Trajectory: Orthogonality error vs. δ",
                      out_dir / "traj_orth_err.png")
        
        h5_path = os.path.join(args.h5_path)
        if os.path.exists(h5_path):
            plot_traj_distributions(h5_path, args.results_dir,
                                     out_dir / "traj_distributions.png")

    # Combined
    if ou_results and traj_results:
        plot_ou_vs_traj(ou_results, traj_results,
                        out_dir / "ou_vs_traj.png")


if __name__ == "__main__":
    main()