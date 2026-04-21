"""
Reacher trajectory distribution analysis figures.

Produces two figures for the paper:
  1. Scatter grid: stationary marginal + per-delta 2D transition differences
     and per-dim (z_t, z_{t+delta}) scatters, annotated with R² and rho.
  2. rho-vs-SIGReg scatter: three panels (z_0, z_1, joint), colored by R²,
     showing the dual constraint that identifiability requires both
     rho off from 1 and approximately-Gaussian transition shape.

Usage:
    python -m analysis.make_reacher_distributions \
        --results_dir results/reacher \
        --data_path data/reacher.h5 \
        --out_dir figures/reacher
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr


DELTAS = [1, 2, 4, 8, 16, 32, 64]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MAX = 100_000


# ═════════════════════════════════════════════════════════════════════════════
# SIGReg (Epps–Pulley) — matches LeJEPA Algorithm 1
# ═════════════════════════════════════════════════════════════════════════════

def _sigreg_nd(x, num_slices=64, n_knots=17, seed=0, device=DEVICE):
    """EP via random slicing on (N, K)."""
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    if x.dim() == 1:
        x = x[:, None]
    N, K = x.shape
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(K, num_slices, generator=g, device=device)
    A = A / A.norm(p=2, dim=0)
    t = torch.linspace(-5, 5, n_knots, device=device)
    phi = torch.exp(-0.5 * t ** 2)
    zt = (x @ A).unsqueeze(2) * t
    cm, sm = torch.cos(zt).mean(0), torch.sin(zt).mean(0)
    err = ((cm - phi) ** 2 + sm ** 2) * phi
    return (torch.trapz(err, t, dim=1) * N).mean().item()


def _sigreg_1d(x, n_knots=17, device=DEVICE):
    """EP directly on 1D (no slicing)."""
    x = torch.as_tensor(np.asarray(x).reshape(-1), dtype=torch.float32, device=device)
    N = x.shape[0]
    t = torch.linspace(-5, 5, n_knots, device=device)
    phi = torch.exp(-0.5 * t ** 2)
    zt = x.unsqueeze(1) * t
    cm, sm = torch.cos(zt).mean(0), torch.sin(zt).mean(0)
    err = ((cm - phi) ** 2 + sm ** 2) * phi
    return (torch.trapz(err, t) * N).item()


def _zscore(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)


def measure(x, n_draws=20, N_max=N_MAX, num_slices=64):
    """
    SIGReg raw + zscored for joint (K-d) and per-dim marginals.
    Averages over n_draws random subsamples / projections.
    Returns dict mapping key -> (mean, std) over draws.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    N, K = x.shape
    rng = np.random.default_rng(0)
    keys = ["joint_raw", "joint_zs"]
    keys += [f"marg_{k}_raw" for k in range(K)]
    keys += [f"marg_{k}_zs" for k in range(K)]
    buf = {k: [] for k in keys}
    for s in range(n_draws):
        xs = x if N <= N_max else x[rng.choice(N, N_max, replace=False)]
        xz = _zscore(xs)
        buf["joint_raw"].append(_sigreg_nd(xs, num_slices, seed=s))
        buf["joint_zs"].append(_sigreg_nd(xz, num_slices, seed=s))
        for k in range(K):
            buf[f"marg_{k}_raw"].append(_sigreg_1d(xs[:, k]))
            buf[f"marg_{k}_zs"].append(_sigreg_1d(xz[:, k]))
    return {k: (np.mean(v), np.std(v)) for k, v in buf.items()}


# ═════════════════════════════════════════════════════════════════════════════
# R² loading — per-seed best-lambda tuning, median across seeds
# ═════════════════════════════════════════════════════════════════════════════

def get_best_r2_per_delta(results_dir, agg="median"):
    """
    For each (delta, seed), pick the lambda with best R²; aggregate seeds
    with median (robust to outliers) or mean.
    """
    by_dls = defaultdict(list)
    for p in Path(results_dir).rglob("result.json"):
        r = json.load(open(p))
        if "delta" not in r or r.get("rho") is not None:
            continue
        by_dls[(r["delta"], r["lamb"], r.get("seed", 0))].append(r)

    deltas = sorted({k[0] for k in by_dls})
    lambs = sorted({k[1] for k in by_dls})
    seeds = sorted({k[2] for k in by_dls})
    agg_fn = np.median if agg == "median" else np.mean

    best = {}
    for delta in deltas:
        per_seed = {"r2": [], "d0": [], "d1": [], "lamb": []}
        for seed in seeds:
            best_lamb, best_r2 = None, -np.inf
            for lamb in lambs:
                runs = by_dls.get((delta, lamb, seed), [])
                if not runs:
                    continue
                r2 = np.mean([r["r2_hz"] for r in runs])
                if r2 > best_r2:
                    best_r2, best_lamb = r2, lamb
            if best_lamb is None:
                continue
            runs = by_dls[(delta, best_lamb, seed)]
            per_seed["r2"].append(np.mean([r["r2_hz"] for r in runs]))
            per_seed["d0"].append(np.mean([r["r2_hz_per_dim"][0] for r in runs]))
            per_seed["d1"].append(np.mean([r["r2_hz_per_dim"][1] for r in runs]))
            per_seed["lamb"].append(best_lamb)
        best[delta] = {
            "r2": agg_fn(per_seed["r2"]),
            "r2_dim0": agg_fn(per_seed["d0"]),
            "r2_dim1": agg_fn(per_seed["d1"]),
            "lamb": np.median(per_seed["lamb"]),
            "n_seeds": len(per_seed["r2"]),
        }
    return best


def best_ou_rho(results_dir):
    """Return the rho of the OU run with highest mean R² across seeds/lambdas."""
    grouped = defaultdict(list)
    for p in Path(results_dir).rglob("result.json"):
        r = json.load(open(p))
        if "rho" not in r or r.get("rho") is None:
            continue
        grouped[(r["rho"], r["lamb"])].append(r)
    if not grouped:
        return None
    best_rho, best_mean = None, -np.inf
    for (rho, lamb), runs in grouped.items():
        m = np.mean([r["r2_hz"] for r in runs])
        if m > best_mean:
            best_mean, best_rho = m, rho
    return best_rho


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1: scatter grid
# ═════════════════════════════════════════════════════════════════════════════

def make_scatter_grid(episodes, r2_dict, save_path, sub=1, s=1e-4):
    """
    Left column: stationary marginal scatter of (z_0, z_1).
    Top row, remaining columns: 2D transition-difference scatter per delta.
    Bottom row, remaining columns: per-dim (z_t, z_{t+delta}) scatter.
    Titles show R² and rho.
    """
    fig = plt.figure(figsize=0.85 * np.array((1 + 3 * len(DELTAS), 5)))
    gs = fig.add_gridspec(2, 2 + len(DELTAS))

    # stationary marginal (spans both rows, first two cols)
    ax = fig.add_subplot(gs[:2, :2])
    ax.scatter(*episodes.reshape(-1, 2)[::sub].T, s=s * 10)
    ax.set_title("Marginal")
    ax.grid()
    ax.set_xlabel(r"$z_0$ (shoulder)")
    ax.set_ylabel(r"$z_1$ (wrist)")

    for i, delta in enumerate(DELTAS):
        # top row: 2D transition differences
        ax = fig.add_subplot(gs[0, 2 + i])
        transitions = episodes[:, delta:] - episodes[:, :-delta]
        ax.scatter(*transitions.reshape(-1, 2)[::sub].T, s=s)
        r2_d0 = r2_dict[delta]["r2_dim0"]
        r2_d1 = r2_dict[delta]["r2_dim1"]
        ax.set_title(r"$\Delta=$" + f"{delta}" + "\n"
                     r"$R^2=(%.2f, %.2f)$" % (r2_d0, r2_d1))
        ax.grid()

        # bottom row: per-dim (z_t, z_{t+delta}) with rho
        ax = fig.add_subplot(gs[1, 2 + i])
        a = episodes[:, delta:, 0].flatten()[::sub]
        b = episodes[:, :-delta, 0].flatten()[::sub]
        rho0 = pearsonr(a, b)[0]
        ax.scatter(a, b, s=s)
        c = episodes[:, delta:, 1].flatten()[::sub]
        d = episodes[:, :-delta, 1].flatten()[::sub]
        rho1 = pearsonr(c, d)[0]
        ax.scatter(c, d, s=s)
        ax.set_title(r"$\rho=(%.2f, %.2f)$" % (rho0, rho1))
        ax.grid()
        if i == 0:
            ax.legend([r"$z_0$ (shoulder)", r"$z_1$ (wrist)"], loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2: rho vs SIGReg scatter (3 panels)
# ═════════════════════════════════════════════════════════════════════════════

def make_rho_vs_sigreg(trans, r2_dict, save_path, best_ou_rho_val=None,
                       gaussian_floor=1.2):
    """
    Three panels (z_0, z_1, joint) showing rho vs SIGReg(zscored),
    colored by R². Vertical line marks the best OU rho for reference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    names = [r"$z_0$ (shoulder)", r"$z_1$ (wrist)", "joint (avg across dims)"]

    all_r2 = []
    for d in DELTAS:
        all_r2.append(r2_dict[d]["r2_dim0"])
        all_r2.append(r2_dict[d]["r2_dim1"])
        all_r2.append(r2_dict[d]["r2"])
    vmin, vmax = min(all_r2), max(all_r2)

    def _one(ax, rhos, sigs, errs, r2s, xlabel, ylabel, title):
        if best_ou_rho_val is not None:
            ax.axvline(best_ou_rho_val, color="crimson", lw=1.4, ls="--",
                       alpha=0.8, label=fr"best OU $\rho={best_ou_rho_val:.2f}$",
                       zorder=1)
        sc = ax.scatter(rhos, sigs, c=r2s, cmap="viridis", s=140,
                        vmin=vmin, vmax=vmax,
                        edgecolors="black", linewidths=0.8, zorder=3)
        ax.errorbar(rhos, sigs, yerr=errs, fmt="none", ecolor="gray",
                    alpha=0.5, zorder=2)
        for d, r, s in zip(DELTAS, rhos, sigs):
            ax.annotate(f"Δ={d}", (r, s), xytext=(6, 6),
                        textcoords="offset points", fontsize=9)
        ax.axhline(gaussian_floor, color="red", lw=1, ls=":", alpha=0.6,
                   label="Gaussian floor")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="lower left", fontsize=8)
        return sc

    # per-dim panels
    for k in range(2):
        rhos = np.array([trans[d]["rho"][k] for d in DELTAS])
        sigs = np.array([trans[d]["sig"][f"marg_{k}_zs"][0] for d in DELTAS])
        errs = np.array([trans[d]["sig"][f"marg_{k}_zs"][1] for d in DELTAS])
        r2s = np.array([r2_dict[d][f"r2_dim{k}"] for d in DELTAS])
        sc = _one(axes[k], rhos, sigs, errs, r2s,
                  xlabel=r"auto-correlation $\rho$",
                  ylabel="SIGReg (zscored, marginal)",
                  title=names[k])
        plt.colorbar(sc, ax=axes[k], label=r"$R^2$")

    # joint panel
    rhos_avg = np.array([np.mean(trans[d]["rho"]) for d in DELTAS])
    sigs_j = np.array([trans[d]["sig"]["joint_zs"][0] for d in DELTAS])
    errs_j = np.array([trans[d]["sig"]["joint_zs"][1] for d in DELTAS])
    r2s_avg = np.array([r2_dict[d]["r2"] for d in DELTAS])
    sc = _one(axes[2], rhos_avg, sigs_j, errs_j, r2s_avg,
              xlabel=r"avg auto-correlation $\bar{\rho}$",
              ylabel="SIGReg (zscored, 2D joint)",
              title=names[2])
    plt.colorbar(sc, ax=axes[2], label=r"$R^2$ (avg)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Measurement pipeline
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_transitions(episodes, n_draws=20):
    """
    For each delta, compute SIGReg stats and per-dim rho on the
    transition-difference distribution z(t+delta) - z(t).
    """
    trans = {}
    for d in DELTAS:
        diffs = (episodes[:, d:] - episodes[:, :-d]).reshape(-1, 2)
        rho = np.array([
            pearsonr(episodes[:, d:, k].flatten(),
                     episodes[:, :-d, k].flatten())[0]
            for k in range(2)
        ])
        trans[d] = {"rho": rho, "sig": measure(diffs, n_draws=n_draws)}
    return trans


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/reacher",
                        help="Directory with result.json files")
    parser.add_argument("--data_path", type=str,
                        default="data/reacher.h5",
                        help="HDF5 file with 'qpos' and 'ep_len' datasets "
                        "(reshaped to (n_episodes, T, 2))")
    parser.add_argument("--out_dir", type=str, default="figures/reacher")
    parser.add_argument("--n_draws", type=int, default=20,
                        help="Random subsamples for SIGReg stats")
    parser.add_argument("--agg", type=str, default="median",
                        choices=["median", "mean"],
                        help="How to aggregate R² across seeds")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load episodes (HDF5 with qpos + ep_len, matching notebook convention)
    import h5py
    with h5py.File(args.data_path, "r") as f:
        qpos = np.array(f["qpos"])
        ep_len = np.array(f["ep_len"])
    T = int(ep_len[0])
    episodes = qpos.reshape(-1, T, 2)
    print(f"Loaded {len(episodes)} episodes of length {T} from {args.data_path}")

    r2_dict = get_best_r2_per_delta(args.results_dir, agg=args.agg)
    print(f"Loaded R² for deltas: {sorted(r2_dict.keys())}")
    for d in DELTAS:
        v = r2_dict[d]
        print(f"  Δ={d:2d}  λ={v['lamb']:.0e}  n={v['n_seeds']}  "
              f"R²={v['r2']:.3f}  dim0={v['r2_dim0']:.3f}  "
              f"dim1={v['r2_dim1']:.3f}")

    ou_rho = best_ou_rho(args.results_dir)
    if ou_rho is not None:
        print(f"Best OU rho: {ou_rho}")

    # measure transitions
    print("Computing SIGReg on transitions...")
    trans = compute_all_transitions(episodes, n_draws=args.n_draws)

    # figures
    make_scatter_grid(episodes, r2_dict,
                      save_path=out_dir / "distribution.png")
    make_rho_vs_sigreg(trans, r2_dict,
                       save_path=out_dir / "rho_vs_sigreg.png",
                       best_ou_rho_val=ou_rho)


if __name__ == "__main__":
    main()