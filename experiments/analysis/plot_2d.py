"""
2D illustration figure: one row per mixing (z, g(z), h(z)).
Picks best seed per mixing by final_loss.

Usage:
    python analysis/plot_2d.py --results_dir results/2d/ --out figures/
"""

import argparse, os, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import colorsys

MIXING_ORDER = ["spiral", "banana", "sinusoid", "nvp"]


def make_colors(z):
    x, y = z[:, 0], z[:, 1]
    angles = np.arctan2(y, x)
    radii = np.sqrt(x**2 + y**2)
    hue = (angles + np.pi) / (2 * np.pi)
    lightness = 0.3 + 0.4 * (radii / (radii.max() + 1e-8))
    saturation = np.full_like(hue, 0.85)
    return [colorsys.hls_to_rgb(h, l, s) for h, l, s in zip(hue, lightness, saturation)]


def load_best_per_mixing(results_dir):
    """Load all results, pick best lejepa run per mixing by final_loss."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.pt")))
    by_mix = {}
    for path in files:
        r = torch.load(path, map_location="cpu", weights_only=False)
        if r.get("mode") != "lejepa":
            continue
        mix = r["mixing"]
        if mix not in by_mix or r["final_loss"] < by_mix[mix]["final_loss"]:
            by_mix[mix] = r
    return by_mix


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/2d/")
    p.add_argument("--out", default="figures/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    best = load_best_per_mixing(args.results_dir)

    s, lim = 5, 4
    for mix_name in MIXING_ORDER:
        if mix_name not in best:
            print(f"Missing {mix_name}"); continue
        res = best[mix_name]
        z, x, h = res["z"], res["x"], res["h"]
        colors = make_colors(z)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for i, (ax, data, labels) in enumerate(zip(
            axes, [z, x, h],
            [("True Latent 0", "True Latent 1"),
             ("Observation 0", "Observation 1"),
             ("Learned Latent 0", "Learned Latent 1")],
        )):
            ax.scatter(data[:, 0], data[:, 1], c=colors, s=s, linewidths=0)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.grid(alpha=0.3)
            if i == 0 or i == 2:
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)

        fig.tight_layout()
        out_path = os.path.join(args.out, f"fig_2d_{mix_name}.jpg")
        fig.savefig(out_path, bbox_inches="tight", dpi=500)
        print(f"Saved {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
