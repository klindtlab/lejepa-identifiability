"""
Laplace ablation: 4 rows (mixings) x 4 cols (z, g(z), h_lejepa, h_whiten).
Picks best seed per (mixing, mode) by final_loss.

Usage:
    python analysis/plot_laplace.py --results_dir results/laplace/ --out figures/
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


def load_best(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "*.pt")))
    by_key = {}
    for path in files:
        r = torch.load(path, map_location="cpu", weights_only=False)
        key = r["run_name"]  # e.g. "spiral_lejepa"
        if key not in by_key or r["final_loss"] < by_key[key]["final_loss"]:
            by_key[key] = r
    return by_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/laplace/")
    p.add_argument("--out", default="figures/")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    best = load_best(args.results_dir)

    s, lim = 5, 4
    for mix in MIXING_ORDER:
        lej_key = f"{mix}_lejepa"
        wht_key = f"{mix}_whiten"
        if lej_key not in best or wht_key not in best:
            print(f"Missing {mix}"); continue

        lej, wht = best[lej_key], best[wht_key]
        z = lej["z"]
        colors = make_colors(z)

        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        col_labels = [
            ("True Latent 0", "True Latent 1"),
            ("Observation 0", "Observation 1"),
            ("Learned (LeJEPA) 0", "Learned (LeJEPA) 1"),
            ("Learned (Whiten) 0", "Learned (Whiten) 1"),
        ]
        panels = [z, lej["x"], lej["h"], wht["h"]]
        r2s = [None, None, lej["r2_hz"], wht["r2_hz"]]

        for i, (ax, data, labels, r2) in enumerate(zip(axes, panels, col_labels, r2s)):
            ax.scatter(data[:, 0], data[:, 1], c=colors, s=s, linewidths=0)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.grid(alpha=0.3)
            # if r2 is not None:
            #     ax.text(0.95, 0.05, f"$R^2$={r2:.3f}", transform=ax.transAxes,
            #             ha="right", va="bottom", fontsize=9,
            #             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        fig.tight_layout()
        out_path = os.path.join(args.out, f"fig_laplace_{mix}.jpg")
        fig.savefig(out_path, bbox_inches="tight", dpi=500)
        print(f"Saved {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
