"""
Aggregate results from any experiment into a flat CSV.

Usage:
    python analysis/aggregate.py --results_dir results/2d/ --out results/2d/summary.csv
    python analysis/aggregate.py --results_dir results/ --recursive --out results/all.csv
"""

import argparse, glob, json, os
import pandas as pd


SCALAR_KEYS = [
    "experiment", "run_name", "mixing", "encoder", "mode", "source_dist",
    "seed", "N", "lamb", "rho", "lr", "steps", "batch_size", "n_layers", "hidden",
    "r2_zx", "r2_xz", "r2_zh", "r2_hz",
    "orth_err", "orth_err_normalized",
    "epsilon", "delta", "D_bound", "approx_bound",
    "procrustes_mse", "L_h", "trace_cov",
    "final_align", "final_sigreg", "final_whiten", "final_loss",
]


def load_results(results_dir, recursive=False):
    pattern = os.path.join(results_dir, "**/*.json") if recursive else os.path.join(results_dir, "*.json")
    files = sorted(glob.glob(pattern, recursive=recursive))
    print(f"Found {len(files)} .json files")

    rows = []
    for path in files:
        try:
            with open(path) as f:
                r = json.load(f)
            row = {k: r.get(k) for k in SCALAR_KEYS}
            row["file"] = os.path.relpath(path, results_dir)
            rows.append(row)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--recursive", action="store_true")
    args = p.parse_args()

    df = load_results(args.results_dir, recursive=args.recursive)
    if len(df) == 0:
        print("No results found.")
        return

    print(f"\n{len(df)} runs loaded")
    print(df.to_string(index=False))

    out = args.out or os.path.join(args.results_dir, "summary.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
