"""
Generate LaTeX table from scaling results.

Usage:
    python analysis/make_table_scaling.py --results_dir results/scaling/
"""

import argparse, glob, json
import numpy as np


def column_scale(stds, threshold=0.05):
    """Return exponent k such that std / 10^k displays as a small number.
    Returns 0 (no scaling) if all stds are >= threshold."""
    valid = [s for s in stds if s > 0]
    if not valid:
        return 0
    m = max(valid)
    if m >= threshold:
        return 0
    # scaled max in [10, 100)
    return int(np.floor(np.log10(m))) - 1


def fmt_std(s, k):
    """Format std after dividing by 10^k."""
    scaled = s if k == 0 else s / (10 ** k)
    if scaled >= 10:
        return f"{scaled:.0f}"
    elif scaled >= 1:
        return f"{scaled:.1f}"
    else:
        return f"{scaled:.2f}"


def scale_header(k):
    """Sub-header label for a scaled column."""
    if k == 0:
        return r"{\scriptsize $\pm$std}"
    return rf"{{\scriptsize $\pm$std\,$\times 10^{{{k}}}$}}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/scaling/")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{args.results_dir}/*.json")):
        with open(path) as f:
            rows.append(json.load(f))

    dims = sorted(set(r["N"] for r in rows))

    # Columns: (key, mean_decimals, label)
    cols = [
        ("r2_zx",                3, r"$R^2(z \to x)$"),
        ("r2_xz",                3, r"$R^2(x \to z)$"),
        ("r2_zh",                5, r"$R^2(z \to h)$"),
        ("r2_hz",                5, r"$R^2(h \to z)$"),
        ("orth_err_normalized", 3, r"$\|\hat Q^\top \hat Q - I\|_F / \sqrt{N}$"),
        ("final_align",          4, "Align"),
        ("final_sigreg",         2, "SIGReg"),
    ]

    # Aggregate per (dim, column)
    agg = {}  # (N, key) -> (mean, std)
    for N in dims:
        rs = [r for r in rows if r["N"] == N]
        for key, _, _ in cols:
            vals = [r[key] for r in rs]
            agg[(N, key)] = (float(np.mean(vals)), float(np.std(vals)))

    # Per-column scale (across all N) so the std column shows clean numbers
    scales = {}
    for key, _, _ in cols:
        stds = [agg[(N, key)][1] for N in dims]
        scales[key] = column_scale(stds)

    # ---- Emit LaTeX ----
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{\textbf{Scaling Experiment} (mean $\pm$ std, 5 seeds). "
          r"The RealNVP mixing is consistently nonlinear across dimensions "
          r"($R^2(x \to z) < 1$). The learned model nonetheless recovers the "
          r"true latents at all dimensions. Training losses are stable; "
          r"orthogonality error grows with $N$.}")
    print(r"\label{tab:scaling}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{r cc cc c cc}")
    print(r"\toprule")
    print(r" \multicolumn{1}{c}{\textbf{Latents}} "
          r"& \multicolumn{2}{c}{\textbf{Mixing difficulty}} "
          r"& \multicolumn{2}{c}{\textbf{Linear identifiability}} "
          r"& \multicolumn{1}{c}{\textbf{Orthogonality}} "
          r"& \multicolumn{2}{c}{\textbf{LeJEPA losses}} \\")
    print(r"\cmidrule(lr){1-1} \cmidrule(lr){2-3} \cmidrule(lr){4-5} "
          r"\cmidrule(lr){6-6} \cmidrule(lr){7-8}")
    print("$N$ & " + " & ".join(label for _, _, label in cols) + r" \\")
    print(" & " + " & ".join(scale_header(scales[key]) for key, _, _ in cols)
          + r" \\")
    print(r"\midrule")

    for N in dims:
        cells = [str(N)]
        for key, dec, _ in cols:
            m, s = agg[(N, key)]
            k = scales[key]
            mean_str = f"{m:.{dec}f}"
            std_str = fmt_std(s, k)
            cells.append(rf"{mean_str}\tiny{{$\pm${std_str}}}")
        print("  " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()