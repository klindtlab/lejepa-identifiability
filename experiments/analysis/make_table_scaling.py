"""
Generate LaTeX table from scaling results.

Usage:
    python analysis/make_table_scaling.py --results_dir results/scaling/
"""

import argparse, glob, json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/scaling/")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(f"{args.results_dir}/*.json")):
        with open(path) as f:
            rows.append(json.load(f))

    dims = sorted(set(r["N"] for r in rows))

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\label{tab:scaling}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{r cc cc c cc}")
    print(r" & \multicolumn{2}{c}{Mixing difficulty} & \multicolumn{2}{c}{Linear identifiability} & \multicolumn{1}{c}{Orthogonality} & \multicolumn{2}{c}{LeJEPA losses} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-6} \cmidrule(lr){7-8}")
    print(r"$N$ & $R^2(z \to x)$ & $R^2(x \to z)$ & $R^2(z \to h)$ & $R^2(h \to z)$ & $\|A^\top A - I\|_F / \sqrt{N}$ & Align & SIGReg \\")
    print(r"\midrule")

    for N in dims:
        rs = [r for r in rows if r["N"] == N]

        def fmt(key):
            vals = [r[key] for r in rs]
            m, s = np.mean(vals), np.std(vals)
            if m > 0.99:
                return f"{m:.5f}\\tiny{{$\\pm${s:.0e}}}"
            elif m > 0.1:
                return f"{m:.3f}\\tiny{{$\\pm${s:.0e}}}"
            else:
                return f"{m:.3f}\\tiny{{$\\pm${s:.0e}}}"

        def fmt_align(key):
            vals = [r[key] for r in rs]
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.4f}\\tiny{{$\\pm${s:.0e}}}"

        def fmt_sigreg(key):
            vals = [r[key] for r in rs]
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.3f}\\tiny{{$\\pm${s:.0e}}}"

        print(f"  {N} & {fmt('r2_zx')} & {fmt('r2_xz')} & {fmt('r2_zh')} & {fmt('r2_hz')} & {fmt('orth_err_normalized')} & {fmt_align('final_align')} & {fmt_sigreg('final_sigreg')} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\vspace{5pt}")
    print(r"\caption{")
    print(r"    \textbf{Scaling experiment:}")
    print(r"    metrics across latent dimension $N$ (mean $\pm$ std over 5 seeds).")
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()