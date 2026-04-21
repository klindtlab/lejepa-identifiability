"""
Generate LaTeX tables for the paper, matching scaling table style.

Usage:
    python analysis/make_table_reacher.py --results_dir results/reacher
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_all_results(results_dir):
    ou, traj = [], []
    for p in Path(results_dir).rglob("result.json"):
        r = json.load(open(p))
        if "delta" in r and r.get("rho") is None:
            traj.append(r)
        elif "rho" in r:
            ou.append(r)
    return ou, traj


def best_lambda_per_x(results, x_key):
    grouped = defaultdict(list)
    for r in results:
        grouped[(r[x_key], r["lamb"])].append(r)

    best = {}
    for x in sorted(set(k[0] for k in grouped)):
        best_mean, best_lamb = -np.inf, None
        for lamb in set(k[1] for k in grouped if k[0] == x):
            m = np.mean([r["r2_hz"] for r in grouped[(x, lamb)]])
            if m > best_mean:
                best_mean, best_lamb = m, lamb
        best[x] = {
            "lamb": best_lamb,
            "runs": grouped[(x, best_lamb)],
        }
    return best


def pm(vals, fmt=".2f"):
    """Format as value\\tiny{±std} matching paper style."""
    m, s = np.mean(vals), np.std(vals)
    return f"{m:{fmt}}\\tiny{{$\\pm${s:.0e}}}"


def make_combined_table(ou_results, traj_results):
    ou_best = best_lambda_per_x(ou_results, "rho")
    traj_best = best_lambda_per_x(traj_results, "delta")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{r cc | r cc ccc}")
    lines.append(r" \multicolumn{3}{c}{\textbf{OU (Gaussian)}} & "
                 r"\multicolumn{6}{c}{\textbf{Trajectory (non-Gaussian)}} \\")
    lines.append(r"\cmidrule(lr){1-3} \cmidrule(lr){4-9}")
    lines.append(r"$\rho$ & $R^2(z \to h)$ & $R^2(h \to z)$ & "
                 r"$\delta$ & $\rho_0$ & $\rho_1$ & "
                 r"$R^2(z \to h)$ & $R^2(h \to z_0)$ & $R^2(h \to z_1)$ \\")
    lines.append(r"\midrule")

    ou_rhos = sorted(ou_best.keys())
    traj_deltas = sorted(traj_best.keys())
    n_rows = max(len(ou_rhos), len(traj_deltas))

    for i in range(n_rows):
        # OU columns
        if i < len(ou_rhos):
            rho = ou_rhos[i]
            runs = ou_best[rho]["runs"]
            r2_zh = pm([r["r2_zh"] for r in runs])
            r2_hz = pm([r["r2_hz"] for r in runs])
            ou_str = f"  {rho:.2f} & {r2_zh} & {r2_hz}"
        else:
            ou_str = r"  & &"

        # Traj columns
        if i < len(traj_deltas):
            delta = traj_deltas[i]
            runs = traj_best[delta]["runs"]
            rho0 = runs[0].get("rho_shoulder", None)
            rho1 = runs[0].get("rho_wrist", None)
            rho0_s = f"{rho0:.3f}" if rho0 is not None else "---"
            rho1_s = f"{rho1:.3f}" if rho1 is not None else "---"
            r2_zh = pm([r["r2_zh"] for r in runs])
            r2_d0 = pm([r["r2_hz_per_dim"][0] for r in runs])
            r2_d1 = pm([r["r2_hz_per_dim"][1] for r in runs])
            traj_str = f"{delta} & {rho0_s} & {rho1_s} & {r2_zh} & {r2_d0} & {r2_d1}"
        else:
            traj_str = r"& & & & &"

        lines.append(f"{ou_str} & {traj_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\vspace{5pt}")
    lines.append(r"\caption{")
    lines.append(r"    \textbf{Pixel-observation identifiability on DMC Reacher} "
                 r"(mean $\pm$ std over 3 seeds, best $\lambda$ per condition).")
    lines.append(r"    \textbf{Left:} OU process with Gaussian marginals. "
                 r"$R^2$ increases monotonically with $\rho$, reaching $0.95$ "
                 r"at $\rho = 0.99$, confirming linear identifiability from pixels.")
    lines.append(r"    \textbf{Right:} Real SAC trajectories with non-Gaussian marginals. "
                 r"The two joints have different autocorrelation timescales ($\rho_0 \neq \rho_1$) "
                 r"and the wrist has a near-uniform marginal distribution, "
                 r"leading to anisotropic and reduced identifiability.")
    lines.append(r"}")
    lines.append(r"\label{tab:reacher}")
    lines.append(r"\vspace{-20pt}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_ou_table_standalone(ou_results):
    """Standalone OU table for appendix if needed."""
    ou_best = best_lambda_per_x(ou_results, "rho")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{r c cc}")
    lines.append(r"\toprule")
    lines.append(r" \multicolumn{1}{c}{\textbf{Correlation}} & "
                 r"\multicolumn{1}{c}{\textbf{Regularizer}} & "
                 r"\multicolumn{2}{c}{\textbf{Linear identifiability}} \\")
    lines.append(r"\cmidrule(lr){1-1} \cmidrule(lr){2-2} \cmidrule(lr){3-4}")
    lines.append(r"$\rho$ & $\lambda$ & $R^2(z \to h)$ & $R^2(h \to z)$ \\")
    lines.append(r"\midrule")

    for rho in sorted(ou_best.keys()):
        runs = ou_best[rho]["runs"]
        lamb = ou_best[rho]["lamb"]
        r2_zh = pm([r["r2_zh"] for r in runs])
        r2_hz = pm([r["r2_hz"] for r in runs])
        lines.append(f"  {rho:.2f} & {lamb:.0e} & {r2_zh} & {r2_hz} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{5pt}")
    lines.append(r"\caption{")
    lines.append(r"    \textbf{OU (Gaussian) identifiability from pixels} "
                 r"(mean $\pm$ std over 3 seeds).")
    lines.append(r"    $R^2$ increases monotonically with temporal correlation $\rho$, "
                 r"reaching $0.95$ at $\rho = 0.99$.")
    lines.append(r"}")
    lines.append(r"\label{tab:reacher_ou}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_traj_table_standalone(traj_results):
    """Standalone traj table for appendix if needed."""
    traj_best = best_lambda_per_x(traj_results, "delta")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{r cc c c cc}")
    lines.append(r"\toprule")
    lines.append(r" \multicolumn{1}{c}{\textbf{Stride}} & "
                 r"\multicolumn{2}{c}{\textbf{Autocorrelation}} & "
                 r"\multicolumn{1}{c}{\textbf{Regularizer}} & "
                 r"\multicolumn{1}{c}{\textbf{Identifiability}} & "
                 r"\multicolumn{2}{c}{\textbf{Per-dimension}} \\")
    lines.append(r"\cmidrule(lr){1-1} \cmidrule(lr){2-3} \cmidrule(lr){4-4} "
                 r"\cmidrule(lr){5-5} \cmidrule(lr){6-7}")
    lines.append(r"$\delta$ & $\rho_0$ & $\rho_1$ & $\lambda$ & "
                 r"$R^2(z \to h)$ & $R^2(h \to z_0)$ & $R^2(h \to z_1)$ \\")
    lines.append(r"\midrule")

    for delta in sorted(traj_best.keys()):
        runs = traj_best[delta]["runs"]
        lamb = traj_best[delta]["lamb"]
        rho0 = runs[0].get("rho_shoulder", None)
        rho1 = runs[0].get("rho_wrist", None)
        rho0_s = f"{rho0:.3f}" if rho0 is not None else "---"
        rho1_s = f"{rho1:.3f}" if rho1 is not None else "---"
        r2_zh = pm([r["r2_zh"] for r in runs])
        r2_d0 = pm([r["r2_hz_per_dim"][0] for r in runs])
        r2_d1 = pm([r["r2_hz_per_dim"][1] for r in runs])
        lines.append(f"  {delta} & {rho0_s} & {rho1_s} & {lamb:.0e} "
                     f"& {r2_zh} & {r2_d0} & {r2_d1} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\vspace{5pt}")
    lines.append(r"\caption{")
    lines.append(r"    \textbf{Trajectory (non-Gaussian) identifiability from pixels} "
                 r"(mean $\pm$ std over 3 seeds).")
    lines.append(r"    The shoulder ($z_0$) and wrist ($z_1$) have different "
                 r"autocorrelation timescales and marginal distributions, "
                 r"leading to anisotropic identifiability.")
    lines.append(r"}")
    lines.append(r"\label{tab:reacher_traj}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/reacher")
    args = parser.parse_args()

    ou, traj = load_all_results(args.results_dir)
    print(f"Loaded {len(ou)} OU runs, {len(traj)} traj runs\n")

    if ou and traj:
        print("=" * 70)
        print("COMBINED TABLE (for main text)")
        print("=" * 70)
        print(make_combined_table(ou, traj))
        print()

    if ou:
        print("=" * 70)
        print("OU TABLE (standalone, for appendix)")
        print("=" * 70)
        print(make_ou_table_standalone(ou))
        print()

    if traj:
        print("=" * 70)
        print("TRAJ TABLE (standalone, for appendix)")
        print("=" * 70)
        print(make_traj_table_standalone(traj))


if __name__ == "__main__":
    main()