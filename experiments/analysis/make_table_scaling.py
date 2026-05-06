"""
Generate LaTeX tables from scaling results.

Emits four tables:
  1. tab:scaling-comparison  (main text)
       Three-way R^2(h -> z) comparison: SIGReg, VICReg, InfoNCE.

  2. tab:scaling-sigreg      (appendix)
       Detailed per-method table for SIGReg: mixing difficulty,
       linear identifiability (both directions), orthogonality error,
       alignment loss, SIGReg loss.

  3. tab:scaling-vicreg      (appendix)
       Same structure as SIGReg, but with whitening loss column.

  4. tab:scaling-infonce     (appendix)
       Same structure as SIGReg, but with InfoNCE loss column.

The three appendix tables let each method tell its own failure-mode story:
  - SIGReg / VICReg: orthogonality error grows gradually with N
  - InfoNCE: regularizer loss explodes / fails to converge at high N

Usage:
    python analysis/make_table_scaling.py --results_dir results/scaling/
"""

import argparse, glob, json, os
import numpy as np
from collections import defaultdict
import math


# ──────────────────────────────────────────────────────────────────────────
# Number formatting helpers (shared)
# ──────────────────────────────────────────────────────────────────────────
def column_scale(stds, threshold=0.05):
    valid = [s for s in stds if s > 0 and not np.isnan(s)]
    if not valid:
        return 0
    m = max(valid)
    if m >= threshold:
        return 0
    return int(np.floor(np.log10(m))) - 1


def fmt_std(s, k):
    if np.isnan(s):
        return "---"
    scaled = s if k == 0 else s / (10 ** k)
    if scaled >= 10:
        return f"{math.floor(scaled):.0f}"
    elif scaled >= 1:
        return f"{math.floor(scaled * 10) / 10:.1f}"
    else:
        return f"{math.floor(scaled * 100) / 100:.2f}"


def scale_header(k):
    if k == 0:
        return r"{\scriptsize $\pm$std}"
    return rf"{{\scriptsize $\pm$std\,$\times 10^{{{k}}}$}}"


def fmt_cell(m, s, dec, k):
    if np.isnan(m):
        return "---"
    factor = 10 ** dec
    m_floored = math.floor(m * factor) / factor
    return rf"{m_floored:.{dec}f}\tiny{{$\pm${fmt_std(s, k)}}}"


# ──────────────────────────────────────────────────────────────────────────
# Main-text: three-way comparison on R^2(h -> z)
# ──────────────────────────────────────────────────────────────────────────
def render_table_comparison(by_key, dims):
    modes = ("lejepa", "whiten", "infonce")

    def agg(N, mode, key):
        rs = by_key.get((N, mode), [])
        vals = [r[key] for r in rs if r.get(key) is not None]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    def agg_mixing(N, key):
        all_rs = sum((by_key.get((N, m), []) for m in modes), [])
        vals = [r[key] for r in all_rs if r.get(key) is not None]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    mix_scale  = column_scale([agg_mixing(N, "r2_xz")[1] for N in dims])
    r2_scales  = {m: column_scale([agg(N, m, "r2_hz")[1] for N in dims]) for m in modes}

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{\textbf{Scaling Comparison Across Regularizers} (mean $\pm$ std, 5 seeds). "
          r"All three Gaussianity-enforcing objectives are tested on the same RealNVP mixing "
          r"with matched encoder. SIGReg and VICReg (batch-statistic estimators) maintain "
          r"$R^2 > 0.999$ up to $N{=}1024$, consistent with Thm.~\ref{thm:approx}. "
          r"InfoNCE (pair-based) matches at low $N$ but degrades at scale under fixed kernel "
          r"width $\sigma{=}1$, illustrating the per-dimension tuning required by pair-based estimators. "
          r"Per-method details (orthogonality, regularizer loss) in App.~\ref{app:scaling}, "
          r"Tabs.~\ref{tab:scaling-sigreg}--\ref{tab:scaling-infonce}.}")
    print(r"\label{tab:scaling-comparison}")
    print(r"\begin{tabular}{r c ccc}")
    print(r"\toprule")
    print(r"  & \textbf{Mixing} & \multicolumn{3}{c}{\textbf{Linear identifiability} $R^2(h \to z)$} \\")
    print(r"\cmidrule(lr){3-5}")
    print(r"$N$ & $R^2(x \to z)$ & SIGReg & VICReg & InfoNCE \\")
    sub_cells = [
        scale_header(mix_scale),
        scale_header(r2_scales["lejepa"]),
        scale_header(r2_scales["whiten"]),
        scale_header(r2_scales["infonce"]),
    ]
    print(" & " + " & ".join(sub_cells) + r" \\")
    print(r"\midrule")

    for N in dims:
        cells = [rf"{N}"]
        m, s = agg_mixing(N, "r2_xz")
        cells.append(fmt_cell(m, s, 3, mix_scale))
        for mode in modes:
            m_, s_ = agg(N, mode, "r2_hz")
            cells.append(fmt_cell(m_, s_, 6, r2_scales[mode]))
        print("  " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ──────────────────────────────────────────────────────────────────────────
# Appendix: detailed per-method table
# ──────────────────────────────────────────────────────────────────────────

# Each method has its own native regularizer loss key.
METHOD_SPECS = {
    "lejepa": {
        "name":            "SIGReg",
        "label":           "tab:scaling-sigreg",
        "reg_loss_key":    "final_sigreg",
        "reg_loss_label":  "SIGReg",
        "reg_loss_dec":    2,
        "caption_tail":    (
            "The RealNVP mixing is consistently nonlinear across dimensions "
            r"($R^2(x \to z) < 1$). The learned model nonetheless recovers the "
            "true latents at all dimensions. Training losses are stable; "
            r"orthogonality error grows gradually with $N$."
        ),
    },
    "whiten": {
        "name":            "VICReg",
        "label":           "tab:scaling-vicreg",
        "reg_loss_key":    "final_whiten",
        "reg_loss_label":  "Whitening",
        "reg_loss_dec":    4,
        "caption_tail":    (
            "VICReg matches SIGReg on linear identifiability across all dimensions; "
            r"orthogonality error grows similarly with $N$. The whitening loss is "
            "stable and small throughout."
        ),
    },
    "infonce": {
        "name":            "InfoNCE",
        "label":           "tab:scaling-infonce",
        "reg_loss_key":    "final_loss",
        "reg_loss_label":  "InfoNCE",
        "reg_loss_dec":    3,
        "caption_tail":    (
            r"InfoNCE matches the batch-statistic methods at low $N$ but degrades "
            "at scale under a fixed Gaussian kernel width. The InfoNCE column "
            "shows the total contrastive loss (not decomposable into alignment "
            "plus regularizer), which inflates with $N$ as the per-dimension "
            "kernel-width assumption breaks down."
        ),
    },
}


def render_table_per_method(by_key, dims, mode):
    spec = METHOD_SPECS[mode]
    rows = sum((by_key.get((N, mode), []) for N in dims), [])
    if not rows:
        print(f"% No rows for mode={mode}, skipping {spec['label']}")
        return

    cols = [
        ("r2_zx",                3, r"$R^2(z \to x)$"),
        ("r2_xz",                3, r"$R^2(x \to z)$"),
        ("r2_zh",                5, r"$R^2(z \to h)$"),
        ("r2_hz",                5, r"$R^2(h \to z)$"),
        ("orth_err_normalized",  3, r"$\|\hat Q^\top \hat Q - I\|_F / \sqrt{N}$"),
        ("final_align",          4, "Align"),
        (spec["reg_loss_key"],   spec["reg_loss_dec"], spec["reg_loss_label"]),
    ]

    agg = {}
    for N in dims:
        rs = [r for r in rows if r["N"] == N]
        for key, _, _ in cols:
            vals = [r[key] for r in rs if r.get(key) is not None]
            agg[(N, key)] = ((float(np.mean(vals)), float(np.std(vals)))
                             if vals else (float("nan"), float("nan")))

    scales = {key: column_scale([agg[(N, key)][1] for N in dims])
              for key, _, _ in cols}

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(rf"\caption{{\textbf{{Scaling Experiment ({spec['name']})}} "
          r"(mean $\pm$ std, 5 seeds). " + spec["caption_tail"] + "}")
    print(rf"\label{{{spec['label']}}}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{r cc cc c cc}")
    print(r"\toprule")
    print(r" \multicolumn{1}{c}{\textbf{Latents}} "
          r"& \multicolumn{2}{c}{\textbf{Mixing difficulty}} "
          r"& \multicolumn{2}{c}{\textbf{Linear identifiability}} "
          r"& \multicolumn{1}{c}{\textbf{Orthogonality}} "
          rf"& \multicolumn{{2}}{{c}}{{\textbf{{{spec['name']} losses}}}} \\")
    print(r"\cmidrule(lr){1-1} \cmidrule(lr){2-3} \cmidrule(lr){4-5} "
          r"\cmidrule(lr){6-6} \cmidrule(lr){7-8}")
    print("$N$ & " + " & ".join(label for _, _, label in cols) + r" \\")
    print(" & " + " & ".join(scale_header(scales[key]) for key, _, _ in cols) + r" \\")
    print(r"\midrule")

    for N in dims:
        log2N = int(np.log2(N))
        cells = [rf"$2^{{{log2N}}}$"]
        for key, dec, _ in cols:
            m, s = agg[(N, key)]
            cells.append(fmt_cell(m, s, dec, scales[key]))
        print("  " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")
    print()


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/scaling/")
    args = p.parse_args()

    by_key = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path) as f:
            r = json.load(f)
        mode = r.get("mode", "lejepa")
        by_key[(r["N"], mode)].append(r)

    dims = sorted({N for (N, _) in by_key})
    modes_present = sorted({mode for (_, mode) in by_key})
    print(f"% Found modes: {modes_present}, dims: {dims}\n")

    print(r"% ── Main text: three-way comparison ──")
    render_table_comparison(by_key, dims)

    print(r"% ── Appendix: detailed per-method tables ──")
    for mode in ("lejepa", "whiten", "infonce"):
        render_table_per_method(by_key, dims, mode)


if __name__ == "__main__":
    main()