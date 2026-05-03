"""
Ablation table: 4 mixings × 3 source distributions × {SIGReg, Whitening}.
Reports R²(h→z). The winner per (mixing, α) is bolded when the difference
is statistically significant (Welch's t-test, p < 0.05).

Usage:
    python analysis/make_table_ablation.py --out figures/tab_ablation.tex
"""
import argparse, glob, json, os
import numpy as np
from scipy import stats
from collections import defaultdict

MIXINGS = [("spiral", "Spiral"), ("banana", "Banana"),
           ("sinusoid", "Sinusoid"), ("nvp", "NVP")]
ALPHAS = [
    (None, r"Gaussian ($\alpha = 2$)"),
    (0.25, r"Heavy tail / sparse ($\alpha = 1/4$)"),
    (16.0, r"Light tail / uniform ($\alpha = 16$)"),
]
# METRIC = "r2_hz" # marginal
METRIC = "r2_hz_grid" # grid
P_THRESH = 0.05


def alpha_key(r):
    if r.get("source_dist") == "gennorm":
        return r.get("source_alpha")
    return None


def fmt(vals, bold=False):
    if not vals:
        return r"$-$"
    s = f"{np.mean(vals):.3f} \\pm {np.std(vals):.3f}"
    return rf"$\mathbf{{{s}}}$" if bold else f"${s}$"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+",
                   default=["results/2d",
                            "results/ablation_alpha_0.25",
                            "results/ablation_alpha_16"])
    p.add_argument("--out", default="figures/tab_ablation.tex")
    args = p.parse_args()

    groups = defaultdict(list)
    for d in args.dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(path) as f:
                r = json.load(f)
            if METRIC not in r:
                continue
            key = (r["mixing"], alpha_key(r), r["mode"])
            groups[key].append(r[METRIC])

    col_spec = "l" + " cc" * len(ALPHAS)
    multicol = " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in ALPHAS)
    cmidrules = "".join(rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(ALPHAS)))
    method_hdr = " & ".join(["SIGReg & Whitening"] * len(ALPHAS))

    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {multicol} \\",
        cmidrules,
        rf"Mixing & {method_hdr} \\",
        r"\midrule",
    ]
    for mix_key, mix_name in MIXINGS:
        row = [mix_name]
        for alpha_k, _ in ALPHAS:
            v_lej = groups.get((mix_key, alpha_k, "lejepa"), [])
            v_wht = groups.get((mix_key, alpha_k, "whiten"), [])
            bold_lej = bold_wht = False
            if len(v_lej) >= 2 and len(v_wht) >= 2:
                _, pval = stats.ttest_ind(v_lej, v_wht, equal_var=False)
                if pval < P_THRESH:
                    if np.mean(v_lej) > np.mean(v_wht):
                        bold_lej = True
                    else:
                        bold_wht = True
            row.append(fmt(v_lej, bold=bold_lej))
            row.append(fmt(v_wht, bold=bold_wht))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    out = "\n".join(lines)
    print(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(out + "\n")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()