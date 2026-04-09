# LeJEPA Identifiability

Code and formal verification for:

**The Gaussian Strikes Back: Identifiability of LeJEPA World Models**
David Klindt (Cold Spring Harbor Laboratory), Yann LeCun (New York University) and Randall Balestriero (Brown University)

We prove that any encoder minimizing pairwise distance between positive pairs while preserving Gaussianity (via SIGReg) must recover the true latent variables up to an orthogonal transformation. The proof uses Hermite polynomials — the natural Fourier basis of the Gaussian measure — to show that any nonlinear distortion strictly increases the alignment loss.

## Quick Start

Try the 2D demo in your browser (runs in ~30s on a T4 GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ozjRk3FfUIDX7WBqlOKvhNcIamy0JxCH?usp=sharing)

## Repository Structure

```
lejepa-identifiability/
├── lean/                           # Lean 4 formal verification
│   ├── LeJEPA/
│   │   ├── Hermite.lean            # Theorem (Hermite polynomial proof)
│   │   ├── Uniqueness.lean         # Uniqueness of Gaussian proof
│   │   ├── Approx.lean             # Approximate identifiability bound
│   │   └── Dirichlet.lean          # Alternative proof (Dirichlet energy)
│   ├── LeJEPA.lean
│   ├── lakefile.lean
│   └── lean-toolchain              # Lean 4 v4.28.0
├── experiments/
│   ├── lejepa_id/                  # Shared library
│   │   ├── mixing.py               # Mixing functions (spiral, banana, sinusoid, coupling)
│   │   ├── models.py               # MLP and matched (inverse-NVP) encoders
│   │   ├── losses.py               # SIGReg, whitening, alignment
│   │   ├── metrics.py              # R², orthogonality, bound quantities
│   │   ├── data.py                 # Gaussian/Laplace sampling, OU augmentation
│   │   └── engine.py               # Training loop (warmup + cosine LR, online data)
│   ├── run.py                      # Unified runner (loads config, trains, saves)
│   ├── analysis/                   # Post-hoc plotting and tables
│   │   ├── plot_2d.py              # 2D scatter figures
│   │   ├── plot_laplace.py         # Laplace ablation figures
│   │   ├── plot_scaling.py         # Scaling curves
│   │   ├── plot_bound.py           # Bound verification + heatmaps
│   │   ├── plot_scatter.py         # Cross-experiment diagnostics
│   │   ├── aggregate.py            # Collect .json results into CSV
│   │   ├── make_table_scaling.py   # LaTeX table from scaling results
│   │   └── run_all.sh              # Regenerate all figures
│   ├── configs/                    # Experiment hyperparameters
│   │   ├── 2d.yaml
│   │   ├── laplace.yaml
│   │   ├── scaling.yaml
│   │   └── grid.yaml
│   └── slurm/                      # SLURM launch scripts (CSHL cluster)
│       ├── launch_2d.sh
│       ├── launch_laplace.sh
│       ├── launch_scaling.sh
│       └── launch_grid.sh
├── requirements.txt
└── README.md
```

## Formal Verification (Lean 4)

All three theoretical results are formalized in Lean 4 with Mathlib. The project compiles with **zero `sorry` obligations** — every logical chain from axiomatized premises to conclusions is machine-checked.

Axiomatized components are standard results not yet available in Mathlib (Hermite polynomial infrastructure, Mazur-Ulam, AM-GM with uniform weights). See the paper appendix for the full verification inventory.

### Building

```bash
cd lean
lake build    # requires Lean 4 v4.28.0, fetches Mathlib automatically
```

## Experiments

All experiments share the same training infrastructure (`lejepa_id/engine.py`) and are launched through a single runner that reads all parameters from YAML configs:

```bash
pip install -r requirements.txt
cd experiments
python run.py --config configs/<experiment>.yaml --run <run_name> --seed <seed>
```

Training uses online data generation (infinite data regime), a warmup + cosine LR schedule, and saves results as `.json` (scalars and training curves). 2D and Laplace experiments additionally save `.pt` files with scatter-plot arrays.

### 2D Illustrations

Four mixing functions (spiral, banana, sinusoidal shear, NVP) with MLP or matched encoders. Multiple seeds, best picked by final loss.

```bash
python run.py --config configs/2d.yaml --run spiral --seed 1337
python analysis/plot_2d.py --results_dir results/2d/ --out figures/
```

### Laplace Ablation

Same mixings with Laplace sources, trained with both LeJEPA and whitening objectives. Shows that linear identifiability fails for non-Gaussian sources.

```bash
python run.py --config configs/laplace.yaml --run spiral_lejepa --seed 1337
python analysis/plot_laplace.py --results_dir results/laplace/ --out figures/
```

### Scaling (N = 2 to 1024)

Matched (inverse-NVP) encoder scaling with latent dimension. Each (N, seed) trains K=3 encoders in parallel (for N ≤ 32) and picks the best.

```bash
python run.py --config configs/scaling.yaml --N 16 --seed 0
python analysis/plot_scaling.py --results_dir results/scaling/ --out figures/
```

### Grid Search / Bound Verification

Sweep over regularization weight and OU correlation on the 2D spiral mixing.

```bash
python run.py --config configs/grid.yaml --lamb 0.01 --rho 0.9 --seed 0
python analysis/plot_bound.py --results_dirs results/grid results/2d results/scaling results/laplace --out figures/
```

### Cross-Experiment Analysis

Pool all results for loss-vs-R², SIGReg-vs-whitening, and bound-vs-error scatter plots:

```bash
python analysis/aggregate.py --results_dir results/ --recursive --out results/all.csv
python analysis/plot_scatter.py --results_dirs results/2d results/laplace results/scaling results/grid --out figures/
```

### Regenerate All Figures

```bash
bash analysis/run_all.sh
```

## Requirements

**Lean**: v4.28.0 + Mathlib v4.28.0 (managed by `lake`)

**Python**: `pip install -r requirements.txt`

## Citation

```bibtex
@article{klindt2025lejepa,
  title={The Gaussian Strikes Back: Identifiability of LeJEPA World Models},
  author={Klindt, David and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT
