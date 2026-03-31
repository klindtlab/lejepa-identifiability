# LeJEPA Identifiability

Code and formal verification for:

**The Gaussian Strikes Back: Identifiability of LeJEPA World Models**
David Klindt (Cold Spring Harbor Laboratory) and Randall Balestriero (Brown University)

We prove that any encoder minimizing pairwise distance between positive pairs while preserving Gaussianity (via SIGReg) must recover the true latent variables up to an orthogonal transformation. The proof uses Hermite polynomials — the natural Fourier basis of the Gaussian measure — to show that any nonlinear distortion strictly increases the alignment loss.

## Quick Start

Try the 2D demo in your browser (runs in ~30s on a T4 GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ozjRk3FfUIDX7WBqlOKvhNcIamy0JxCH?usp=sharing)

## Repository Structure

```
lejepa-identifiability/
├── lean/                        # Lean 4 formal verification
│   ├── LeJEPA/
│   │   ├── ThmHermite.lean      # Theorem (Hermite polynomial proof)
│   │   ├── ThmDirichlet.lean    # Alternative proof (Dirichlet energy)
│   │   └── PropApprox.lean      # Approximate identifiability bound
│   ├── LeJEPA.lean
│   ├── lakefile.lean
│   └── lean-toolchain           # Lean 4 v4.28.0
├── experiments/
│   ├── demo_2d.py               # 2D demo (also on Colab above)
│   ├── run_single.py            # Training: single run with metrics + bound
│   ├── aggregate_results.py     # Heatmaps + training curves from grid search
│   ├── plot_bound.py            # Bound verification scatter plots
│   └── launch_grid.sh           # SLURM launcher for λ × ρ grid search
├── requirements.txt
└── README.md
```

## Formal Verification (Lean 4)

All three theoretical results are formalized in Lean 4 with Mathlib. The project compiles with **zero `sorry` obligations** — every logical chain from axiomatized premises to conclusions is machine-checked.

Axiomatized components are standard results not yet available in Mathlib (Hermite polynomial infrastructure, Mazur–Ulam, AM-GM with uniform weights). See the paper appendix for the full verification inventory.

### Building

```bash
cd lean
lake build    # requires Lean 4 v4.28.0, fetches Mathlib automatically
```

## Experiments

### 2D Demo

The demo trains an encoder on a nonlinear spiral mixing and supports two modes: `lejepa` (alignment + SIGReg) and `whiten` (alignment + covariance whitening). Run locally or on [Colab](https://colab.research.google.com/drive/1ozjRk3FfUIDX7WBqlOKvhNcIamy0JxCH?usp=sharing):

```bash
python experiments/demo_2d.py
```

### Grid Search (λ × ρ)

Single run with all metrics (MCC, R², orthogonality, bound quantities):

```bash
python experiments/run_single.py --lamb 0.01 --rho 0.9 --seed 0 --steps 10000
```

Full grid search (6 λ values × 7 ρ values × 5 seeds = 210 runs) on SLURM:

```bash
sbatch experiments/launch_grid.sh
```

After completion, generate figures:

```bash
python experiments/aggregate_results.py --results_dir results/ --out figures/
python experiments/plot_bound.py --results_dir results/ --out figures/
```

## Requirements

**Lean**: v4.28.0 + Mathlib v4.28.0 (managed by `lake`)

**Python**: `pip install -r requirements.txt`

## Citation

```bibtex
@inproceedings{klindt2025lejepa,
  title={The Gaussian Strikes Back: Identifiability of LeJEPA World Models},
  author={Klindt, David and Balestriero, Randall},
  booktitle={NeurIPS},
  year={2025}
}
```

## License

MIT
