# LeJEPA Identifiability
### When Does LeJEPA Learn a World Model?

[David Klindt](https://scholar.google.com/citations?user=EpT-nUAAAAAJ&hl=en), [Yann LeCun](https://scholar.google.com/citations?user=WLN3QrAAAAAJ&hl=en) and [Randall Balestriero](https://scholar.google.com/citations?user=S1x_xqcAAAAJ&hl=en&oi=ao)

**Abstract:** A representation that scrambles the true degrees of freedom of the world cannot support reliable planning or compositional generalization. We prove that LeJEPA (alignment plus Gaussian regularization) linearly recovers the world's latent variables from nonlinear observations, a property known as *linear identifiability*, in a broad class of worlds where latents evolve under stationary, additive-noise transitions. Our main result is that among all such worlds, the Gaussian is the *unique* latent distribution for which this guarantee holds. The forward direction rests on a spectral decomposition in which each degree of nonlinearity is strictly penalized by alignment, making the linear map the optimum; the converse rules out every non-Gaussian alternative. We further prove an *approximate identifiability* result where the guarantee degrades gracefully, and show that linear, orthogonal identifiability enables *optimal latent-space planning*. We validate the theory across 2D examples to 1024-dimensional latents, distributional ablations, and pixel-based robotic control. All theorems are formally verified in Lean 4.

<p align="center">
   <b>[ <a href="https://arxiv.org/abs/TODO">Paper</a> | <a href="https://klindtlab.github.io/lejepa-identifiability/">Website</a> | <a href="https://colab.research.google.com/drive/1ozjRk3FfUIDX7WBqlOKvhNcIamy0JxCH?usp=sharing">Colab</a> | <a href="https://youtu.be/EioGDo67ZDs">Video</a> ]</b>
</p>

<p align="center">
  <a href="https://youtu.be/EioGDo67ZDs">
    <img src="https://img.youtube.com/vi/EioGDo67ZDs/maxresdefault.jpg" width="80%" alt="Video summary">
  </a>
</p>

If you find this work useful, please cite:

```bibtex
@article{klindt2026lejepa,
  title={When Does LeJEPA Learn a World Model?},
  author={Klindt, David and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:TODO},
  year={2026}
}
```

## Quick Start

Try the 2D demo in your browser (~30s on a T4 GPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ozjRk3FfUIDX7WBqlOKvhNcIamy0JxCH?usp=sharing)

## Repository Structure

```
lejepa-identifiability/
├── lean/                           # Lean 4 formal verification
│   ├── LeJEPA/
│   │   ├── Hermite.lean            # Forward direction (Hermite polynomial proof)
│   │   ├── Uniqueness.lean         # Converse (Gaussian uniqueness)
│   │   ├── Approx.lean             # Approximate identifiability bound
│   │   └── Dirichlet.lean          # Alternative proof (Dirichlet energy)
│   ├── LeJEPA.lean
│   ├── lakefile.lean
│   └── lean-toolchain              # Lean 4 v4.28.0
├── experiments/
│   ├── lejepa_id/                  # Shared library
│   │   ├── mixing.py               # Mixing functions (spiral, banana, sinusoid, coupling)
│   │   ├── models.py               # MLP and matched (inverse-NVP) encoders
│   │   ├── losses.py               # SIGReg, whitening, alignment, InfoNCE
│   │   ├── metrics.py              # R², orthogonality, bound quantities
│   │   ├── data.py                 # Gaussian / generalized-normal sampling, OU augmentation
│   │   ├── reacher.py              # Reacher pixel data utilities
│   │   └── engine.py               # Training loop (warmup + cosine LR, online data)
│   ├── run.py                      # Unified runner for 2D / scaling / gennorm / grid
│   ├── run_reacher.py              # Reacher pixel-observation runner
│   ├── prerender.py                # Render Reacher OU and trajectory frames
│   ├── analysis/                   # Post-hoc plotting and tables
│   ├── configs/                    # Experiment hyperparameters (YAML)
│   │   ├── 2d.yaml
│   │   ├── gennorm.yaml
│   │   ├── scaling.yaml
│   │   ├── grid.yaml
│   │   └── reacher.yaml
│   └── slurm/                      # SLURM launch scripts (CSHL cluster)
├── requirements.txt
└── README.md
```

## Formal Verification (Lean 4)

All theoretical results are formalized in Lean 4 with Mathlib. The project compiles with **zero `sorry` obligations** — every logical chain from axiomatized premises to conclusions is machine-checked. Axiomatized components are standard results not yet available in Mathlib (Hermite polynomial infrastructure, Mazur–Ulam, AM–GM with uniform weights). See the paper appendix for the full verification inventory.

```bash
cd lean
lake build    # requires Lean 4 v4.28.0; fetches Mathlib automatically
```

## Experiments

All experiments share the same training infrastructure (`lejepa_id/engine.py`) and read parameters from YAML configs. Training uses online data generation, a warmup + cosine LR schedule, and saves results as `.json` (scalars and curves); 2D and ablation runs additionally save `.pt` files with scatter arrays.

```bash
pip install -r requirements.txt
cd experiments
```

### 2D Illustrations

Four mixing functions (spiral, banana, sinusoidal shear, NVP) with MLP or matched encoders.

```bash
python run.py --config configs/2d.yaml --run spiral_lejepa --seed 1337
python analysis/plot_2d.py --results_dir results/2d/ --out figures/
```

### Scaling (N = 2 to 1024)

Matched (inverse-NVP) encoder scaling with latent dimension, swept across SIGReg / VICReg / InfoNCE objectives. Each (N, seed) trains K=3 encoders in parallel for N ≤ 32 and picks the best by final loss.

```bash
python run.py --config configs/scaling.yaml --N 16 --seed 0
python run.py --config configs/scaling.yaml --N 16 --seed 0 --mode infonce
python analysis/plot_scaling.py --results_dir results/scaling/ --out figures/
```

### Distributional Ablation (Generalized Normal)

Same mixings sweeping the latent shape parameter α (heavy-tailed → Laplace → Gaussian → uniform). Demonstrates that linear identifiability fails away from the Gaussian (α = 2).

```bash
python run.py --config configs/gennorm.yaml --run spiral_lejepa --alpha 2.0 --seed 1337
python analysis/plot_gennorm.py --results_dir results/gennorm/ --out figures/
```

### Grid Search / Bound Verification

Sweep over regularization weight λ and OU correlation ρ on the 2D spiral mixing.

```bash
python run.py --config configs/grid.yaml --lamb 0.01 --rho 0.9 --seed 0
python analysis/plot_bound.py --results_dirs results/grid results/2d results/scaling --out figures/
```

### Reacher (Pixel-Based RL)

CNN encoder on rendered DMC Reacher frames, comparing OU pairs against trajectory pairs from a learned policy.

```bash
python prerender.py ou --rho 0.95
python prerender.py traj --delta 16 --h5_path data/reacher.h5
python run_reacher.py --config configs/reacher.yaml --data_dir data/reacher/ou/rho=0.95
```

### Cross-Experiment Analysis

```bash
python analysis/aggregate.py --results_dir results/ --recursive --out results/all.csv
python analysis/plot_scatter.py --results_dirs results/2d results/gennorm results/scaling results/grid --out figures/
```

### Regenerate All Figures

```bash
bash analysis/run_all.sh
```

## Requirements

- **Lean**: v4.28.0 + Mathlib v4.28.0 (managed by `lake`)
- **Python**: `pip install -r requirements.txt`

## License

MIT
