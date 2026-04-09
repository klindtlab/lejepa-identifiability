#!/bin/bash

python analysis/aggregate.py --results_dir results/2d/ --out results/2d/summary.csv
python analysis/aggregate.py --results_dir results/ --recursive --out results/all.csv
python analysis/plot_2d.py --results_dir results/2d/ --out figures/
python analysis/plot_bound.py --results_dirs results/grid results/2d results/scaling results/laplace --out figures/
python analysis/plot_laplace.py --results_dir results/laplace/ --out figures/
python analysis/plot_scaling.py --results_dir results/scaling/ --out figures/
python analysis/plot_scatter.py --results_dirs results/2d results/scaling results/grid results/laplace --out figures/
python analysis/make_table_scaling.py --results_dir results/scaling/
