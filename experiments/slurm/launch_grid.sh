#!/bin/bash
#SBATCH --job-name=lejepa_grid
#SBATCH --output=logs/grid_%A_%a.out
#SBATCH --error=logs/grid_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-8   # 9 lambda values; rhos and seeds loop inside

LAMBS=(1e-6 1e-5 1e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1)
RHOS=(0.3 0.5 0.7 0.8 0.9 0.95 0.99)
SEEDS=(0 1 2)

LAMB=${LAMBS[$SLURM_ARRAY_TASK_ID]}

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs
for RHO in "${RHOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "lamb=${LAMB} rho=${RHO} seed=${SEED}"
        python run.py --config configs/grid.yaml \
            --lamb "${LAMB}" --rho "${RHO}" --seed "${SEED}"
    done
done
