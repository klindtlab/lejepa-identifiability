#!/bin/bash
#SBATCH --job-name=lejepa_ou
#SBATCH --output=logs/ou_%A_%a.out
#SBATCH --error=logs/ou_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-6

# Each task: prerender eval (skipped if exists) + 200k images for one rho,
# then train 4 lambdas × 3 seeds × 3 inits = 36 training runs

RHOS=(0.3 0.5 0.7 0.8 0.9 0.95 0.99)
RHO_RAW=${RHOS[$SLURM_ARRAY_TASK_ID]}
RHO=$(printf "%.2f" $RHO_RAW)

eval "$(conda shell.bash hook)"
conda activate pytorch
export MUJOCO_GL=egl
mkdir -p logs

echo "Node: $(hostname) | rho=${RHO} | Start: $(date)"

# Step 1: Prerender (eval + this rho)
python prerender.py eval
python prerender.py ou --rho "${RHO}"

# Step 2: Train
python run_reacher.py --config configs/reacher.yaml \
    --data_dir "data/reacher/ou/rho=${RHO}"

echo "Done: $(date)"
