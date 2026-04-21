#!/bin/bash
#SBATCH --job-name=lejepa_traj
#SBATCH --output=logs/traj_%A_%a.out
#SBATCH --error=logs/traj_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-6

# Each task: prerender eval (skipped if exists) + 200k images for one delta,
# then train 4 lambdas × 3 seeds × 3 inits = 36 training runs

DELTAS=(1 2 4 8 16 32 64)
DELTA=${DELTAS[$SLURM_ARRAY_TASK_ID]}

H5_PATH="data/reacher.h5"

eval "$(conda shell.bash hook)"
conda activate pytorch
export MUJOCO_GL=egl
mkdir -p logs

echo "Node: $(hostname) | delta=${DELTA} | Start: $(date)"

# Step 1: Prerender (eval + this delta)
python prerender.py eval
python prerender.py traj --delta "${DELTA}" --h5_path "${H5_PATH}"

# Step 2: Train
python run_reacher.py --config configs/reacher.yaml \
    --data_dir "data/reacher/traj/delta=${DELTA}"

echo "Done: $(date)"
