#!/bin/bash
#SBATCH --job-name=lejepa_scale
#SBATCH --output=logs/scale_%A_%a.out
#SBATCH --error=logs/scale_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-9   # 10 dims

DIMS=(2 4 8 16 32 64 128 256 512 1024)
SEEDS=(0 1 2 3 4)

N=${DIMS[$SLURM_ARRAY_TASK_ID]}

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs

for SEED in "${SEEDS[@]}"; do
    echo "N=${N} seed=${SEED}"
    python -u run.py --config configs/scaling.yaml \
        --N "${N}" --seed "${SEED}"
done


