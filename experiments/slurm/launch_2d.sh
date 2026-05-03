#!/bin/bash
#SBATCH --job-name=lejepa_2d
#SBATCH --output=logs/2d_%A_%a.out
#SBATCH --error=logs/2d_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-7   # 8 runs; seeds loop inside

RUNS=(spiral_lejepa spiral_whiten banana_lejepa banana_whiten \
      sinusoid_lejepa sinusoid_whiten nvp_lejepa nvp_whiten)
SEEDS=(1337 1338 1339)

RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs
for SEED in "${SEEDS[@]}"; do
    echo "${RUN} seed=${SEED}"
    python run.py --config configs/2d.yaml \
        --run "${RUN}" --seed "${SEED}"
done