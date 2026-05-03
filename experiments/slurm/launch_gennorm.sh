#!/bin/bash
#SBATCH --job-name=lejepa_gennorm
#SBATCH --output=logs/gennorm_%A_%a.out
#SBATCH --error=logs/gennorm_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-71   # 8 runs x 9 alphas; seeds loop inside

RUNS=(spiral_lejepa spiral_whiten banana_lejepa banana_whiten \
      sinusoid_lejepa sinusoid_whiten nvp_lejepa nvp_whiten)
ALPHAS=(0.125 0.25 0.5 1.0 2.0 4.0 8.0 16.0 32.0)
SEEDS=(1337 1338 1339)

N_ALPHAS=${#ALPHAS[@]}
RUN_IDX=$(( SLURM_ARRAY_TASK_ID / N_ALPHAS ))
ALPHA_IDX=$(( SLURM_ARRAY_TASK_ID % N_ALPHAS ))
RUN=${RUNS[$RUN_IDX]}
ALPHA=${ALPHAS[$ALPHA_IDX]}

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs
for SEED in "${SEEDS[@]}"; do
    echo "${RUN} alpha=${ALPHA} seed=${SEED}"
    python run.py --config configs/gennorm.yaml \
        --run "${RUN}" --alpha "${ALPHA}" --seed "${SEED}"
done