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
#SBATCH --array=0-11   # 4 runs x 3 seeds

RUNS=(spiral banana sinusoid nvp)
SEEDS=(1337 1338 1339)

N_SEEDS=${#SEEDS[@]}
RUN_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

eval "$(conda shell.bash hook)"
conda activate pytorch

mkdir -p logs
python run.py --config configs/2d.yaml \
    --run "${RUNS[$RUN_IDX]}" --seed "${SEEDS[$SEED_IDX]}"
