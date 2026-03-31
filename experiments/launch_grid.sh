#!/bin/bash
#SBATCH --job-name=lejepa_lsweep
#SBATCH --output=/grid/klindt/home/klindt/LeJEPA/logs/lsweep_%A_%a.out
#SBATCH --error=/grid/klindt/home/klindt/LeJEPA/logs/lsweep_%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4     # 5 lambda values, seeds loop inside

eval "$(conda shell.bash hook)"
conda activate pytorch

WORK_DIR="/grid/klindt/home/klindt/LeJEPA"
SCRIPT="${WORK_DIR}/run_scaling.py"
RESULTS="${WORK_DIR}/results_lambda_sweep"
mkdir -p "${WORK_DIR}/logs" "${RESULTS}"

echo "============================================"
echo "LeJEPA Lambda Sweep (N=16)"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "============================================"

LAMBS=(0.5 1.0 5.0 10.0 50.0)
SEEDS=(0 1 2)
N=16
RHO=0.95

TASK_ID=${SLURM_ARRAY_TASK_ID}
LAMB=${LAMBS[$TASK_ID]}

echo "N=${N}  LAMB=${LAMB}  RHO=${RHO}"
echo "--------------------------------------------"

for SEED in "${SEEDS[@]}"; do
    echo "[$(date)] N=${N} lamb=${LAMB} seed=${SEED}"
    python "${SCRIPT}" \
        --N "${N}" --seed "${SEED}" \
        --lamb "${LAMB}" --rho "${RHO}" \
        --steps 20000 --batch_size 512 \
        --hidden_mult 20 --mixing_h_mult 4 \
        --log_every 100 --out "${RESULTS}"
done

echo "Done! Task ${TASK_ID} (lamb=${LAMB}) finished at $(date)"