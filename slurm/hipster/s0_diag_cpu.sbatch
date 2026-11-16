#!/usr/bin/env bash
#SBATCH --job-name=s0_diag_cpu
#SBATCH --partition=capacity
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=jobs/%x-%j.out
# S0 spectral diagnostics without a GPU (capacity partition, 16 CPUs). For small models only
# (GPT-2: 48 layers up to 768 x 3072). Estimate 0.5-2 h pessimistic, limit = 2x.
#   sbatch slurm/hipster/s0_diag_cpu.sbatch <hf model id> [extra diagnostics.py args]
MODEL=${1:?"usage: sbatch slurm/hipster/s0_diag_cpu.sbatch <hf model id> [args]"}; shift || true
source /home/$USER/elasticrank/repo/slurm/hipster/common.sh cpu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # model and data are prefetched on the login node
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
RD=$(run_dir "S0_$(basename "$MODEL")"); echo "[s0] run dir: $RD"
python spectral/diagnostics.py --selftest
python spectral/diagnostics.py --model "$MODEL" --out "$RD" --device cpu "$@"
echo "[s0] done"
