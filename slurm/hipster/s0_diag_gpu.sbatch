#!/usr/bin/env bash
#SBATCH --job-name=s0_diag_gpu
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=jobs/%x-%j.out
# S0 spectral diagnostics on one GPU (performance: 1 x RTX 6000 Ada, 32 CPUs per GPU as the
# partition enforces; no --exclusive on this shared cluster). No training.
# Estimate for TinyLlama-1.1B at stride 2 (11 blocks x 7 layers): 2-4 h pessimistic, limit = 2x.
#   sbatch slurm/hipster/s0_diag_gpu.sbatch <hf model id> [extra diagnostics.py args]
MODEL=${1:?"usage: sbatch slurm/hipster/s0_diag_gpu.sbatch <hf model id> [args]"}; shift || true
source /home/$USER/elasticrank/repo/slurm/hipster/common.sh gpu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # model and data are prefetched on the login node
RD=$(run_dir "S0_$(basename "$MODEL")"); echo "[s0] run dir: $RD"
python spectral/diagnostics.py --selftest
python spectral/diagnostics.py --model "$MODEL" --out "$RD" --device cuda "$@"
echo "[s0] done"
