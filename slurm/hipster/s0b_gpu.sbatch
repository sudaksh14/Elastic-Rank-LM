#!/usr/bin/env bash
#SBATCH --job-name=s0b_shared
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=jobs/%x-%j.out
# S0b spectral diagnostics (shared residual rotation, held-out error), GPT-2 then TinyLlama-1.1B, one job.
# capacity: 1 x L4 24 GB fits (model is moved to CPU after the Gram collection, Grams stay on CPU).
# CUDA: torch 2.8.0+cu128 -> module cuda/12.9.1 (closest available), see common.sh. No --mem, no --exclusive.
# Estimate: S0 took 8 min per model; S0b does about 5x the linear algebra (fp64 solves/eigh on the L4's slow fp64):
# 30-50 min pessimistic per model, so 2 h total, limit = 2x = 4 h.
#   sbatch slurm/hipster/s0b_gpu.sbatch        (run from the repo root /home/skalra/elasticrank/repo)
source /home/$USER/elasticrank/repo/slurm/hipster/common.sh gpu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # models and wikitext are prefetched on the login node
python spectral/diagnostics_shared.py --selftest
for MODEL in gpt2 TinyLlama/TinyLlama-1.1B-Chat-v1.0; do
  RD=$(run_dir "S0b_$(basename "$MODEL")"); echo "[s0b] run dir: $RD"
  STRIDE=1; [[ "$MODEL" == TinyLlama* ]] && STRIDE=2
  python spectral/diagnostics_shared.py --model "$MODEL" --out "$RD" --device cuda --layer-stride $STRIDE
done
echo "[s0b] done"
