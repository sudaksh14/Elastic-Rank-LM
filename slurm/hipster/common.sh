#!/usr/bin/env bash
# Sourced by every Elastic-Rank-LM job on hipster (UvA FNWI cluster). Conventions follow the
# FlexSLM / FlexLLaVA hipster launchers:
#  - partitions: performance = RTX 6000 Ada 48 GB, 32 CPUs per GPU; capacity = L4 24 GB, 16 CPUs
#    per GPU; max 7 days; shared multi-tenant cluster, so never --exclusive, request exactly what
#    the job needs; time limit = 2x a pessimistic runtime estimate;
#  - conda env my_env (python 3.9, torch 2.8.0+cu128, transformers 4.57.6), used read-only;
#  - /home is backed up, /scratch is wiped every few weeks and not backed up: repo, caches and run
#    results live in /home/$USER/elasticrank; only checkpoints (reproducible) go to /scratch/$USER/elasticrank.
# Usage at the top of a job script:  source /home/$USER/elasticrank/repo/slurm/hipster/common.sh gpu|cpu
set -eo pipefail
KIND=${1:-gpu}
export EROOT=/home/$USER/elasticrank                  # repo, hf_cache, runs: backed up
export REPO=$EROOT/repo
export CKPT=/scratch/$USER/elasticrank/checkpoints      # checkpoints only: scratch is deleted every few weeks
export HF_HOME=$EROOT/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$EROOT/runs" "$HF_HOME" "$CKPT"
if [[ "$KIND" == gpu ]]; then
  if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi
eval "$(/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate my_env
cd "$REPO"
echo "[common] host=$(hostname) job=${SLURM_JOB_ID:-none} kind=$KIND python=$(which python) sha=$(cat "$REPO/GIT_SHA" 2>/dev/null || echo nogit)"
python -c "
import torch, transformers
print(f'[common] torch {torch.__version__} transformers {transformers.__version__} '
      f'cuda_available={torch.cuda.is_available()} devices={torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'[common]   cuda:{i} {torch.cuda.get_device_name(i)}')
"
if [[ "$KIND" == gpu ]]; then
  python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "[common] ERROR: GPU job but torch sees no GPU" >&2; exit 4; }
fi

# run_dir <EXP-ID>  -> creates $EROOT/runs/<EXP>_<date>_<sha>[_<time>] and records provenance
run_dir () {
  local sha; sha=$(cat "$REPO/GIT_SHA" 2>/dev/null || echo nogit)
  local d=$EROOT/runs/${1}_$(date +%Y%m%d)_${sha}
  [ -e "$d" ] && d=${d}_$(date +%H%M%S)
  mkdir -p "$d"
  echo "$sha" > "$d/git_sha.txt"
  scontrol show job "${SLURM_JOB_ID:-0}" > "$d/slurm_job.txt" 2>/dev/null || true
  echo "$d"
}
