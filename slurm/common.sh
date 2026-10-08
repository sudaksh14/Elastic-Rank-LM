# Sourced by every job. Keeps ALL caches/outputs on /var/scratch (home quota is limited).
export EROOT=/var/scratch/skalra/elastoSLM
export REPO=$EROOT/repo
export CKPT=$EROOT/checkpoints
export IMAGENET=/var/scratch/dchabal/quokka/data/imagenet
export HF_HOME=$EROOT/cache/hf
export TORCH_HOME=$EROOT/cache/torch
export XDG_CACHE_HOME=$EROOT/cache/xdg
export PIP_CACHE_DIR=$EROOT/cache/pip
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=$EROOT/cache/wandb
export TMPDIR=${TMPDIR:-/var/scratch/skalra/tmp}
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"
export PYENV=/var/scratch/skalra/prune_llm      # py3.10, torch 2.7.0+cu126, tp 1.6.1, transformers 4.28.0
source /var/scratch/skalra/minimonda3/etc/profile.d/conda.sh
conda activate "$PYENV"
export PYTHONPATH=$REPO/Elastoformer:$REPO/FlexRank/flexrank/src:$REPO/slurm/pyshim:${PYTHONPATH:-}:$EROOT/pylibs   # pylibs = extra pkgs (pytest...) kept OUT of the shared env

# run_dir <EXP-ID>  -> creates $EROOT/runs/<EXP>_<date>_<sha>[_r<n>] and records provenance
run_dir () {
  local sha; sha=$(cat "$REPO/GIT_SHA" 2>/dev/null || echo nogit)
  local d=$EROOT/runs/${1}_$(date +%Y%m%d)_${sha}
  [ -e "$d" ] && d=${d}_$(date +%H%M%S)
  mkdir -p "$d"/{logs,ckpt,plots}
  echo "$sha" > "$d/git_sha.txt"; pip freeze > "$d/pip_freeze.txt" 2>/dev/null
  scontrol show job "${SLURM_JOB_ID:-0}" > "$d/slurm_job.txt" 2>/dev/null || true
  nvidia-smi -L > "$d/gpus.txt" 2>/dev/null || true
  echo "$d"
}
