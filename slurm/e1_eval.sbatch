#!/bin/bash
# usage: sbatch -p fatq --export=ALL,CRS="0.8",LIMIT=0,TAG=E1 slurm/e1_eval.sbatch     (GPU: -p defq --gres=gpu:1 --export=...,DEVICE=cuda)
#SBATCH --job-name=E1_eval
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E1_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir ${TAG:-E1}_cr${CRS// /-}); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e1_eval.sbatch" "$RD/cmd.sh"
DEVICE=${DEVICE:-cpu}; [ "$DEVICE" = cpu ] && export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=36
cd "$REPO/Elastoformer"
set -x
python tools/eval_checkpoints.py --crs $CRS --levels ${LEVELS:-1 2 3 4 5 6} --data_path $IMAGENET --out "$RD" \
   --device $DEVICE --batch_size 128 --workers 12 --threads 36 --limit ${LIMIT:-0}
