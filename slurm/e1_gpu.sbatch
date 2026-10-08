#!/bin/bash
# E1 on GPU: all three CRs sequentially, 1 GPU.  sbatch slurm/e1_gpu.sbatch
#SBATCH --job-name=E1_gpu
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E1gpu_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E1); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e1_gpu.sbatch" "$RD/cmd.sh"; nvidia-smi -L
cd "$REPO/Elastoformer"; set -x
python tools/eval_checkpoints.py --crs 0.4 0.6 0.8 --data_path $IMAGENET --out "$RD" --device cuda --batch_size 256 --workers 8
