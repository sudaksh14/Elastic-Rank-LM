#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=E3_bench
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E3_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E3); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e3_bench.sbatch" "$RD/cmd.sh"; nvidia-smi -L; nvidia-smi --query-gpu=name,clocks.max.sm,power.limit --format=csv
cd $REPO/Elastoformer; set -x
python tools/bench_kernels.py --out $RD/bench.csv
