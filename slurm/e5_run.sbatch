#!/bin/bash
# E5 (training-free): FlexRank DP vs uniform internal widths on the pruning axis, same ranking, matched params. 1 GPU. Smoke first, aborts on failure.
#SBATCH --job-name=E5_dp_prune
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E5_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E5); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e5_run.sbatch" "$RD/cmd.sh"; nvidia-smi -L; mkdir -p $EROOT/data_lists
cd $REPO/Elastoformer; set -x
python eval_prune_dp.py --out $RD/smoke --crs 0.8 --N 2 --calib_n 256 --probe_n 128 --val_n 1000 --dp_cuts 4 --dp_min_p 0.1 || { echo "SMOKE FAILED"; exit 1; }
python eval_prune_dp.py --out $RD --crs 0.8 0.6 --N 5 --calib_n 4096 --probe_n 1024 --val_n 10000 --dp_cuts 12 --dp_min_p 0.05
