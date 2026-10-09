#!/bin/bash
# E2: training-free fronts (prune vs rank), DP, r_max caps, additive-error check. 1 GPU. A short real-data smoke runs first and aborts the job on failure.
#SBATCH --exclusive
#SBATCH --job-name=E2_fronts
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E2_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E2); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e2_run.sbatch" "$RD/cmd.sh"; nvidia-smi -L
mkdir -p $EROOT/data_lists; cd $REPO/Elastoformer
set -x
python eval_fronts.py --out $RD/smoke --crs 0.8 --N 2 --calib_n 256 --probe_n 128 --val_n 1000 --decomps datasvd --caps 1.0 0.5 --profiles uniform dp \
   --dp_cuts 4 --dp_min_p 0.05 --n_random 6 --workers 10 || { echo "SMOKE FAILED"; exit 1; }
python eval_fronts.py --out $RD --crs 0.4 0.6 0.8 --N 5 --calib_n 4096 --probe_n 1024 --val_n 10000 --prune_full_val --decomps svd datasvd --caps 1.0 0.5 0.3 \
   --profiles uniform dp --dp_cuts 12 --dp_min_p 0.02 --n_random 30 --workers 10
