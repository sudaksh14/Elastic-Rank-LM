#!/bin/bash
# E6: nested LOW-RANK DeiT-B, joint training (FlexRank-style KD), budgets = the pruning ladder's param totals (CR, N=3), compute matched to E4a.
#   sbatch --export=ALL,RCAP=<1.0|0.5|0.3>,CR=<0.8> slurm/e6_run.sbatch
# A short real-data smoke test runs first and aborts the job on failure.
#SBATCH --job-name=E6_rank
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12
#SBATCH --time=14:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E6_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
CR=${CR:-0.8}
RD=$(run_dir E6_rcap${RCAP}_cr${CR}); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e6_run.sbatch" "$RD/cmd.sh"; nvidia-smi -L; mkdir -p $EROOT/data_lists
cd $REPO/Elastoformer
COMMON="--axis rank --model_name facebook/deit-base-patch16-224 --pruning_type l1 --pruning_ratio $CR --pruning_steps 3 --decomp datasvd --r_cap $RCAP --loss kd --sampler uniform --amp --lr 5e-5 --batch_size 128 --micro_batch 64 --workers 10 --val_workers 4"
set -x
python train_nested.py --exp_name E6smoke $COMMON --out $RD/smoke --train_subset 0.01 --total_level_epochs 0.4 --warmup_epochs 0.1 --val_limit 500 \
   --calib_n 256 --probe_n 128 --dp_cuts 6 --dp_min_p 0.05 --print_freq 5 || { echo "SMOKE FAILED"; exit 1; }
python train_nested.py --exp_name E6_rcap${RCAP}_cr$CR $COMMON --out $RD --train_subset 0.1 --subset_seed 0 --total_level_epochs 40 --warmup_epochs 1 \
   --calib_n 4096 --probe_n 1024 --dp_cuts 12 --dp_min_p 0.02 --eval_every_level_epochs 10
