#!/bin/bash
# E4a pilot arm. sbatch --export=ALL,ARM=<F|Fc|J|JKD>,CR=<0.5|0.8> slurm/e4a_run.sbatch
#   F   frozen-prefix, paper-faithful        (main.py)
#   Fc  frozen-prefix, corrected init (C12)  (main.py --corrected_init)
#   J   joint nested sampling, CE            (train_nested.py)
#   JKD joint nested sampling, KD from the frozen pretrained model
# Budget: N=3 (4 levels), 10 epochs per stage on a 10% class-balanced train subset = 40 level-epochs for every arm. 1 GPU.
#SBATCH --exclusive
#SBATCH --job-name=E4a
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E4a_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E4a_${ARM}_cr${CR}); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e4a_run.sbatch" "$RD/cmd.sh"; echo "ARM=$ARM CR=$CR" > "$RD/arm.txt"; nvidia-smi -L
COMMON="--model_name facebook/deit-base-patch16-224 --pruning_type l1 --pruning_ratio $CR --pruning_steps 3 --train_subset 0.1 --subset_seed 0 --amp --lr 5e-5"
set -x
case $ARM in
  F|Fc)
    cd $RD
    python $REPO/Elastoformer/main.py --exp_name E4a_${ARM}_cr$CR $COMMON --dataset_name imagenet --data_path $IMAGENET --iterative \
      --train_batch_size 128 --val_batch_size 256 --workers 32 --val_workers 8 --test_accuracy --rebuild --epochs 10 --core_epochs 10 \
      --lr-warmup-epochs 1 --core_weight_decay 0.05 --clip-grad-norm 1 --eval_every 5 --results_dir $RD --save_as $RD/ckpt/ \
      $( [ "$ARM" = Fc ] && echo --corrected_init ) ;;
  J|JKD)
    python $REPO/Elastoformer/train_nested.py --exp_name E4a_${ARM}_cr$CR --out $RD $COMMON --sampler uniform --loss $( [ "$ARM" = JKD ] && echo kd || echo ce ) \
      --total_level_epochs 40 --batch_size 128 --warmup_epochs 1 --workers 32 --val_workers 8 --eval_every_level_epochs 10 ;;
  *) echo "unknown ARM"; exit 2 ;;
esac
