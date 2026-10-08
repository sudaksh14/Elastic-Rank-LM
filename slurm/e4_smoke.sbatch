#!/bin/bash
# GPU smoke test of every code path the E4a jobs use (real DeiT-B weights, tiny data). ~15 min. E4a runs depend on it (afterok).
#SBATCH --job-name=E4_smoke
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12
#SBATCH --time=01:00:00
#SBATCH --output=/var/scratch/skalra/elastoSLM/runs/E4smoke_%j.out
set -uo pipefail
source /var/scratch/skalra/elastoSLM/repo/slurm/common.sh
RD=$(run_dir E4smoke); echo "run dir: $RD"; exec > >(tee -a "$RD/logs/job.log") 2>&1
cp "$REPO/slurm/e4_smoke.sbatch" "$RD/cmd.sh"; nvidia-smi -L
rc=0
set -x
COMMON="--model_name facebook/deit-base-patch16-224"
# F arm (frozen prefix): faithful, then corrected init. debug=4000 train / 1000 val images, 1 epoch per stage, N=3
for v in "" "--corrected_init"; do
  d=$RD/F${v// /}; mkdir -p $d && cd $d
  python $REPO/Elastoformer/main.py --exp_name smokeF $COMMON --dataset_name imagenet --data_path $IMAGENET --pruning_type l1 --pruning_ratio 0.8 \
     --iterative --pruning_steps 3 --train_batch_size 64 --val_batch_size 128 --workers 8 --val_workers 4 --train_subset 0.01 --debug --test_accuracy \
     --rebuild --epochs 1 --core_epochs 1 --lr-warmup-epochs 0 --lr 5e-5 --core_weight_decay 0.05 --amp --clip-grad-norm 1 --results_dir $d --save_as $d/ckpt/ $v || { echo "FAILED F $v"; rc=1; }
done
# J arm (joint): CE+uniform, KD+sandwich
for cfg in "uniform ce" "sandwich kd"; do set -- $cfg
  python $REPO/Elastoformer/train_nested.py --exp_name smokeJ_$1_$2 --out $RD/J_$1_$2 $COMMON --pruning_ratio 0.8 --pruning_steps 3 --sampler $1 --loss $2 \
     --total_level_epochs 2 --train_subset 0.01 --batch_size 64 --amp --warmup_epochs 0.2 --val_limit 1000 --workers 8 --print_freq 10 || { echo "FAILED J $cfg"; rc=1; }
done
echo SMOKE_DONE rc=$rc; exit $rc
