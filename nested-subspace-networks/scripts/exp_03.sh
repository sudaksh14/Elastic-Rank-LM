#!/usr/bin/env bash
set -euo pipefail

# Experiment 03: Ablation Study of Training Objectives
# Trains 6 loss variants, evaluates at ID + OOD ranks, produces ablation table.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/cifar10_emb"
CHECKPOINT_DIR="checkpoints/exp_03"
GPU=0

# Step 1: Ensure CIFAR-10 embeddings exist
if [ ! -f "$DATA_DIR/train.pt" ] || [ ! -f "$DATA_DIR/test.pt" ]; then
    echo "=== Extracting CIFAR-10 ResNet18 embeddings ==="
    uv run python experiments/extract_cifar10_features.py --root "$DATA_DIR"
else
    echo "=== CIFAR-10 embeddings already exist, skipping extraction ==="
fi

# Step 2: Run experiment
echo "=== Running Experiment 03 ==="
uv run python experiments/exp_03.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --gpus "$GPU" \
    --num_bootstraps 5

echo ""
echo "=== Summary Table ==="
cat results/exp_03/ablation_summary.csv
