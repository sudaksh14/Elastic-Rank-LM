#!/usr/bin/env bash
set -euo pipefail

# Experiment 01: Native-Rank Training vs Rank Truncation
# Reproduces results in results/exp_01/

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/cifar10_emb"
CHECKPOINT_DIR="checkpoints/exp_01"
GPU=0

# Step 1: Extract CIFAR-10 embeddings (skip if already done)
if [ ! -f "$DATA_DIR/train.pt" ] || [ ! -f "$DATA_DIR/test.pt" ]; then
    echo "=== Extracting CIFAR-10 ResNet18 embeddings ==="
    uv run python experiments/extract_cifar10_features.py --root "$DATA_DIR"
else
    echo "=== CIFAR-10 embeddings already exist, skipping extraction ==="
fi

# Step 2: Run experiment (train + evaluate)
echo "=== Running Experiment 01 ==="
uv run python experiments/exp_01.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --gpus "$GPU"

echo ""
echo "=== Results ==="
echo "Native rank results:"
cat results/exp_01/native_rank_results.csv
echo ""
echo "Truncated rank results:"
cat results/exp_01/truncated_rank_results.csv