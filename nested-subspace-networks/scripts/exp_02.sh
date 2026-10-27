#!/usr/bin/env bash
set -euo pipefail

# Experiment 02: Native-Rank Training vs Nested (epoch_curriculum) Training
# Reuses native-rank checkpoints from exp_01 when available.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/cifar10_emb"
CHECKPOINT_DIR="checkpoints/exp_02"
NATIVE_CHECKPOINT_DIR="checkpoints/exp_01"
GPU=0

# Step 1: Ensure CIFAR-10 embeddings exist
if [ ! -f "$DATA_DIR/train.pt" ] || [ ! -f "$DATA_DIR/test.pt" ]; then
    echo "=== Extracting CIFAR-10 ResNet18 embeddings ==="
    uv run python experiments/extract_cifar10_features.py --root "$DATA_DIR"
else
    echo "=== CIFAR-10 embeddings already exist, skipping extraction ==="
fi

# Step 2: Run experiment (trains native ranks if exp_01 checkpoints missing, always trains nested model)
echo "=== Running Experiment 02 ==="
uv run python experiments/exp_02.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --native_checkpoint_dir "$NATIVE_CHECKPOINT_DIR" \
    --gpus "$GPU"

echo ""
echo "=== Results ===
echo "Native rank results:"
cat results/exp_02/native_rank_results.csv
echo ""
echo "Nested (epoch_curriculum) rank results:"
cat results/exp_02/nested_rank_results.cs