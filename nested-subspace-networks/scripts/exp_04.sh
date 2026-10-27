#!/usr/bin/env bash
set -euo pipefail

# Experiment 04: Learned Log Variances During Multi-Rank Training
# Trains a DynamicMLP at rank 64 (teacher_mode=max), tracks per-rank log
# variances across training, and produces a visualization.

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="data/cifar10_emb"
CHECKPOINT_DIR="checkpoints/exp_04"
GPU=0

# Step 1: Ensure CIFAR-10 embeddings exist
if [ ! -f "$DATA_DIR/train.pt" ] || [ ! -f "$DATA_DIR/test.pt" ]; then
    echo "=== Extracting CIFAR-10 ResNet18 embeddings ==="
    uv run python experiments/extract_cifar10_features.py --root "$DATA_DIR"
else
    echo "=== CIFAR-10 embeddings already exist, skipping extraction ==="
fi

# Step 2: Run experiment
echo "=== Running Experiment 04 ==="
uv run python experiments/exp_04.py \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --gpus "$GPU"

echo ""
echo "=== Results ==="
echo "Log variance history: results/exp_04/log_variance_history.csv"
echo "Visualization:        results/exp_04/show_log_variances.pdf"
