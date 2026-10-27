#!/usr/bin/env bash
set -euo pipefail

# Experiment 05: End-to-End LLM Low-Rank Performance Pipeline
#
# Trains baseline + low-rank (epoch_curriculum) models for 4 LLMs on MNLI,
# evaluates at all ranks, assembles results, and generates the performance figure.
#
# Usage:
#   bash scripts/exp_05.sh                  # Full pipeline (train + eval + plot)
#   bash scripts/exp_05.sh --smoke_test     # Tiny data subset for validation
#   bash scripts/exp_05.sh --use_existing   # Use pre-trained checkpoints (eval + plot only)

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ============================== Configuration ================================

EXISTING_CHECKPOINT_DIR="/mnt/pdata/pr501/shortcut-models/checkpoints"
DATA_DIR="/mnt/pdata/pr501/shortcut-models/dataset"
RESULTS_DIR="results/exp_05"
CHECKPOINT_DIR="/mnt/pdata/pr501/shortcut-models/checkpoints/exp_05"
GPU_0=0
GPU_1=1

MAX_RANK=1024
MAX_EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
SAMPLING_STRATEGY="epoch_curriculum"

MODELS=("EleutherAI/pythia-2.8b" "EleutherAI/gpt-neo-2.7B" "Qwen/Qwen2-0.5B" "google/gemma-2b")
RUN_NAMES=("pythia" "gpt-neo2" "qwen" "gemma")

# Existing checkpoint paths (for --use_existing mode)
EXISTING_BASE_CKPTS=(
    "${EXISTING_CHECKPOINT_DIR}/pythia-base/final_model/"
    "${EXISTING_CHECKPOINT_DIR}/gpt-neo2-base/checkpoint-30000/"
    "${EXISTING_CHECKPOINT_DIR}/qwen2-base/final_model/"
    "${EXISTING_CHECKPOINT_DIR}/gemma-2b-full-0802/final_model/"
)
EXISTING_LR_CKPTS=(
    "${EXISTING_CHECKPOINT_DIR}/pythia-epoch-curriculum/checkpoint-36816/"
    "${EXISTING_CHECKPOINT_DIR}/gpt-neo2-epoch-curriculum/checkpoint-36816/"
    "${EXISTING_CHECKPOINT_DIR}/qwen2-lowrank-epoch-curriculum/checkpoint-36816/"
    "${EXISTING_CHECKPOINT_DIR}/gemma-2b-lowrank-0802/final_model/"
)

# ============================== Parse flags ==================================

SMOKE_TEST=false
USE_EXISTING=false

for arg in "$@"; do
    case "$arg" in
        --smoke_test)  SMOKE_TEST=true ;;
        --use_existing) USE_EXISTING=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

TRAIN_EXTRA_ARGS=""
EVAL_EXTRA_ARGS=""

export WANDB_DIR="/mnt/pdata/pr501/shortcut-models/wandb_logs"
mkdir -p "$WANDB_DIR"

if $SMOKE_TEST; then
    echo "=== SMOKE TEST MODE: single small model, tiny data ==="
    MODELS=("Qwen/Qwen2-0.5B")
    RUN_NAMES=("qwen")
    MAX_EPOCHS=1
    MAX_RANK=64
    TRAIN_EXTRA_ARGS="--subset_proportion 0.005 --eval_subset_proportion 0.01"
    EVAL_EXTRA_ARGS="--eval_subset_proportion 0.01"
    export WANDB_MODE=disabled
fi

mkdir -p "$RESULTS_DIR"

# ============================== Helper =======================================

pick_gpu() {
    # Alternate GPUs across model index
    if (( $1 % 2 == 0 )); then echo "$GPU_0"; else echo "$GPU_1"; fi
}

# ============================== Stage 1: Train baselines =====================

train_baselines() {
    echo ""
    echo "================================================================"
    echo "  STAGE 1: Training baseline (full-rank) models"
    echo "================================================================"
    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local name="${RUN_NAMES[$i]}"
        local gpu
        gpu=$(pick_gpu "$i")
        local run_name="${name}-base"
        local ckpt_dir="${CHECKPOINT_DIR}/${run_name}"

        if [ -d "${ckpt_dir}/final_model" ]; then
            echo "  [SKIP] ${run_name}: final_model already exists"
            continue
        fi

        echo ""
        echo "--- Training ${run_name} on GPU ${gpu} ---"
        CUDA_VISIBLE_DEVICES="$gpu" uv run python experiments/train_llm_lowrank.py \
            --base_model_id "$model" \
            --model_name "$run_name" \
            --checkpoint_path "$CHECKPOINT_DIR" \
            --data_dir "$DATA_DIR" \
            --max_epochs "$MAX_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LEARNING_RATE" \
            --max_rank "$MAX_RANK" \
            --avoid_lowrank \
            $TRAIN_EXTRA_ARGS
    done
}

# ============================== Stage 2: Train low-rank =====================

train_lowrank() {
    echo ""
    echo "================================================================"
    echo "  STAGE 2: Training low-rank (epoch_curriculum) models"
    echo "================================================================"
    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local name="${RUN_NAMES[$i]}"
        local gpu
        gpu=$(pick_gpu "$i")
        local run_name="${name}-epoch-curriculum"
        local ckpt_dir="${CHECKPOINT_DIR}/${run_name}"

        if [ -d "${ckpt_dir}/final_model" ]; then
            echo "  [SKIP] ${run_name}: final_model already exists"
            continue
        fi

        echo ""
        echo "--- Training ${run_name} on GPU ${gpu} ---"
        CUDA_VISIBLE_DEVICES="$gpu" uv run python experiments/train_llm_lowrank.py \
            --base_model_id "$model" \
            --model_name "$run_name" \
            --checkpoint_path "$CHECKPOINT_DIR" \
            --data_dir "$DATA_DIR" \
            --max_epochs "$MAX_EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --learning_rate "$LEARNING_RATE" \
            --max_rank "$MAX_RANK" \
            --sampling_strategy "$SAMPLING_STRATEGY" \
            $TRAIN_EXTRA_ARGS
    done
}

# ============================== Stage 3: Evaluate baselines ==================

eval_baselines() {
    echo ""
    echo "================================================================"
    echo "  STAGE 3: Evaluating baseline models"
    echo "================================================================"
    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local name="${RUN_NAMES[$i]}"
        local gpu
        gpu=$(pick_gpu "$i")
        local results_file="${RESULTS_DIR}/results_${name}_base.json"

        if [ -f "$results_file" ]; then
            echo "  [SKIP] ${name} base: results file already exists"
            continue
        fi

        local base_ckpt
        if $USE_EXISTING; then
            base_ckpt="${EXISTING_BASE_CKPTS[$i]}"
        else
            base_ckpt="${CHECKPOINT_DIR}/${name}-base/final_model/"
        fi

        echo ""
        echo "--- Evaluating ${name} base on GPU ${gpu} ---"
        echo "    checkpoint: ${base_ckpt}"
        CUDA_VISIBLE_DEVICES="$gpu" uv run python experiments/eval_llms.py \
            --model_checkpoint_to_load "$base_ckpt" \
            --base_model_id "$base_ckpt" \
            --avoid_lowrank \
            --results_file "$results_file" \
            --data_dir "$DATA_DIR" \
            $EVAL_EXTRA_ARGS
    done
}

# ============================== Stage 4: Evaluate low-rank ==================

eval_lowrank() {
    echo ""
    echo "================================================================"
    echo "  STAGE 4: Evaluating low-rank (epoch_curriculum) models"
    echo "================================================================"
    for i in "${!MODELS[@]}"; do
        local model="${MODELS[$i]}"
        local name="${RUN_NAMES[$i]}"
        local gpu
        gpu=$(pick_gpu "$i")
        local results_file="${RESULTS_DIR}/results_${name}_epoch_curriculum.json"

        if [ -f "$results_file" ]; then
            echo "  [SKIP] ${name} epoch_curriculum: results file already exists"
            continue
        fi

        local lr_ckpt
        if $USE_EXISTING; then
            lr_ckpt="${EXISTING_LR_CKPTS[$i]}"
        else
            lr_ckpt="${CHECKPOINT_DIR}/${name}-epoch-curriculum/final_model/"
        fi

        echo ""
        echo "--- Evaluating ${name} epoch_curriculum on GPU ${gpu} ---"
        echo "    checkpoint: ${lr_ckpt}"
        CUDA_VISIBLE_DEVICES="$gpu" uv run python experiments/eval_llms.py \
            --model_checkpoint_to_load "$lr_ckpt" \
            --base_model_id "$model" \
            --results_file "$results_file" \
            --max_rank "$MAX_RANK" \
            --data_dir "$DATA_DIR" \
            $EVAL_EXTRA_ARGS
    done
}

# ============================== Stage 5: Assemble + plot ====================

assemble_and_plot() {
    echo ""
    echo "================================================================"
    echo "  STAGE 5: Assembling results and generating figure"
    echo "================================================================"
    uv run python experiments/exp_05.py \
        --results_dir "$RESULTS_DIR" \
        --params_json "data/params_with_percentages.json"
}

# ============================== Main =========================================

echo "============================================================"
echo "  Experiment 05: LLM Low-Rank Performance Pipeline"
echo "============================================================"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Max rank:     ${MAX_RANK}"
echo "  Strategy:     ${SAMPLING_STRATEGY}"
echo "  Smoke test:   ${SMOKE_TEST}"
echo "  Use existing: ${USE_EXISTING}"
echo "============================================================"

if ! $USE_EXISTING; then
    train_baselines
    train_lowrank
fi

eval_baselines
eval_lowrank
assemble_and_plot

echo ""
echo "============================================================"
echo "  Experiment 05 complete!"
echo "============================================================"
echo "  Results:      ${RESULTS_DIR}/llm_performance.csv"
echo "  Figure:       ${RESULTS_DIR}/performance_across_llms_flops.pdf"
echo "  JSON files:   ${RESULTS_DIR}/results_*.json"
echo "============================================================"
