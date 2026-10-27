#!/bin/bash

# This script evaluates the remaining baseline and low-rank models.
# It alternates between GPU 2 and 3 for the evaluations.

# --- Configuration ---
CHECKPOINT_DIR="/mnt/pdata/pr501/shortcut-models/checkpoints"
RESULTS_DIR="results"

# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# --- Evaluate Pythia 2.8B Low-Rank ---
echo "🚀 [GPU 2] Evaluating Pythia 2.8B (Epoch Curriculum)..."
CUDA_VISIBLE_DEVICES="2" python experiments/eval_llms.py \
    --model_checkpoint_to_load "${CHECKPOINT_DIR}/pythia-epoch-curriculum/checkpoint-36816/" \
    --base_model_id 'EleutherAI/pythia-2.8b' \
    --results_file "${RESULTS_DIR}/results_pythia_epoch_curriculum.json" \
    --max_rank 1024

# --- Evaluate GPT-Neo 2.7B Low-Rank ---
echo "🚀 [GPU 3] Evaluating GPT-Neo 2.7B (Epoch Curriculum)..."
CUDA_VISIBLE_DEVICES="3" python experiments/eval_llms.py \
    --model_checkpoint_to_load "${CHECKPOINT_DIR}/gpt-neo2-epoch-curriculum/checkpoint-36816/" \
    --base_model_id 'EleutherAI/gpt-neo-2.7B' \
    --results_file "${RESULTS_DIR}/results_gpt-neo2_epoch_curriculum.json" \
    --max_rank 1024

# --- Evaluate Qwen2 0.5B Low-Rank ---
echo "🚀 [GPU 2] Evaluating Qwen2 0.5B (Epoch Curriculum)..."
CUDA_VISIBLE_DEVICES="2" python experiments/eval_llms.py \
    --model_checkpoint_to_load "${CHECKPOINT_DIR}/qwen2-lowrank-epoch-curriculum/checkpoint-36816/" \
    --base_model_id 'Qwen/Qwen2-0.5B' \
    --results_file "${RESULTS_DIR}/results_qwen_epoch_curriculum.json" \
    --max_rank 1024

echo "✅ All evaluations complete."