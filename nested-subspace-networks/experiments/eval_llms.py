import os
import argparse
import numpy as np
import torch
import evaluate
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    set_seed,
)
from lowrank.trainer import LowRankTrainer, ComparableBaselineTrainer
from lowrank.models.layers import DynamicLowRankLinear, decompose_and_replace
import math
import torch.nn as nn
from types import MethodType

from safetensors.torch import load_file

def parse_args():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model on MNLI.")
    
    # --- Paths and Configuration ---
    parser.add_argument('--model_checkpoint_to_load', type=str, required=True,
                        help='Path to the saved model checkpoint directory to evaluate.')
    parser.add_argument('--results_file', type=str, default='evaluation_results.json',
                        help='Path to save the evaluation results JSON file.')
    parser.add_argument('--data_dir', type=str, default='/mnt/pdata/pr501/shortcut-models/dataset',
                        help='Base directory for datasets.')
    
    # --- Model & Run Configuration (should match the training run) ---
    parser.add_argument('--base_model_id', type=str, default='google/gemma-2b', help='Base model ID used for tokenizer.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Maximum sequence length.')
    parser.add_argument('--max_rank', type=int, default=1024, help='Maximum rank of the model (for low-rank models).')
    parser.add_argument('--avoid_lowrank', action='store_true', 
                        help='Set this flag if evaluating a baseline (full-rank) model.')

    # --- Evaluation Hyperparameters ---
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--eval_subset_proportion', type=float, default=0.15, help='Proportion of validation set to use.')

    return parser.parse_args()

def main(args):
    """Main evaluation function."""
    set_seed(args.seed)
    model_cache_dir = os.path.join(args.data_dir, 'huggingface_models')
    dataset_cache_dir = os.path.join(args.data_dir, "huggingface_datasets")
    
    # --- Load Tokenizer ---
    print(f"Loading tokenizer for {args.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, cache_dir=model_cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # --- CHANGE 1: Define TrainingArguments early to get the correct device ---
    # This object inspects the environment (including CUDA_VISIBLE_DEVICES)
    # and sets the appropriate device.
    training_args = TrainingArguments(
        output_dir="./eval_tmp", # Temporary directory
        per_device_eval_batch_size=args.eval_batch_size,
        bf16=True,
        dataloader_num_workers=args.num_workers,
        report_to="none",
    )

    # --- CHANGE 2: Simplified and corrected model loading logic ---
    if args.avoid_lowrank:
        # For baseline models, loading with device_map="auto" is simplest and best.
        print(f"Loading FULL-RANK model from: {args.model_checkpoint_to_load}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_id,
            num_labels=3,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype=torch.bfloat16,
            cache_dir=model_cache_dir,
            device_map='auto',
        )
    else:
        # For LOW-RANK models, we perform the required two-step loading process.
        print(f"1. Building base architecture from '{args.base_model_id}'...")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model_id,
            num_labels=3,
            pad_token_id=tokenizer.pad_token_id,
            torch_dtype=torch.bfloat16,
            cache_dir=model_cache_dir,
            device_map='auto',
        )

        print("2. Modifying architecture with custom low-rank layers...")
        if 'Qwen' in args.base_model_id or 'gemma' in args.base_model_id:
            decompose_and_replace(model.model, max_rank=args.max_rank, target_substring='mlp')
        elif 'gpt-neo-2' in args.base_model_id:
            decompose_and_replace(model.transformer, max_rank=args.max_rank, target_substring='mlp')
        elif 'pythia' in args.base_model_id:
            decompose_and_replace(model.gpt_neox, max_rank=args.max_rank, target_substring='mlp')
        else:
            raise ValueError("Model architecture not supported in evaluation script logic")

        # Attach attributes
        model.max_rank = args.max_rank
        model.train_ranks = [
            args.max_rank // (2**k) 
            for k in range(int(math.log2(args.max_rank)) + 1) if args.max_rank // (2**k) > 0
        ]

        model.log_vars_ce = nn.ParameterDict({
            str(r): nn.Parameter(torch.zeros(())) for r in model.train_ranks
        }).to(model.device) 
        
        # 4. Define the _set_rank_all method and attach it to the model instance.
        def _set_rank_all_method(self, rank):
            for m in self.modules():
                if isinstance(m, DynamicLowRankLinear):
                    m.set_rank(rank)
        
        model._set_rank_all = MethodType(_set_rank_all_method, model)
        
        print("Low-rank attributes and methods attached to model instance.")
        weights_path = os.path.join(args.model_checkpoint_to_load, 'model.safetensors')
        print(f"3. Loading fine-tuned weights from '{weights_path}'...")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=False)
        
        # --- CHANGE 3: Use the dynamic device from TrainingArguments ---
        print(f"4. Moving model to target device: {training_args.device}")
        model = model.to(training_args.device)
        print("✅ Low-rank model and trained weights loaded successfully.")
        
    # --- CRITICAL: Re-attach attributes for LowRankTrainer ---
    if not args.avoid_lowrank:
        print("Re-attaching custom attributes for Low-Rank model...")
        model.max_rank = args.max_rank
        model.train_ranks = [
            args.max_rank // (2**k) 
            for k in range(int(math.log2(args.max_rank)) + 1) if args.max_rank // (2**k) > 0
        ]
        
        def _set_rank_all_method(self, rank):
            for m in self.modules():
                if isinstance(m, DynamicLowRankLinear):
                    m.set_rank(rank)
        
        model._set_rank_all = MethodType(_set_rank_all_method, model)
        print("Low-rank attributes and methods re-attached successfully.")

    # --- Load and Preprocess Dataset ---
    dataset = load_dataset("glue", "mnli", cache_dir=dataset_cache_dir)

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"], examples["hypothesis"],
            truncation=True, padding="max_length", max_length=args.max_seq_length,
        )

    tokenized_dataset = dataset.map(
        preprocess_function, batched=True, num_proc=args.num_workers,
        remove_columns=["premise", "hypothesis", "idx"]
    )
    
    num_eval_matched = int(len(tokenized_dataset["validation_matched"]) * args.eval_subset_proportion)
    num_eval_mismatched = int(len(tokenized_dataset["validation_mismatched"]) * args.eval_subset_proportion)

    eval_dataset_matched = tokenized_dataset["validation_matched"].select(range(num_eval_matched))
    eval_dataset_mismatched = tokenized_dataset["validation_mismatched"].select(range(num_eval_mismatched))
    
    # --- Configure Evaluation ---
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # --- Select the appropriate trainer ---
    if not args.avoid_lowrank:
        trainer = LowRankTrainer(
            model=model, args=training_args, processing_class=tokenizer, compute_metrics=compute_metrics, custom_args=args
        )
    else:
        trainer = ComparableBaselineTrainer(
            model=model, args=training_args, processing_class=tokenizer, compute_metrics=compute_metrics
        )
        
    # --- Run Evaluation ---
    all_results = {}
    
    print("\n--- Evaluating on Matched Validation Set ---")
    matched_results = trainer.evaluate(eval_dataset=eval_dataset_matched)
    all_results['matched'] = matched_results
    print(matched_results)

    print("\n--- Evaluating on Mismatched Validation Set ---")
    mismatched_results = trainer.evaluate(eval_dataset=eval_dataset_mismatched, metric_key_prefix="eval_mismatched")
    all_results['mismatched'] = mismatched_results
    print(mismatched_results)
    
    # --- Save Results to File ---
    with open(args.results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"✅ Evaluation complete. Results saved to {args.results_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)