import os
import argparse
import numpy as np
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed
)
import wandb
from lowrank.models.llm import CustomGemmaForSequenceClassification
from lowrank.models.layers import DynamicLowRankLinear, decompose_and_replace, replace_linear_with_lowrank
from lowrank.trainer import LowRankTrainer, ComparableBaselineTrainer
import math
import torch.nn as nn
from types import MethodType
from safetensors.torch import load_file

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune a Mistral 7B model on MNLI.")
    
    # --- Paths and Directories ---
    parser.add_argument('--data_dir', type=str, default='/mnt/pdata/pr501/shortcut-models/dataset',
                        help='Base directory for datasets.')
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/pdata/pr501/shortcut-models/checkpoints/',
                        help='Directory to save model checkpoints and outputs.')
    
    # --- Run Configuration ---
    parser.add_argument('--model_name', type=str, default='mistral-7b-mnli-baseline',
                        help='A name for the training run, used for logging and checkpointing.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    # --- Training Hyperparameters ---
    parser.add_argument('--max_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device for training.')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size per device for evaluation.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Initial learning rate.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Maximum sequence length for tokenization.')
    parser.add_argument('--base_model_id', type=str, default='google/gemma-2b', help='hugging face model')
    parser.add_argument('--max_rank', type=int, default=1024, help='Maximum rank of the model.')
    parser.add_argument('--subset_proportion', type=float, default=1.0, help='Proportion of the dataset to be used')
    parser.add_argument('--eval_subset_proportion', type=float, default=0.15)
    parser.add_argument('--sampling_strategy', type=str, default='uniform', help='Sampling strategy for the ranks during training time')
    parser.add_argument('--warmup_proportion', type=float, default=0.15,
                        help='Proportion of total training steps to use for single-rank warmup (e.g., 0.15 for 15%).')
    parser.add_argument('--avoid_lowrank', action='store_true')
    return parser.parse_args()

def main(args):
    """Main training and evaluation function."""
    
    # --- Setup ---
    set_seed(args.seed)
    # This will be the main directory for this specific run
    output_dir = os.path.join(args.checkpoint_path, args.model_name)
    # Set cache directory for datasets to avoid filling up home directory
    model_cache_dir = os.path.join(args.data_dir, 'huggingface_models')
    dataset_cache_dir = os.path.join(args.data_dir, "huggingface_datasets")
    
    # Initialize WandB logging
    wandb.init(project="low-rank-performance-llm", name=args.model_name, config=args)
    
    # --- Load Tokenizer and Model ---
    model_id = args.base_model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_cache_dir)
    # Mistral has no default pad token, so we use the EOS token for padding.
    tokenizer.pad_token = tokenizer.eos_token
   
   ### Custom block ---
    # The train_ranks will be generated automatically inside the model's __init__
    # # This pre-trainead model is currently not loading correctly. Reason is that the config doesnt match with the 
    # required custom layers because CustomGemma does not load them properly.

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_id,
        num_labels=3,
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=model_cache_dir,
    )

    if not args.avoid_lowrank:

        if 'Qwen' in  args.base_model_id or 'gemma' in  args.base_model_id:
            decompose_and_replace(model.model, max_rank=args.max_rank, target_substring='mlp')
        elif 'gpt-neo-2' in  args.base_model_id:
            decompose_and_replace(model.transformer, max_rank=args.max_rank, target_substring='mlp')
        elif 'pythia' in  args.base_model_id:
            decompose_and_replace(model.gpt_neox, max_rank=args.max_rank, target_substring='mlp')
        else:
            raise ValueError("Model name not found")

        #decompose_and_replace(model.model, max_rank=args.max_rank, target_substring="mlp")
        # 3. Manually attach the custom attributes your Trainer needs directly to the model INSTANCE.
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

    ## Dataset load
    print(f"Loading MNLI dataset with cache directory: {dataset_cache_dir}")
    dataset = load_dataset("glue", "mnli", cache_dir=dataset_cache_dir)

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
        )

    print("Tokenizing the dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=["premise", "hypothesis", "idx"]
    )

    # Select a subset if a proportion less than 1.0 is specified
    if args.subset_proportion < 1.0:
        print(f"💡 Using a subset of the data: {args.subset_proportion * 100:.1f}%")
        
        # Calculate the number of samples for each split
        num_train_samples = int(len(tokenized_dataset["train"]) * args.subset_proportion)
        num_eval_matched_samples = int(len(tokenized_dataset["validation_matched"]) * args.subset_proportion)
        num_eval_mismatched_samples = int(len(tokenized_dataset["validation_mismatched"]) * args.subset_proportion)
        
        # Use .select() to get the subset
        train_dataset = tokenized_dataset["train"].select(range(num_train_samples))
        eval_dataset_matched = tokenized_dataset["validation_matched"].select(range(num_eval_matched_samples))
        eval_dataset_mismatched = tokenized_dataset["validation_mismatched"].select(range(num_eval_mismatched_samples))
    else:
        # Use the full dataset
        train_dataset = tokenized_dataset["train"]
    
    print(f"Using {args.eval_subset_proportion*100:.1f}% of the val dataset")
    
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

    # --- Configure Trainer ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=args.model_name,
        num_train_epochs=args.max_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        warmup_ratio=0.02,
        max_grad_norm=1.0,
        # Logging and Saving
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=15,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        # Performance
        fp16=False, # bf16 is generally better for modern GPUs
        bf16=True,
        dataloader_num_workers=args.num_workers,
    )

    if not args.avoid_lowrank:
        trainer = LowRankTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset_matched, # Primary eval set during training
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
            custom_args=args,
        )

    else:

        trainer = ComparableBaselineTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset_matched, # Primary eval set during training
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
    # --- Run Training and Evaluation ---
    print("Starting training...")
    trainer.train()
    
    print("\n--- Saving final model ---")
    trainer.save_model(os.path.join(output_dir, "final_model"))

    print("\n--- Evaluating on Matched Validation Set ---")
    matched_results = trainer.evaluate(eval_dataset=eval_dataset_matched)
    print(matched_results)
    wandb.log({"final_eval_matched": matched_results})

    print("\n--- Evaluating on Mismatched Validation Set ---")
    mismatched_results = trainer.evaluate(eval_dataset=eval_dataset_mismatched)
    print(mismatched_results)
    wandb.log({"final_eval_mismatched": mismatched_results})
    
    wandb.finish()
    print("Run finished successfully.")

if __name__ == '__main__':
    args = parse_args()
    main(args)