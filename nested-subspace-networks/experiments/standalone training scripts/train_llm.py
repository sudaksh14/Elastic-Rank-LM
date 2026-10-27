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
    # --- Model Specific (Placeholders for your future work) ---
    # parser.add_argument('--model_type', type=str, choices=['baseline', 'lowrank'], default='baseline')
    # parser.add_argument('--train_rank', type=int, default=64)
    
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
    # --- MODIFIED BLOCK: Explicitly load config for custom models like Phi-3 ---
    trust_remote = True if 'phi-3' in model_id else False
    
    # 1. Load the configuration and specify our classification needs
    config = AutoConfig.from_pretrained(
        model_id, 
        num_labels=3, 
        trust_remote_code=trust_remote, 
        cache_dir=model_cache_dir
    )
    config.pad_token_id = tokenizer.pad_token_id

    # 2. Load the model using the prepared config
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        config=config, # Pass the explicit config
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote,
        cache_dir=model_cache_dir,
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    trust_remote = True if 'phi-3' in model_id else False

    # --- Load and Preprocess Dataset ---
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
    
    train_dataset = tokenized_dataset["train"]
    eval_dataset_matched = tokenized_dataset["validation_matched"]
    eval_dataset_mismatched = tokenized_dataset["validation_mismatched"]

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
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        
        # Logging and Saving
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="wandb",
        
        # Performance
        fp16=False, # bf16 is generally better for modern GPUs
        bf16=True,
        dataloader_num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_matched, # Primary eval set during training
        tokenizer=tokenizer,
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