"""
Experiment 01: Native-Rank Training vs Rank Truncation on CIFAR-10 Embeddings
==============================================================================

This experiment compares two strategies for obtaining a low-rank MLP
(using DynamicLowRankLinear layers trained with simple max-rank CE):

  1. **Native-rank training**: Train a separate model at each rank r in {1,2,4,8,16,32,64}.
  2. **Rank truncation**: Train one model at rank 64, then truncate to lower ranks at test time.

Usage:
    # Full pipeline (train + evaluate):
    uv run python experiments/exp_01.py \
        --data_dir data/cifar10_emb \
        --checkpoint_dir checkpoints/exp_01 \
        --gpus 0

    # Evaluation only (from existing checkpoints):
    uv run python experiments/exp_01.py \
        --data_dir data/cifar10_emb \
        --checkpoint_dir checkpoints/exp_01 \
        --eval_only \
        --gpus 0

Outputs (saved to results/exp_01/):
    - native_rank_results.csv:    Test accuracy per natively-trained rank
    - truncated_rank_results.csv: Test accuracy when truncating a rank-64 model
"""

import sys
import os
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lowrank.models.mlp import DynamicMLP
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule

RANKS = [64, 32, 16, 8, 4, 2, 1]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_01')


def get_checkpoint_path(checkpoint_dir, model_name, epoch):
    filename = f"{model_name}-epoch={epoch:03d}.ckpt"
    return os.path.join(checkpoint_dir, filename)


def find_best_checkpoint(checkpoint_dir, model_name, max_epochs, save_every):
    """Find the last saved checkpoint for a model."""
    for epoch in range(max_epochs - 1, -1, -save_every):
        path = get_checkpoint_path(checkpoint_dir, model_name, epoch)
        if os.path.exists(path):
            return path, epoch
    return None, None


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Manual evaluation loop that respects the currently set active rank."""
    model.eval()
    model.to(device)
    correct, total_loss, n = 0, 0.0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction='sum').item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def train_native_ranks(args):
    """Train one low-rank MLP per rank using simple max-rank CE."""
    for rank in RANKS:
        model_name = f"exp01-native-rank-{rank}"
        print(f"\n{'='*60}")
        print(f"Training native model at rank {rank}")
        print(f"{'='*60}")

        pl.seed_everything(42, workers=True)
        torch.set_float32_matmul_precision('high')

        dm = CIFAR10EmbeddingDataModule(
            feat_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        model = DynamicMLP(
            input_size=512,
            num_classes=10,
            rank=rank,
            teacher_mode='none',
        )

        ckpt_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            every_n_epochs=args.save_every,
            filename=f"{model_name}-{{epoch:03d}}",
            save_top_k=-1,
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
            devices=[args.gpus],
            callbacks=[ckpt_callback],
            precision='bf16-mixed',
            logger=False,
            enable_progress_bar=True,
        )
        trainer.fit(model, dm)


def evaluate_native_ranks(args):
    """Evaluate each natively-trained model at its own rank."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "native_rank_results.csv")
    device = f"cuda:{args.gpus}"

    dm = CIFAR10EmbeddingDataModule(
        feat_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup('test')
    test_loader = dm.test_dataloader()

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "test_loss", "test_acc"])

        for rank in RANKS:
            model_name = f"exp01-native-rank-{rank}"
            ckpt_path, epoch = find_best_checkpoint(
                args.checkpoint_dir, model_name, args.max_epochs, args.save_every
            )
            if ckpt_path is None:
                print(f"No checkpoint found for {model_name}, skipping.")
                continue

            print(f"\nEvaluating native model: rank={rank}, epoch={epoch}")
            model = DynamicMLP.load_from_checkpoint(ckpt_path)
            model._set_rank_all(rank)

            loss, acc = evaluate_model(model, test_loader, device)
            writer.writerow([rank, f"{loss:.6f}", f"{acc:.4f}"])
            print(f"  rank {rank}: acc = {acc:.4f}, loss = {loss:.6f}")

    print(f"\nNative rank results saved to {csv_path}")


def evaluate_truncated_ranks(args):
    """Load the rank-64 model and evaluate it at all lower ranks via truncation."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "truncated_rank_results.csv")
    device = f"cuda:{args.gpus}"

    model_name = "exp01-native-rank-64"
    ckpt_path, epoch = find_best_checkpoint(
        args.checkpoint_dir, model_name, args.max_epochs, args.save_every
    )
    if ckpt_path is None:
        print(f"No checkpoint found for {model_name}.")
        return

    print(f"\nEvaluating rank-64 model at all ranks (checkpoint epoch={epoch})")

    dm = CIFAR10EmbeddingDataModule(
        feat_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup('test')
    test_loader = dm.test_dataloader()

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "test_loss", "test_acc"])

        for rank in RANKS:
            model = DynamicMLP.load_from_checkpoint(ckpt_path)
            model._set_rank_all(rank)

            loss, acc = evaluate_model(model, test_loader, device)
            writer.writerow([rank, f"{loss:.6f}", f"{acc:.4f}"])
            print(f"  rank {rank}: acc = {acc:.4f}, loss = {loss:.6f}")

    print(f"\nTruncated rank results saved to {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CIFAR-10 embedding directory (contains train.pt, test.pt)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--gpus', type=int, default=0, help='GPU device index')
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, only evaluate existing checkpoints')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.eval_only:
        train_native_ranks(args)

    evaluate_native_ranks(args)
    evaluate_truncated_ranks(args)

    print(f"\nDone. Results are in {RESULTS_DIR}/")
    print("  - native_rank_results.csv")
    print("  - truncated_rank_results.csv")
