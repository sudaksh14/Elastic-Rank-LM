"""
Experiment 02: Native-Rank Training vs Nested (epoch_curriculum) on CIFAR-10 Embeddings
========================================================================================

Compares two strategies for a low-rank MLP (DynamicLowRankLinear):
  1. **Native-rank training**: Train a separate model at each rank (teacher_mode='none').
     Reuses checkpoints from exp_01 if they exist, otherwise trains from scratch.
  2. **Nested training**: Train ONE model at rank 64 with teacher_mode='epoch_curriculum'
     (multi-rank anchor/variant CE with curriculum sampling), then evaluate at all ranks.

Usage:
    uv run python experiments/exp_02.py \
        --data_dir data/cifar10_emb \
        --checkpoint_dir checkpoints/exp_02 \
        --native_checkpoint_dir checkpoints/exp_01 \
        --gpus 0

    # Evaluation only:
    uv run python experiments/exp_02.py \
        --data_dir data/cifar10_emb \
        --checkpoint_dir checkpoints/exp_02 \
        --native_checkpoint_dir checkpoints/exp_01 \
        --eval_only --gpus 0

Outputs (saved to results/exp_02/):
    - native_rank_results.csv:  Test accuracy per natively-trained rank
    - nested_rank_results.csv:  Test accuracy of the nested model evaluated at each rank
"""

import sys
import os
import csv
import argparse

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lowrank.models.mlp import DynamicMLP
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule

RANKS = [64, 32, 16, 8, 4, 2, 1]
MAX_RANK = 64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_02')


# ---------------------------------------------------------------------------
# Shared utilities (identical to exp_01)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Native-rank training  (reuses exp_01 checkpoints when available)
# ---------------------------------------------------------------------------

def train_native_ranks(args):
    """Train one DynamicMLP per rank with teacher_mode='none'.

    Skips any rank whose checkpoint already exists in native_checkpoint_dir.
    """
    for rank in RANKS:
        model_name = f"exp01-native-rank-{rank}"
        existing, _ = find_best_checkpoint(
            args.native_checkpoint_dir, model_name, args.max_epochs, args.save_every
        )
        if existing:
            print(f"Native rank-{rank} checkpoint already exists, skipping training.")
            continue

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
            input_size=512, num_classes=10,
            rank=rank, teacher_mode='none',
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=args.native_checkpoint_dir,
            every_n_epochs=args.save_every,
            filename=f"{model_name}-{{epoch:03d}}",
            save_top_k=-1,
        )

        pl.Trainer(
            max_epochs=args.max_epochs, accelerator="gpu", devices=[args.gpus],
            callbacks=[ckpt_cb], precision='bf16-mixed',
            logger=False, enable_progress_bar=True,
        ).fit(model, dm)


def evaluate_native_ranks(args):
    """Evaluate each natively-trained model at its own rank."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "native_rank_results.csv")
    device = f"cuda:{args.gpus}"

    dm = CIFAR10EmbeddingDataModule(
        feat_dir=args.data_dir, batch_size=args.batch_size,
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
                args.native_checkpoint_dir, model_name,
                args.max_epochs, args.save_every,
            )
            if ckpt_path is None:
                print(f"No checkpoint for {model_name}, skipping.")
                continue

            print(f"\nEvaluating native model: rank={rank}, epoch={epoch}")
            model = DynamicMLP.load_from_checkpoint(ckpt_path)
            model._set_rank_all(rank)

            loss, acc = evaluate_model(model, test_loader, device)
            writer.writerow([rank, f"{loss:.6f}", f"{acc:.4f}"])
            print(f"  rank {rank}: acc = {acc:.4f}, loss = {loss:.6f}")

    print(f"\nNative rank results saved to {csv_path}")


# ---------------------------------------------------------------------------
# Nested (epoch_curriculum) training
# ---------------------------------------------------------------------------

NESTED_MODEL_NAME = "exp02-nested-rank-64"


def train_nested_model(args):
    """Train a single DynamicMLP at rank 64 with epoch_curriculum multi-rank objective."""
    existing, _ = find_best_checkpoint(
        args.checkpoint_dir, NESTED_MODEL_NAME, args.max_epochs, args.save_every,
    )
    if existing:
        print(f"Nested model checkpoint already exists ({existing}), skipping training.")
        return

    print(f"\n{'='*60}")
    print(f"Training nested model (epoch_curriculum) at max rank {MAX_RANK}")
    print(f"{'='*60}")

    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    dm = CIFAR10EmbeddingDataModule(
        feat_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = DynamicMLP(
        input_size=512, num_classes=10,
        rank=MAX_RANK, teacher_mode='epoch_curriculum',
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        every_n_epochs=args.save_every,
        filename=f"{NESTED_MODEL_NAME}-{{epoch:03d}}",
        save_top_k=-1,
    )

    pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=[args.gpus],
        callbacks=[ckpt_cb], precision='bf16-mixed',
        logger=False, enable_progress_bar=True,
    ).fit(model, dm)


def evaluate_nested_ranks(args):
    """Evaluate the nested (epoch_curriculum) model at every rank."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "nested_rank_results.csv")
    device = f"cuda:{args.gpus}"

    ckpt_path, epoch = find_best_checkpoint(
        args.checkpoint_dir, NESTED_MODEL_NAME, args.max_epochs, args.save_every,
    )
    if ckpt_path is None:
        print(f"No checkpoint for {NESTED_MODEL_NAME}.")
        return

    print(f"\nEvaluating nested model at all ranks (checkpoint epoch={epoch})")

    dm = CIFAR10EmbeddingDataModule(
        feat_dir=args.data_dir, batch_size=args.batch_size,
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

    print(f"\nNested rank results saved to {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CIFAR-10 embedding dir (train.pt, test.pt)')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Dir to save/load the nested model checkpoint')
    parser.add_argument('--native_checkpoint_dir', type=str, default=None,
                        help='Dir with native-rank checkpoints (default: checkpoints/exp_01)')
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, only evaluate existing checkpoints')
    args = parser.parse_args()

    if args.native_checkpoint_dir is None:
        args.native_checkpoint_dir = os.path.join(
            os.path.dirname(__file__), '..', 'checkpoints', 'exp_01',
        )

    return args


if __name__ == '__main__':
    args = parse_args()

    if not args.eval_only:
        train_native_ranks(args)
        train_nested_model(args)

    evaluate_native_ranks(args)
    evaluate_nested_ranks(args)

    print(f"\nDone. Results are in {RESULTS_DIR}/")
    print("  - native_rank_results.csv")
    print("  - nested_rank_results.csv")
