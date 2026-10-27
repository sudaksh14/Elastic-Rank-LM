"""
Experiment 03: Ablation Study of Training Objectives on CIFAR-10 Embeddings
============================================================================

Trains a DynamicMLP (from lowrank_mlp_ce) at rank 64 with each loss variant,
then evaluates at in-distribution ranks [64,32,16,8,4,2,1] and out-of-distribution
ranks [3,5,10,20,30,40,50].  Produces the ablation table:

  Method | Highest Test Acc | Avg. ID Acc | Avg. OOD Acc

Usage:
    uv run python experiments/exp_03.py \
        --data_dir data/cifar10_emb \
        --checkpoint_dir checkpoints/exp_03 \
        --gpus 0

    # Evaluation only:
    uv run python experiments/exp_03.py \
        --data_dir data/cifar10_emb \
        --checkpoint_dir checkpoints/exp_03 \
        --eval_only --gpus 0

Outputs (saved to results/exp_03/):
    - ablation_summary.csv:   The aggregate table (Highest, Avg ID, Avg OOD)
    - ablation_per_rank.csv:  Per-rank accuracy for every method
"""

import sys
import os
import csv
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lowrank.models.lowrank_mlp_ce import DynamicMLP
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule

# Table rows in the order they should appear
ABLATIONS = OrderedDict([
    ('CE Only (Anchor)',            'ce_teacher_only'),
    ('One CE + Hard Ortho.',        'ce_orthogonality_hard_no_stud'),
    ('Two CEs',                     'ce_no_consistency'),
    ('Two CEs + Logits Reg.',       'ce_with_consistency'),
    ('Two CEs + Residual Ortho.',   'ce_orthogonality'),
    ('Two CEs + Hidden Reg.',       'ce_with_feature_consistency'),
])

ID_RANKS = [64, 32, 16, 8, 4, 2, 1]
OOD_RANKS = [3, 5, 10, 20, 30, 40, 50]
ALL_EVAL_RANKS = sorted(set(ID_RANKS + OOD_RANKS), reverse=True)
MAX_RANK = 64
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_03')


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_checkpoint_path(checkpoint_dir, model_name, epoch):
    return os.path.join(checkpoint_dir, f"{model_name}-epoch={epoch:03d}.ckpt")


def find_best_checkpoint(checkpoint_dir, model_name, max_epochs, save_every):
    for epoch in range(max_epochs - 1, -1, -save_every):
        path = get_checkpoint_path(checkpoint_dir, model_name, epoch)
        if os.path.exists(path):
            return path, epoch
    return None, None


@torch.no_grad()
def evaluate_model(model, dataloader, device):
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
# Training
# ---------------------------------------------------------------------------

def train_ablation_models(args):
    for display_name, loss_variant in ABLATIONS.items():
        model_name = f"exp03-{loss_variant}"
        existing, _ = find_best_checkpoint(
            args.checkpoint_dir, model_name, args.max_epochs, args.save_every,
        )
        if existing:
            print(f"[{display_name}] checkpoint exists, skipping training.")
            continue

        print(f"\n{'='*60}")
        print(f"Training: {display_name}  (loss_variant={loss_variant})")
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
            rank=MAX_RANK, teacher_mode='max',
            loss_variant=loss_variant,
        )

        ckpt_cb = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            every_n_epochs=args.save_every,
            filename=f"{model_name}-{{epoch:03d}}",
            save_top_k=-1,
        )

        pl.Trainer(
            max_epochs=args.max_epochs, accelerator="gpu", devices=[args.gpus],
            callbacks=[ckpt_cb], precision='bf16-mixed',
            logger=False, enable_progress_bar=True,
        ).fit(model, dm)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ablation_models(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = f"cuda:{args.gpus}"

    dm = CIFAR10EmbeddingDataModule(
        feat_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup('test')
    test_dataset = dm.test_dataloader().dataset
    n_samples = len(test_dataset)

    per_rank_rows = []
    summary_rows = []

    for display_name, loss_variant in ABLATIONS.items():
        model_name = f"exp03-{loss_variant}"
        ckpt_path, epoch = find_best_checkpoint(
            args.checkpoint_dir, model_name, args.max_epochs, args.save_every,
        )
        if ckpt_path is None:
            print(f"No checkpoint for {model_name}, skipping.")
            continue

        print(f"\n--- Evaluating: {display_name} (epoch={epoch}) ---")
        model = DynamicMLP.load_from_checkpoint(ckpt_path)
        model.eval()
        model.to(device)

        # Per-bootstrap, per-rank accuracies
        boot_rank_accs = {r: [] for r in ALL_EVAL_RANKS}

        for b in range(args.num_bootstraps):
            np.random.seed(42 + b)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            loader = DataLoader(
                Subset(test_dataset, indices),
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            for rank in ALL_EVAL_RANKS:
                model._set_rank_all(rank)
                _, acc = evaluate_model(model, loader, device)
                boot_rank_accs[rank].append(acc)

        # Per-rank results
        rank_means = {}
        for rank in ALL_EVAL_RANKS:
            m, s = np.mean(boot_rank_accs[rank]), np.std(boot_rank_accs[rank])
            rank_means[rank] = m
            per_rank_rows.append({
                "method": display_name, "rank": rank,
                "mean_acc": f"{m:.4f}", "std_acc": f"{s:.4f}",
            })
            print(f"  rank {rank:<3}: acc = {m:.4f} +/- {s:.4f}")

        # Aggregate metrics (bootstrap-aware)
        boot_highest = boot_rank_accs[MAX_RANK]
        boot_avg_id = [
            np.mean([boot_rank_accs[r][b] for r in ID_RANKS])
            for b in range(args.num_bootstraps)
        ]
        boot_avg_ood = [
            np.mean([boot_rank_accs[r][b] for r in OOD_RANKS])
            for b in range(args.num_bootstraps)
        ]

        summary_rows.append({
            "method": display_name,
            "highest_test_acc": f"{np.mean(boot_highest):.2f} +/- {np.std(boot_highest):.2f}",
            "avg_id_acc":       f"{np.mean(boot_avg_id):.2f} +/- {np.std(boot_avg_id):.2f}",
            "avg_ood_acc":      f"{np.mean(boot_avg_ood):.2f} +/- {np.std(boot_avg_ood):.2f}",
        })
        print(f"  => Highest={np.mean(boot_highest):.4f}  "
              f"Avg ID={np.mean(boot_avg_id):.4f}  "
              f"Avg OOD={np.mean(boot_avg_ood):.4f}")

    # Write per-rank CSV
    per_rank_path = os.path.join(RESULTS_DIR, "ablation_per_rank.csv")
    with open(per_rank_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=["method", "rank", "mean_acc", "std_acc"])
        w.writeheader()
        w.writerows(per_rank_rows)
    print(f"\nPer-rank results saved to {per_rank_path}")

    # Write summary CSV
    summary_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "highest_test_acc", "avg_id_acc", "avg_ood_acc",
        ])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Summary table saved to {summary_path}")

    # Pretty-print the summary
    print(f"\n{'Method':<30} {'Highest':>14} {'Avg ID':>14} {'Avg OOD':>14}")
    print("-" * 76)
    for row in summary_rows:
        print(f"{row['method']:<30} {row['highest_test_acc']:>14} "
              f"{row['avg_id_acc']:>14} {row['avg_ood_acc']:>14}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--num_bootstraps', type=int, default=5)
    parser.add_argument('--eval_only', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.eval_only:
        train_ablation_models(args)

    evaluate_ablation_models(args)

    print(f"\nDone. Results are in {RESULTS_DIR}/")
