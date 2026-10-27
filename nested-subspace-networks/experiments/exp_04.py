"""
Experiment 04: Learned Log Variances During Multi-Rank Training
================================================================

Trains a DynamicMLP (from lowrank.models.mlp) at rank 64 with teacher_mode='max',
tracks the learned per-rank log-variance parameters (s_rank) across training steps,
and produces a visualization of how these uncertainty weights evolve.

These are the same s_rank values that get logged to wandb and visualised in
notebooks/3. experiment_figures.ipynb, but captured locally via a custom
callback instead.

Usage:
    uv run python experiments/exp_04.py \\
        --data_dir data/cifar10_emb \\
        --checkpoint_dir checkpoints/exp_04 \\
        --gpus 0

    # Plot only (from existing CSV):
    uv run python experiments/exp_04.py --plot_only

Outputs (saved to results/exp_04/):
    - log_variance_history.csv:  Per-step s_rank values for each rank
    - show_log_variances.pdf:    Log-variance-vs-global-step plot
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lowrank.models.mlp import DynamicMLP
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule

MAX_RANK = 64
RANKS_TO_TRACK = [1, 2, 4, 8, 16, 32, 64]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_04')


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

class LogVarianceTracker(pl.Callback):
    """Records all log_vars_ce parameter values at every training step."""

    def __init__(self):
        self.records = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        record = {'global_step': trainer.global_step}
        for rank_str, param in pl_module.log_vars_ce.items():
            record[f's_rank_{rank_str}'] = param.item()
        self.records.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.records)


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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_track(args):
    model_name = "exp04-log-variance"

    existing, _ = find_best_checkpoint(
        args.checkpoint_dir, model_name, args.max_epochs, args.save_every,
    )
    if existing and not args.force:
        print(f"Checkpoint exists ({existing}), skipping training. "
              "Use --force to retrain.")
        return None

    print(f"\n{'='*60}")
    print(f"Training: DynamicMLP (rank={MAX_RANK}, teacher_mode=max)")
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
    )

    tracker = LogVarianceTracker()

    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        every_n_epochs=args.save_every,
        filename=f"{model_name}-{{epoch:03d}}",
        save_top_k=-1,
    )

    pl.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=[args.gpus],
        callbacks=[ckpt_cb, tracker], precision='bf16-mixed',
        logger=False, enable_progress_bar=True,
    ).fit(model, dm)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = tracker.to_dataframe()

    s_rank_cols = sorted(
        [c for c in df.columns if c.startswith('s_rank_')],
        key=lambda c: int(c.split('_')[-1]),
    )
    df = df[['global_step'] + s_rank_cols]

    csv_path = os.path.join(RESULTS_DIR, "log_variance_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nLog variance history saved to {csv_path}")
    print(f"  Shape: {df.shape[0]} steps x {len(s_rank_cols)} ranks")

    return df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_log_variances(df=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if df is None:
        csv_path = os.path.join(RESULTS_DIR, "log_variance_history.csv")
        if not os.path.exists(csv_path):
            print(f"No data found at {csv_path}. Run training first.")
            return
        df = pd.read_csv(csv_path)

    s_rank_cols = [f's_rank_{r}' for r in RANKS_TO_TRACK
                   if f's_rank_{r}' in df.columns]
    ranks_present = [r for r in RANKS_TO_TRACK
                     if f's_rank_{r}' in df.columns]

    if not s_rank_cols:
        print("No s_rank columns found in data.")
        return

    cmap = plt.get_cmap('YlOrBr')
    colors = cmap(np.linspace(0.3, 0.95, len(s_rank_cols)))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, col in enumerate(s_rank_cols):
        rank = ranks_present[i]
        plot_df = df[['global_step', col]].dropna()
        ax.plot(plot_df['global_step'], plot_df[col],
                color=colors[i], linewidth=3)
        last_x = plot_df['global_step'].iloc[-1]
        last_y = plot_df[col].iloc[-1]
        ax.text(last_x * 1.03, last_y, f'Rank = {rank}',
                fontsize=13, color=colors[i], va='center', fontweight='bold')

    ax.set_xlabel('Global Step', fontsize=14)
    ax.set_ylabel('Log Variance', fontsize=14)
    ax.set_title('Log Variance (s_rank) by Rank',
                 fontsize=16, fontweight='500', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

    fig.tight_layout()
    pdf_path = os.path.join(RESULTS_DIR, "show_log_variances.pdf")
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {pdf_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data_dir', type=str, default='data/cifar10_emb',
                        help='Path to CIFAR-10 embedding directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/exp_04',
                        help='Directory to save/load model checkpoints')
    parser.add_argument('--gpus', type=int, default=0, help='GPU device index')
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--plot_only', action='store_true',
                        help='Skip training, regenerate plot from existing CSV')
    parser.add_argument('--force', action='store_true',
                        help='Retrain even if checkpoint exists')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not args.plot_only:
        df = train_and_track(args)
        plot_log_variances(df)
    else:
        plot_log_variances()

    print(f"\nDone. Results are in {RESULTS_DIR}/")
    print("  - log_variance_history.csv")
    print("  - show_log_variances.pdf")
