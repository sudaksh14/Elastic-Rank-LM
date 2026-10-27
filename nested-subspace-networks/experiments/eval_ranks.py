import os
import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import wandb
import torch.nn.functional as F
# --- MINIMAL ADDITION: Import DataLoader and Subset for bootstrapping ---
from torch.utils.data import DataLoader, Subset

# --- Import your project's modules ---
from lowrank.models.mlp import DynamicMLP
from lowrank.data.cifar import CIFAR10DataModule
from lowrank.data.mnist import MNISTDataModule
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule

# --- Configuration (Unchanged) ---
RUNS_TO_EVALUATE = {
    'CE Only (Anchor)': ['for-paper-low-rank-teacher-only', 'pauliusrauba/low-rank-performance/pwu2ynn4'],
    'One CE + Hard Orthogonality': ['for-paper-low-rank-hard-orthogonality', 'pauliusrauba/low-rank-performance/8gjuyn7e'],
    'Two CEs': ['for-paper-low-rank-ce-no-consistency', 'pauliusrauba/low-rank-performance/2ciim0l9'],
    'Two CEs + Logits Regularization': ['for-paper-low-rank-ce-with-consistency', 'pauliusrauba/low-rank-performance/y0kespia'],
    'Two CEs + Residual Orthogonality': ['for-paper-low-rank-ce-with-residual-orthogonality', 'pauliusrauba/low-rank-performance/enjfnffl'],
    'Two CEs + Hidden Regularization': ['for-paper-low-rank-ce-with-feature-consistency', 'pauliusrauba/low-rank-performance/czeuga4y'],
}
IN_D_RANKS = [64, 32, 16, 8, 4, 2, 1]
OOD_RANKS = [3, 5, 10, 20, 30, 40, 50]


def find_best_checkpoint(model_name, run_path, chkpt_base_dir):
    """ (Unchanged from your version) """
    print(f"\n🔍 Searching for best checkpoint for '{model_name}' ({run_path})")
    api = wandb.Api()
    try:
        run = api.run(run_path)
    except wandb.errors.CommError:
        print(f"❌ ERROR: Run '{run_path}' not found.")
        return None, None, None

    history = run.scan_history(keys=["epoch", "val/loss"])
    df = pd.DataFrame(history).dropna().drop_duplicates(subset='epoch', keep='last')

    if df.empty or 'val/loss' not in df.columns:
        print("❌ ERROR: No 'val/loss' metrics found for this run.")
        return None, None, None

    save_every = run.config.get('save_every_n_epochs', 10)
    max_epochs = run.config.get('max_epochs', 100)
    first_save_epoch = save_every - 1
    saved_epochs = np.arange(first_save_epoch, max_epochs, save_every)
    
    df_filtered = df[df.epoch.isin(saved_epochs)].copy()

    if df_filtered.empty:
        print(f"❌ ERROR: No validation metrics found for the expected saved epochs.")
        return None, None, None

    best_idx = df_filtered['val/loss'].idxmin()
    best_epoch = int(df_filtered.loc[best_idx]['epoch'])
    
    checkpoint_filename = f"{model_name}-epoch={best_epoch:03d}.ckpt"
    checkpoint_path = os.path.join(chkpt_base_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ ERROR: Checkpoint file not found at: {checkpoint_path}")
        return None, None, run.config

    print(f"✅ Best saved checkpoint found at epoch {best_epoch}. Path: {checkpoint_path}")
    return checkpoint_path, best_epoch, run.config


@torch.no_grad()
def evaluate_on_test_set(model, dataloader, device):
    """ (This is your original evaluation function, now used for each bootstrap sample) """
    model.eval()
    model.to(device)
    total_correct, total_loss, total_seen = 0, 0.0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += loss.item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total_seen += y.size(0)
    acc = total_correct / total_seen if total_seen > 0 else 0
    avg_loss = total_loss / total_seen if total_seen > 0 else 0
    return avg_loss, acc


def get_datamodule(dataset_name, data_dir, batch_size, num_workers):
    """ (Unchanged from your version) """
    if dataset_name == 'mnist':
        return MNISTDataModule(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == 'cifar10':
        return CIFAR10DataModule(data_dir=os.path.join(data_dir, 'cifar10'), batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == 'cifar10_emb':
        feat_dir = os.path.join(data_dir, 'cifar10_emb')
        return CIFAR10EmbeddingDataModule(feat_dir=feat_dir, batch_size=batch_size, num_workers=num_workers)
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")


def main(args):
    """ --- MODIFIED MAIN LOOP WITH MANUAL BOOTSTRAPPING --- """
    pl.seed_everything(42)
    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    all_results = []
    
    all_eval_ranks = sorted(list(set(IN_D_RANKS + OOD_RANKS)), reverse=True)

    for display_name, info in RUNS_TO_EVALUATE.items():
        model_name, run_path = info
        
        ckpt_path, best_epoch, config = find_best_checkpoint(model_name, run_path, args.checkpoint_dir)
        if not ckpt_path:
            continue

        model = DynamicMLP.load_from_checkpoint(ckpt_path)
        dm = get_datamodule(args.dataset, args.data_dir, args.batch_size, args.num_workers)
        dm.setup('test')
        
        # --- FIX: Get the dataset from the dataloader to solve the AttributeError ---
        test_dataset = dm.test_dataloader().dataset
        n_samples = len(test_dataset)

        current_run_results = {"Experiment Name": display_name, "Best Epoch": best_epoch}
        print(f"--- Evaluating: {display_name} ---")

        # --- Manual Bootstrap for Highest Rank ---
        highest_rank = model.hparams.rank
        model._set_rank_all(highest_rank)
        
        bootstrap_losses, bootstrap_accs = [], []
        print(f"  ...running {args.num_bootstraps} bootstraps for rank {highest_rank}...")
        for _ in range(args.num_bootstraps):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_subset = Subset(test_dataset, bootstrap_indices)
            bootstrap_loader = DataLoader(
                bootstrap_subset, batch_size=args.batch_size, num_workers=args.num_workers
            )
            loss, acc = evaluate_on_test_set(model, bootstrap_loader, device)
            bootstrap_losses.append(loss)
            bootstrap_accs.append(acc)

        # Calculate and store stats
        mean_loss, std_loss = np.mean(bootstrap_losses), np.std(bootstrap_losses)
        mean_acc, std_acc = np.mean(bootstrap_accs), np.std(bootstrap_accs)
        current_run_results[f"Test Loss (Rank {highest_rank})"] = mean_loss
        current_run_results[f"Test Loss Std (Rank {highest_rank})"] = std_loss
        current_run_results[f"Test Acc (Rank {highest_rank})"] = mean_acc
        current_run_results[f"Test Acc Std (Rank {highest_rank})"] = std_acc
        print(f"  Rank {highest_rank} (Trained): Acc={mean_acc:.4f}±{std_acc:.4f}, Loss={mean_loss:.4f}±{std_loss:.4f}")

        # --- Manual Bootstrap for Other Ranks ---
        for r in all_eval_ranks:
            model._set_rank_all(r)
            
            bootstrap_accs_r = []
            print(f"  ...running {args.num_bootstraps} bootstraps for rank {r}...")
            for _ in range(args.num_bootstraps):
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                bootstrap_subset = Subset(test_dataset, bootstrap_indices)
                bootstrap_loader = DataLoader(
                    bootstrap_subset, batch_size=args.batch_size, num_workers=args.num_workers
                )
                _, acc = evaluate_on_test_set(model, bootstrap_loader, device)
                bootstrap_accs_r.append(acc)

            # Calculate and store stats
            mean_acc_r, std_acc_r = np.mean(bootstrap_accs_r), np.std(bootstrap_accs_r)
            current_run_results[f"Rank {r} Acc"] = mean_acc_r
            current_run_results[f"Rank {r} Acc Std"] = std_acc_r
            print(f"  Rank {r:<3}: Acc={mean_acc_r:.4f}±{std_acc_r:.4f}")

        all_results.append(current_run_results)

    if not all_results:
        print("\nNo results were generated. Exiting.")
        return

    df = pd.DataFrame(all_results)
    
    # --- Update column ordering for the new Std columns ---
    cols_order = ["Experiment Name", "Best Epoch"]
    base_metrics = sorted([c for c in df.columns if "Test " in c], reverse=True)
    cols_order.extend(base_metrics)
    
    rank_metrics = sorted([c for c in df.columns if "Rank " in c and "Test" not in c], 
                           key=lambda x: (int(x.split(' ')[1]), "Std" in x), reverse=True)
    cols_order.extend(rank_metrics)
    
    df = df.reindex(columns=cols_order)
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", args.output_filename)
    df.to_csv(output_path, index=False, float_format='%.4f')
    
    print("\n\n✅ All evaluations complete. Results saved to:", output_path)
    print(df.to_string())


def parse_args():
    """ --- MODIFIED: Add num_bootstraps argument --- """
    parser = argparse.ArgumentParser(description="Run a comprehensive evaluation of multiple models.")
    parser.add_argument('--data_dir', type=str, default='/mnt/pdata/pr501/shortcut-models/dataset', help='Base directory for all datasets.')
    parser.add_argument('--checkpoint_dir', type=str, default='/mnt/pdata/pr501/shortcut-models/checkpoints', help='Base directory where checkpoints are saved.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device_id', type=int, default=0, help='GPU device to use.')
    parser.add_argument('--output_filename', type=str, default="evaluation_results_with_std.csv", help="Name for the output CSV file.")
    parser.add_argument('--dataset', type=str, default='cifar10_emb')
    # New argument to control bootstrapping
    parser.add_argument('--num_bootstraps', type=int, default=5, help="Number of bootstrap samples for uncertainty estimation.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)