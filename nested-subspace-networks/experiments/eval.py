import os
import argparse
import torch
import pytorch_lightning as pl
from lowrank.data.mnist import MNISTDataModule

from lowrank.models.mlp import LightMLP, DynamicMLP
import csv
import numpy as np
from lowrank.data.cifar import CIFAR10DataModule
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule

# ----
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['mlp', 'lowrank'], required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--eval_ranks', type=int, nargs='+', default=[64])
    parser.add_argument('--data_dir', type=str, default='/mnt/pdata/pr501/shortcut-models/dataset/mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--exp_name', type=str, required=True,
                    help="Name for the CSV file (saved to results/{exp_name}.csv)")
    parser.add_argument('--num_repeats', type=int, default=5,
                    help="Number of times to repeat evaluation for each rank (with different seeds)")
    
    parser.add_argument('--dataset', type=str, help='dataset to use')
    return parser.parse_args()

def set_rank(model, rank):
    for module in model.modules():
        if hasattr(module, "set_rank"):
            module.set_rank(rank)

def get_checkpoint_path(model_name, epoch):
    base_dir = "/mnt/pdata/pr501/shortcut-models/checkpoints"
    filename = f"{model_name}-epoch={epoch:03d}.ckpt"
    return os.path.join(base_dir, filename)

def main(args):
    pl.seed_everything(42)

    ckpt_path = get_checkpoint_path(args.model_name, args.epoch)
    
 ### -------- SELECT DATASET --------------
    if args.dataset.lower() == 'mnist':
        dm = MNISTDataModule(data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
        input_size = 28 * 28          # 1 × 28 × 28
        num_classes = 10
    elif args.dataset.lower() == 'cifar10':
        dm = CIFAR10DataModule(data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
        input_size = 32 * 32 * 3      # 3 × 32 × 32
        num_classes = 10
    elif args.dataset.lower() == 'cifar10_emb':
            dm = CIFAR10EmbeddingDataModule(
            feat_dir='/mnt/pdata/pr501/shortcut-models/dataset/cifar10_emb',
            batch_size=args.batch_size,
            num_workers=args.num_workers)
            input_size  = 512          # feature dim from ResNet-18
            num_classes = 10
    else:
        raise ValueError('dataset does not exist')

    # Output CSV file
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"{args.exp_name}.csv")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "mean_loss", "std_loss", "mean_acc", "std_acc"])

        if args.model_type == 'mlp':
            model = LightMLP.load_from_checkpoint(ckpt_path)
            trainer = pl.Trainer(accelerator="gpu", devices=[args.device_id])
            results = trainer.test(model=model, datamodule=dm, verbose=False)[0]
            writer.writerow([None, results["test/loss"], results["test/acc"]])
            print(f"✅ Baseline MLP: loss={results['test/loss']:.4f}, acc={results['test/acc']:.4f}")

        elif args.model_type == 'lowrank':
            model = DynamicMLP.load_from_checkpoint(ckpt_path)
            for r in args.eval_ranks:
                losses, accs = [], []

                print(f"\n🔍 Rank {r} evaluation ({args.num_repeats} runs)...")
                for i in range(args.num_repeats):
                    seed = 42 + i
                    pl.seed_everything(seed, workers=True)

                    model = DynamicMLP.load_from_checkpoint(ckpt_path)
                    set_rank(model, r)

                    trainer = pl.Trainer(accelerator="gpu", devices=[args.device_id], logger=False)
                    result = trainer.test(model=model, datamodule=dm, verbose=False)[0]

                    losses.append(result["test/loss"])
                    accs.append(result["test/acc"])

                mean_loss, std_loss = np.mean(losses), np.std(losses)
                mean_acc, std_acc = np.mean(accs), np.std(accs)

                writer.writerow([r, mean_loss, std_loss, mean_acc, std_acc])
                print(f"Rank {r}: acc = {mean_acc:.4f} ± {std_acc:.4f}, loss = {mean_loss:.4f} ± {std_loss:.4f}")

            
        else:
            raise ValueError("Invalid model type")

    print(f"\n📄 Results saved to: {csv_path}")

# ---------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)