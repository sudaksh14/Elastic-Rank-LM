import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lowrank.data.mnist import MNISTDataModule
from lowrank.models.mlp import LightMLP, DynamicMLP
import argparse
from pytorch_lightning import seed_everything
import wandb
from lowrank.data.cifar import CIFAR10DataModule
from lowrank.data.cifar_emb import CIFAR10EmbeddingDataModule
from lowrank.callbacks import RankEvalCallback
import math, random

import os
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/pdata/pr501/shortcut-models/dataset/mnist')
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/pdata/pr501/shortcut-models/checkpoints/')
    parser.add_argument('--max_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='mlp-baseline')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help = 'List of GPU devices')
    parser.add_argument('--save_every_n_epochs', type=int, default=5, help='How many epochs to save')

    parser.add_argument('--model_type', type=str, choices=['mlp', 'lowrank'], default='mlp')
    parser.add_argument('--train_rank', type=int, default=64)
    parser.add_argument('--test_rank', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='mnist', help='which dataset to train on')
    parser.add_argument('--teacher_mode', type=str, default='max')
    parser.add_argument('--loss_variant',
                        choices=['ce_no_consistency', 'ce_with_consistency', 'ce_with_feature_consistency', 'ce_orthogonality', 'ce_orthogonality_hard', 'ce_orthogonality_hard_no_stud', 'ce_teacher_only'],
                        default='ce_with_consistency')

    return parser.parse_args()

def main(args):
    # Seed
    seed_everything(42, workers=True)  # full reproducibility
    torch.set_float32_matmul_precision('high')

    # Get num workers
    cpu_cores = os.cpu_count()
    gpu_count = len(args.gpus)
    num_workers = min(8, args.batch_size, cpu_cores // max(gpu_count, 1))


    ### -------- SELECT DATASET --------------
    if args.dataset.lower() == 'mnist':
        dm = MNISTDataModule(data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=num_workers)
        input_size = 28 * 28          # 1 × 28 × 28
        num_classes = 10
    elif args.dataset.lower() == 'cifar10':
        dm = CIFAR10DataModule(data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            num_workers=num_workers)
        input_size = 32 * 32 * 3      # 3 × 32 × 32
        num_classes = 10
    elif args.dataset.lower() == 'cifar10_emb':
            dm = CIFAR10EmbeddingDataModule(
            feat_dir='/mnt/pdata/pr501/shortcut-models/dataset/cifar10_emb',
            batch_size=args.batch_size,
            num_workers=num_workers)
            input_size  = 512          # feature dim from ResNet-18
            num_classes = 10
    else:
        raise ValueError('dataset does not exist')


    ### -------- SELECT MODEL  --------------
    if args.model_type == 'mlp':
        model = LightMLP(input_size=input_size, num_classes=num_classes)
    elif args.model_type == 'lowrank':
        model = DynamicMLP(input_size=input_size, rank=args.train_rank, num_classes=num_classes, teacher_mode=args.teacher_mode, loss_variant=args.loss_variant)
    else:
        raise ValueError("Unknown model type")

    
    ### -------- SELECT DIFFERENT PARAMETERS -----------
    logger = WandbLogger(project="low-rank-performance",
                         name=args.model_name,
                         save_code=True)
    ckpt = ModelCheckpoint(dirpath=args.checkpoint_path,
                           every_n_epochs=args.save_every_n_epochs,
                           filename=f"{args.model_name}-{{epoch:03d}}",
                           save_top_k=-1)

    # Evaluate ranks
    rank_list = [64, 32, 16, 8, 4, 2, 1]
    ood_ranks = [3, 5, 10, 20, 30, 40, 50]
    #rank_list = rank_list + ood_ranks
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator="gpu",
                         devices=args.gpus,
                         callbacks=[ckpt, LearningRateMonitor(), RankEvalCallback(rank_list, ood_ranks)],
                         strategy = "ddp_find_unused_parameters_true",
                         precision='bf16-mixed',
                         logger=logger)
    # Fit the model
    trainer.fit(model, dm)

    # Get test performance
    trainer.test(model=model, datamodule=dm)

if __name__ == '__main__':
    args = parse_args()
    main(args)
