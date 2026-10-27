import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class RankEvalCallback(pl.Callback):
    def __init__(self, ranks, ood_ranks):

        super().__init__()
        self.ranks = ranks
        self.ood_ranks = ood_ranks

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """This hook is called after the normal validation finishes"""
        device = pl_module.device
        val_loader = trainer.datamodule.val_dataloader()

        accuracies = {}
        for r in (*self.ranks, *self.ood_ranks):
            pl_module._set_rank_all(r)

            total_correct = 0
            total_seen = 0
            total_loss = 0.0

            with torch.no_grad():
                for x,y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = pl_module(x)
                    loss = F.cross_entropy(logits, y, reduction='sum')

                    total_loss += loss.item()
                    total_correct += (logits.argmax(1) == y).sum().item()
                    total_seen += y.size(0)

            # Compute metrics
            epoch_acc = total_correct / total_seen
            epoch_loss = total_loss / total_seen
            accuracies[r] = epoch_acc
            
            # Log
            trainer.logger.experiment.log({
                f"val/r{r}_acc": epoch_acc,
                f"val/r{r}_loss": epoch_loss,
                "epoch": trainer.current_epoch
            })

        # Restore rank
        pl_module._set_rank_all(pl_module.hparams.rank)

        # --- compute and log averages ---
        avg_acc = sum(accuracies[r] for r in self.ranks) / len(self.ranks)
        avg_acc_ood = sum(accuracies[r] for r in self.ood_ranks) / len(self.ood_ranks)

        trainer.logger.experiment.log(
            {
                'val/average_rank_accuracy': avg_acc,
                'val/average_ood_rank_accuracy': avg_acc_ood,
                'epoch': trainer.current_epoch
            }
        )