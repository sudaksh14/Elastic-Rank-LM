import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import math
import random

from lowrank.models.layers import DynamicLowRankLinear


class LightMLP(pl.LightningModule):
    """Standard (full-rank) MLP baseline."""

    def __init__(self, input_size=28*28, hidden_dim=256, lr: float = 1e-3, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)

    def test_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
        return {"optimizer": opt, "lr_scheduler": sched}


class DynamicMLP(pl.LightningModule):
    """
    An MLP that uses DynamicLowRankLinear layers and trains using a multi-task
    objective with uncertainty-weighted losses for different rank configurations.
    """
    def __init__(self,
        input_size=28*28,
        hidden_dim=256,
        lr: float = 1e-3,
        num_classes=10,
        teacher_mode="max",
        loss_variant='ce_with_consistency',
        rank=64,
        train_ranks=None):

        super().__init__()
        self.save_hyperparameters()

        if self.hparams.train_ranks is None:
            self.hparams.train_ranks = [
                max(1, self.hparams.rank // (2**k))
                for k in range(int(math.log2(self.hparams.rank)) + 1)
            ]
            self.hparams.train_ranks.sort(reverse=True)

        self.net = nn.Sequential(
            nn.Flatten(),
            DynamicLowRankLinear(self.hparams.input_size, self.hparams.hidden_dim, self.hparams.rank),
            nn.ReLU(),
            DynamicLowRankLinear(self.hparams.hidden_dim, self.hparams.num_classes, self.hparams.rank),
        )

        self.log_vars_ce = nn.ParameterDict({
            str(r): nn.Parameter(torch.zeros(())) for r in self.hparams.train_ranks
        })

        self.log_var_cons  = nn.Parameter(torch.zeros(()))
        self.log_var_feat  = nn.Parameter(torch.zeros(()))
        self.log_var_ortho = nn.Parameter(torch.zeros(()))

    def _set_rank_all(self, r):
        """Set the rank for all DynamicLowRankLinear layers in the network."""
        for m in self.modules():
            if isinstance(m, DynamicLowRankLinear):
                m.set_rank(r)

    def forward_with_hidden(self, x):
        """Forward pass returning both logits and intermediate low-rank representations."""
        x = self.net[0](x)  # Flatten
        out1, h1 = self.net[1].forward_with_hidden(x)
        out1_relu = self.net[2](out1)
        out2, h2 = self.net[3].forward_with_hidden(out1_relu)
        return out2, [h1, h2]

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.hparams.teacher_mode == 'max':
            r_anchor = self.hparams.rank
            possible_variants = [r for r in self.hparams.train_ranks if r < r_anchor]
            r_variant = random.choice(possible_variants)
        elif self.hparams.teacher_mode == 'progressive':
            possible_anchors = [r for r in self.hparams.train_ranks if r > 1]
            r_anchor = random.choice(possible_anchors)
            r_variant = max(1, r_anchor // 2)
        elif self.hparams.teacher_mode == 'none':
            self._set_rank_all(self.hparams.rank)
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            acc = (logits.argmax(1) == y).float().mean()
            self.log_dict({"train/loss": loss, "train/acc": acc},
                          on_step=True, on_epoch=True)
            return loss
        else:
            raise ValueError("Invalid teacher_mode specified")

        self._set_rank_all(r_anchor)
        logits_anchor, hiddens_anchor = self.forward_with_hidden(x)
        ce_anchor = F.cross_entropy(logits_anchor, y)

        self._set_rank_all(r_variant)
        logits_variant, hiddens_variant = self.forward_with_hidden(x)
        ce_variant = F.cross_entropy(logits_variant, y)

        self._set_rank_all(self.hparams.rank)

        log_var_a = self.log_vars_ce[str(r_anchor)]
        log_var_v = self.log_vars_ce[str(r_variant)]

        loss = (torch.exp(-log_var_a) * ce_anchor + log_var_a) + \
               (torch.exp(-log_var_v) * ce_variant + log_var_v)

        acc = (logits_anchor.argmax(1) == y).float().mean()

        log_data = {
            "train/loss": loss,
            f"train/ce_rank_{r_anchor}": ce_anchor,
            f"train/ce_rank_{r_variant}": ce_variant,
            "train/acc": acc,
            "r_anchor": float(r_anchor),
            "r_variant": float(r_variant),
            f"s_rank_{r_anchor}": log_var_a,
            f"s_rank_{r_variant}": log_var_v,
        }

        self.log_dict(log_data, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self._set_rank_all(self.hparams.rank)
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._set_rank_all(self.hparams.rank)
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"test/acc_rank_{self.hparams.rank}", acc, prog_bar=True)
        self.log(f"test/loss_rank_{self.hparams.rank}", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
