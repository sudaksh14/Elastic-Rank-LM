import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import math
import random

from lowrank.models.layers import DynamicLowRankLinear


class DynamicMLP(pl.LightningModule):
    def __init__(self,
        input_size=28*28, 
        hidden_dim=256,
        lr: float=1e-3, 
        num_classes=10, 
        teacher_mode= "max",
        loss_variant='ce_with_consistency',  # or ce_no_consistency 
        rank=64,
        train_ranks=[max(1, 64 // (2**k)) for k in range(int(math.log2(64))+1)]):

        super().__init__()
        self.save_hyperparameters() # stores in self.hparams
        
        self.net = nn.Sequential(
            nn.Flatten(),
            DynamicLowRankLinear(input_size, hidden_dim, rank), # <-- train rank
            nn.ReLU(),
            DynamicLowRankLinear(hidden_dim, num_classes, rank), # <-- train rank

        )

        # --- log-variances (initialised to log(1) = 0)
        self.train_ranks = train_ranks
        
        self.log_var_cls   = nn.Parameter(torch.zeros(()))
        self.log_var_stu   = nn.Parameter(torch.zeros(()))
        self.log_var_cons  = nn.Parameter(torch.zeros(()))
        self.log_var_feat  = nn.Parameter(torch.zeros(()))
        self.log_var_ortho = nn.Parameter(torch.zeros(()))

        if self.hparams.loss_variant == 'ce_with_consistency':
            self.log_var_cons = nn.Parameter(torch.zeros(()))
        elif self.hparams.loss_variant == 'ce_with_feature_consistency':
            self.log_var_feat = nn.Parameter(torch.zeros(()))



    def _set_rank_all(self, r):
        """Set the rank for all layers"""
        for m in self.modules():
            if hasattr(m, 'set_rank'):
                m.set_rank(r)

    def forward_with_hidden(self, x):
        x = x.view(x.size(0), -1) # Manually flatten

        # First layer
        out1, h1 = self.net[1].forward_with_hidden(x) # Dynamic Low Rank Layer
        out1 = F.relu(out1)

        # Second layer
        out2, h2 = self.net[3].forward_with_hidden(out1)
        return out2, [h1, h2]
                
    def forward(self, x):
        return self.net(x)

    def training_step(self, batch):
        x, y = batch 

        # Ladder wil all ranks
        # ----- forward pass @ rank r_train -------
        self._set_rank_all(self.hparams.rank)

        # ----- pick teacher & student ranks -----
        if self.hparams.teacher_mode == 'max':
            r_t = self.hparams.rank
            r_s = random.choice([r for r in self.train_ranks if r < self.hparams.rank])
        elif self.hparams.teacher_mode == 'progressive':
            r_t = random.choice(self.train_ranks[:-1]) # Exclude rank 1
            r_s = max(1, r_t // 2)

        else:
            raise ValueError("Invalid teacher mode")

        # ----- teacher forward pass
        self._set_rank_all(r_t)
        logits_t, hiddens_t = self.forward_with_hidden(x)
        ce_t = F.cross_entropy(logits_t, y)

        # student forward pass
        self._set_rank_all(r_s)
        logits_s, hiddens_s = self.forward_with_hidden(x)
        ce_s = F.cross_entropy(logits_s, y)

        self._set_rank_all(self.hparams.rank)

        if self.hparams.loss_variant == 'ce_no_consistency':

            loss = (torch.exp(-self.log_var_cls) * ce_t + self.log_var_cls +
                    torch.exp(-self.log_var_stu) * ce_s + self.log_var_stu)

            acc = (logits_s.argmax(1) == y).float().mean()

            self.log_dict({
            'train/loss'      : loss,
            'train/ce_teacher': ce_t,
            'train/ce_student': ce_s,
            'train/acc'       : acc,
            'r_teacher'       : r_t,
            'r_student'       : r_s,
            's_cls'           : self.log_var_cls,
            's_stu'           : self.log_var_stu,
             }, on_step=True, on_epoch=True)

            return loss

        elif self.hparams.loss_variant == 'ce_with_consistency':
            ## --- Variant B: Adding consistency
            cons = F.mse_loss(logits_s, logits_t.detach())
            acc = (logits_t.argmax(1) == y).float().mean()       # report teacher acc
            
            loss = (torch.exp(-self.log_var_cls)  * ce_t + self.log_var_cls   +
                        torch.exp(-self.log_var_stu)  * ce_s + self.log_var_stu   +
                        0.5 * torch.exp(-self.log_var_cons) * cons + 0.5 * self.log_var_cons)

            self._set_rank_all(self.hparams.rank)
            self.log_dict({
                'train/loss'      : loss,
                'train/ce_teacher': ce_t,
                'train/ce_student': ce_s,
                'train/cons'      : cons,
                'train/acc'       : acc,
                'r_teacher'       : r_t,
                'r_student'       : r_s,
                's_cls'           : self.log_var_cls,
                's_stu'           : self.log_var_stu,
                's_cons'          : self.log_var_cons,
            }, on_step=True, on_epoch=True)
            return loss

        elif self.hparams.loss_variant == 'ce_with_feature_consistency':
            # feature level MSE across *all* hidden layers
            feat_loss = torch.tensor(0.0, device=x.device)
            for h_t, h_s in zip(hiddens_t, hiddens_s):
                h_s_n = F.normalize(h_s, dim=1)
                h_t_n = F.normalize(h_t[:, :h_s.size(1)], dim=1)
                feat_loss += F.mse_loss(h_s_n, h_t_n)
            
            acc = (logits_t.argmax(1) == y).float().mean()       # report teacher acc
            
            loss = (torch.exp(-self.log_var_cls)  * ce_t + self.log_var_cls   +
                        torch.exp(-self.log_var_stu)  * ce_s + self.log_var_stu   +
                        0.5 * torch.exp(-self.log_var_feat) * feat_loss + 0.5 * self.log_var_feat)
            
            self.log_dict({
                "train/loss": loss,
                "train/ce_teacher": ce_t,
                "train/ce_student": ce_s,
                "train/feat_cons": feat_loss,
                "train/acc": acc,
                "r_teacher": r_t,
                "r_student": r_s,
                "s_cls": self.log_var_cls,
                "s_stu": self.log_var_stu,
                "s_feat": self.log_var_feat,
            }, on_step=True, on_epoch=True)

            return loss
        elif self.hparams.loss_variant == 'ce_orthogonality':
            ortho_penalty = torch.tensor(0.0, device=x.device)
            for m in self.modules():
                if isinstance(m, DynamicLowRankLinear) and r_t > r_s:
                    
                    # Normalize instead
                    #A = m.A.weight
                    #A_s = A[:r_s, :] # (rank1, in_features)
                    #A_res = A[r_s:r_t, :].detach() # (rank2, in_features)
                    #A_s_n   = F.normalize(A_s, dim=1, eps=1e-8)
                    #A_res_n = F.normalize(A_res, dim=1, eps=1e-8)
                    #pen     = (A_s_n @ A_res_n.T).pow(2).sum()

                    A = m.A.weight # (rank, in_features)
                    A_s = A[:r_s, :] # (rank1, in_features)
                    A_res = A[r_s:r_t, :] # (rank2, in_features)
                    ortho_penalty += torch.norm(A_s @ A_res.T, p='fro') ** 2 # (rank1, infeatures) x (in_features, rank2) = (rank1, rank2)
                    
            loss = (
                torch.exp(-self.log_var_cls)   * ce_t          + self.log_var_cls   +
                torch.exp(-self.log_var_stu)   * ce_s          + self.log_var_stu   +
                torch.exp(-self.log_var_ortho) * ortho_penalty + self.log_var_ortho
            )

            self.log_dict(
                {"train/loss": loss,
                 "train/ce_teacher": ce_t,
                 "train/ce_student": ce_s,
                 "train/ortho_penalty": ortho_penalty,
                 "train/acc": (logits_t.argmax(1) == y).float().mean(),
                 "r_teacher": r_t, "r_student": r_s},
                on_step=True, on_epoch=True)
            return loss

        elif self.hparams.loss_variant == 'ce_orthogonality_hard':
            ortho_penalty = torch.tensor(0.0, device=x.device)
            for m in self.modules():
                if isinstance(m, DynamicLowRankLinear):
                    
                    # Why do we regularize A?: The rows of A are the basis vectors; making them orthonormal means each direction contributes uniquely and non-redundantly to the representation.
                    A = m.A.weight # full basis matrix (max_rank, in features)
                    # NOrmalize the rows to focus on the angle, not magnitude
                    A_n = F.normalize(A, p=2, dim=1, eps=1e-8)

                    # Calculate deviation from identity matrix
                    identity = torch.eye(A_n.size(0), device=A_n.device)
                    pen = torch.norm(A_n @ A_n.T - identity, p='fro')**2
                    ortho_penalty += pen

                  
            loss = (
                torch.exp(-self.log_var_cls)   * ce_t          + self.log_var_cls   +
                torch.exp(-self.log_var_stu)   * ce_s          + self.log_var_stu   +
                torch.exp(-self.log_var_ortho) * ortho_penalty + self.log_var_ortho
            )

            self.log_dict(
                {"train/loss": loss,
                 "train/ce_teacher": ce_t,
                 "train/ce_student": ce_s,
                 "train/ortho_penalty": ortho_penalty,
                 "train/acc": (logits_t.argmax(1) == y).float().mean(),
                 "r_teacher": r_t, "r_student": r_s,},
                on_step=True, on_epoch=True)
            return loss
        
        elif self.hparams.loss_variant == 'ce_orthogonality_hard_no_stud':
            ortho_penalty = torch.tensor(0.0, device=x.device)
            for m in self.modules():
                if isinstance(m, DynamicLowRankLinear):
                    
                    # Why do we regularize A?: The rows of A are the basis vectors; making them orthonormal means each direction contributes uniquely and non-redundantly to the representation.
                    A = m.A.weight # full basis matrix (max_rank, in features)
                    # NOrmalize the rows to focus on the angle, not magnitude
                    A_n = F.normalize(A, p=2, dim=1, eps=1e-8)

                    # Calculate deviation from identity matrix
                    identity = torch.eye(A_n.size(0), device=A_n.device)
                    pen = torch.norm(A_n @ A_n.T - identity, p='fro')**2
                    ortho_penalty += pen

                  
            loss = (
                torch.exp(-self.log_var_cls)   * ce_t          + self.log_var_cls   +
                torch.exp(-self.log_var_ortho) * ortho_penalty + self.log_var_ortho
            )

            self.log_dict(
                {"train/loss": loss,
                 "train/ce_teacher": ce_t,
                 "train/ortho_penalty": ortho_penalty,
                 "train/acc": (logits_t.argmax(1) == y).float().mean(),
                 "r_teacher": r_t, "r_student": r_s,},
                on_step=True, on_epoch=True)
            return loss

        elif self.hparams.loss_variant == 'ce_teacher_only':
            loss = ce_t
            acc = (logits_t.argmax(1) == y).float().mean()

            self.log_dict({
                'train/loss': loss,
                'train/ce_teacher': ce_t,
                'train/acc': acc,
                'r_teacher': r_t,
                'r_student': r_s, # Logged for consistency, though not used in loss
            }, on_step=True, on_epoch=True)

            return loss


        else:
            raise ValueError("N/A")
        
    
    @torch.no_grad()
    def validation_step(self, batch):
        self._set_rank_all(self.hparams.rank)
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)

    def test_step(self, batch):
        self._set_rank_all(self.hparams.rank)
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

