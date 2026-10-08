"""E5 (training-free part): FlexRank's DP allocation of per-layer internal widths on the PRUNING axis vs uniform widths.

For every ladder level k (residual width E_k fixed to the Elastoformer ladder value, i.e. the DepGraph-coupled residual stream is not searched):
  groups = 12 FFN widths F_l (units of the FFN hidden layer) + 12 attention widths Q_l (head-dim positions shared by all heads, so heads stay 12)
  ranking inside a group = group-L1 saliency (same criterion family as the ladder), so nesting along each group is by construction
  uniform  : every layer keeps the ladder's uniform width at level k  (same ranking)
  dp       : FlexRank DPSearchAlgo over the 24 groups (single-group probes at E_k), best profile with params <= the ladder level's total
  tp-ladder: the original torch-pruning ladder (SlicedViT level k) as an extra reference
Nothing is trained. The DP profiles of different levels are NOT constrained to be nested here (a prerequisite for training them jointly is
the nested chain that the same DP produces when run once; see plan section E5, deferred until E4a decides the training schedule).
"""
import os, sys, json, time, argparse, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy import deepcopy
import numpy as np
import torch

from models.elastoformer import ElasticViTForImageClassification, ElasticViTConfig
from models.sliced_vit import SlicedViT, build_level_indices
from utils.nested_metadata import run_nested_pruning, make_importance
from utils import flexrank_bridge as fb
from flexrank.layers.base import ODLayer
from flexrank.profiles.dp import DPSearchAlgo
from flexrank.profiles.utils import count_od_model_params
from eval_fronts import cached_tensors, top1

PSEUDO = 99


class GroupLayer(ODLayer):
    """A prunable width group exposed to FlexRank's DP as if it were a rank-ordered layer (inner_dim = number of kept units)."""

    def __init__(self, size, params_per_unit, on_change):
        super().__init__(size, size)
        self.params_per_unit, self._on_change = params_per_unit, on_change

    def get_num_parameters(self, inner_dim=None):
        return int(self.params_per_unit * (inner_dim or self._inner_dim))

    def _on_inner_dim_changed(self):
        self._on_change()


class GroupedViT(torch.nn.Module):
    """SlicedViT executed with per-layer widths: E fixed (ladder level), F_l / Q_l = top-s units of fixed importance rankings."""

    def __init__(self, sliced, rank_F, rank_Q, E_level, heads, head_dim):
        super().__init__()
        self.sliced, self.rank_F, self.rank_Q, self.E_level, self.heads, self.hd = sliced, rank_F, rank_Q, E_level, heads, head_dim
        nl = len(rank_F)
        E = sliced.idx("E", E_level)
        self.nl = nl
        self.gF = torch.nn.ModuleList([GroupLayer(len(rank_F[l]), 2 * len(E) + 1, self.refresh) for l in range(nl)])
        self.gQ = torch.nn.ModuleList([GroupLayer(head_dim, heads * (4 * len(E) + 3), self.refresh) for l in range(nl)])
        sliced.register_buffer(f"idx_E_{PSEUDO}", E, persistent=False)
        self.refresh()

    def keep_F(self, l, s):
        return torch.sort(self.rank_F[l][:s]).values

    def keep_Q(self, l, s):
        d = self.rank_Q[l][:s]                                        # head-dim positions kept in every head
        return torch.sort(torch.cat([h * self.hd + d for h in range(self.heads)])).values

    def refresh(self):
        dev = self.sliced.idx("E", self.E_level).device
        for l in range(self.nl):
            self.sliced.register_buffer(f"idx_F{l}_{PSEUDO}", self.keep_F(l, self.gF[l].inner_dim).to(dev), persistent=False)
            self.sliced.register_buffer(f"idx_Q{l}_{PSEUDO}", self.keep_Q(l, self.gQ[l].inner_dim).to(dev), persistent=False)

    def set_sizes(self, fs, qs):
        for l in range(self.nl):
            self.gF[l]._inner_dim, self.gQ[l]._inner_dim = int(fs[l]), int(qs[l])
        self.refresh()

    def forward(self, x):
        return types.SimpleNamespace(logits=self.sliced(x, PSEUDO))


def rankings(base, heads, hd):
    """Group-L1 saliency (sum |w| over every coupled weight/bias), descending order of units."""
    sd = base.state_dict(); nl = base.config.num_hidden_layers
    rank_F, rank_Q = [], []
    for l in range(nl):
        p = f"vit.encoder.layer.{l}."
        sF = sd[p + "intermediate.dense.weight"].abs().sum(1) + sd[p + "intermediate.dense.bias"].abs() + sd[p + "output.dense.weight"].abs().sum(0)
        rank_F.append(torch.argsort(sF, descending=True))
        sq = 0
        for t in ("query", "key", "value"):
            w, b = sd[p + f"attention.attention.{t}.weight"], sd[p + f"attention.attention.{t}.bias"]
            sq = sq + (w.abs().sum(1) + b.abs()).view(heads, hd)
        sq = sq + sd[p + "attention.output.dense.weight"].abs().sum(0).view(heads, hd)
        rank_Q.append(torch.argsort(sq.sum(0), descending=True))             # aggregate over heads -> one ranking of head-dim positions
    return rank_F, rank_Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="facebook/deit-base-patch16-224")
    ap.add_argument("--data_path", default="/var/scratch/dchabal/quokka/data/imagenet")
    ap.add_argument("--cache_dir", default="/var/scratch/skalra/elastoSLM/data_lists")
    ap.add_argument("--crs", nargs="+", type=float, default=[0.8, 0.6])
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--calib_n", type=int, default=4096)
    ap.add_argument("--probe_n", type=int, default=1024)
    ap.add_argument("--val_n", type=int, default=10000)
    ap.add_argument("--dp_cuts", type=int, default=12)
    ap.add_argument("--dp_min_p", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.dummy else "cpu")
    if a.dummy:
        cfg = ElasticViTConfig(hidden_size=96, num_hidden_layers=2, num_attention_heads=6, intermediate_size=192, image_size=224, patch_size=16, num_labels=10, qkv_bias=True)
        cfg.pruned_dim = 96
        base = ElasticViTForImageClassification(cfg).eval()
        calibx, caliby = torch.randn(a.calib_n, 3, 224, 224).half(), torch.randint(0, 10, (a.calib_n,))
        valx, valy = torch.randn(a.val_n, 3, 224, 224).half(), torch.randint(0, 10, (a.val_n,))
    else:
        import torchvision.transforms as T
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        base = ElasticViTForImageClassification.from_pretrained(a.model_name).eval()
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        calibx, caliby = cached_tensors(os.path.join(a.cache_dir, f"calib_{a.calib_n}_seed0"), os.path.join(a.data_path, "train"), a.calib_n, 0, 0, a.workers, tf)
        valx, valy = cached_tensors(os.path.join(a.cache_dir, f"val_{a.val_n}_seed0"), os.path.join(a.data_path, "val"), a.val_n, a.val_n // 1000, 0, a.workers, tf)
    probex, probey = calibx[:a.probe_n], caliby[:a.probe_n]
    heads = base.config.num_attention_heads
    sd = base.state_dict(); nl = base.config.num_hidden_layers
    Qfull = sd["vit.encoder.layer.0.attention.attention.query.weight"].shape[0]; hd = Qfull // heads
    rank_F, rank_Q = rankings(base, heads, hd)
    res = dict(args=vars(a), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", cr={})
    path = os.path.join(a.out, "results_e5.json")

    for cr in a.crs:
        core = deepcopy(base).cpu()
        meta = run_nested_pruning(core, deepcopy(base).cpu(), make_importance("l1"), torch.randn(1, 3, 224, 224), a.N, cr)
        groups, _ = build_level_indices(meta["pruned_index_out"], a.N, sd)
        sliced = SlicedViT(deepcopy(base), groups, nl, a.N).to(dev).eval()
        rows = []
        for L in range(1, a.N + 1):                                   # the full level has nothing to allocate
            t0 = time.time()
            E = len(sliced.idx("E", L))
            qs_uni = [len(sliced.idx(f"Q{l}", L)) // heads for l in range(nl)]
            fs_uni = [len(sliced.idx(f"F{l}", L)) for l in range(nl)]
            tot_uni = fb.vit_macs_params(E, [q * heads for q in qs_uni], fs_uni, classes=base.config.num_labels)
            gm = GroupedViT(sliced, [r.to(dev) for r in rank_F], [r.to(dev) for r in rank_Q], L, heads, hd).to(dev)
            ev = fb.CalibEvaluator(gm, probex, probey, dev, amp=not a.dummy)
            nongroup = tot_uni["params"] - sum(g.get_num_parameters(s) for g, s in zip(list(gm.gF) + list(gm.gQ), fs_uni + qs_uni))
            # uniform widths with the SAME ranking
            gm.set_sizes(fs_uni, qs_uni)
            acc_u, loss_u = top1(gm, valx, valy, dev, amp=not a.dummy)
            acc_tp, _ = top1(sliced, valx, valy, dev, level=L, amp=not a.dummy)
            # DP at fixed E_k
            gm.set_sizes([len(r) for r in rank_F], [hd] * nl)
            sol = DPSearchAlgo(evaluator=ev, n_models=a.dp_cuts, min_p=a.dp_min_p).solve()
            pdata = sol.to_profiles_data()
            budget = tot_uni["params"] - nongroup
            try:
                dims, got = pdata.get_profile_for_params(budget)
            except ValueError:
                dims, got = pdata.profiles[-1], pdata.params[-1]
            names = list(sol.layers_stats.keys())               # order == module traversal: gF.0..11 then gQ.0..11
            fs_dp, qs_dp = [int(d) for d in dims[:nl]], [int(d) for d in dims[nl:]]
            gm.set_sizes(fs_dp, qs_dp)
            acc_dp, loss_dp = top1(gm, valx, valy, dev, amp=not a.dummy)
            tot_dp = fb.vit_macs_params(E, [q * heads for q in qs_dp], fs_dp, classes=base.config.num_labels)
            rows.append(dict(level=L, E=E, params_budget=tot_uni["params"], params_dp=tot_dp["params"], macs_uniform=tot_uni["macs_modules"], macs_dp=tot_dp["macs_modules"],
                             top1_uniform_samerank=acc_u, top1_dp=acc_dp, top1_tp_ladder=acc_tp, loss_uniform=loss_u, loss_dp=loss_dp,
                             fs_uniform=fs_uni, qs_uniform=qs_uni, fs_dp=fs_dp, qs_dp=qs_dp, wall_s=time.time() - t0))
            print(f"[CR={cr}] L{L} E={E} budget={tot_uni['params'] / 1e6:.2f}M dp={tot_dp['params'] / 1e6:.2f}M | top1 tp-ladder {acc_tp:.2f}  uniform(same rank) {acc_u:.2f}  DP {acc_dp:.2f}", flush=True)
            res["cr"][str(cr)] = dict(levels=rows)
            json.dump(res, open(path, "w"), indent=1, default=str)
            del gm
    print("E5 (training-free) DONE ->", path)


if __name__ == "__main__":
    main()
