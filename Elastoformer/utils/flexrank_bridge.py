"""Bridge between Elastoformer's HF ElasticViT and the standalone `flexrank` core (rank axis).

* collect_grams        : per-Linear input Gram matrices E[x x^T] on a calibration tensor (what DataSVD consumes)
* to_rank_model        : replace the 72 encoder Linears (q,k,v,o,fc1,fc2 x 12) by ODLinear via flexrank.decompose_linear,
                         optionally capping the stored rank to r_cap_frac * min(in,out) (plan section 1.3 "capped r_max")
* CalibEvaluator       : minimal flexrank `AbstractEvaluator` (model, distr, evaluate()) over a cached calibration tensor
* vit_macs_params      : analytic params / MACs of a (pruned or rank) ViT-B family model, incl. the paper's FLOPs convention
Patch embedding, cls/pos, LayerNorms and the classifier are never decomposed (residual width untouched).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flexrank.layers.decomposition import decompose_linear
from flexrank.layers.linear import ODLinear
from flexrank.layers.base import ODLayer
from flexrank.types import get_single_worker_distributed_info

ENC_LINEARS = ("attention.attention.query", "attention.attention.key", "attention.attention.value",
               "attention.output.dense", "intermediate.dense", "output.dense")


def encoder_linears(model):
    """[(name, nn.Linear)] of the encoder projections, in module-traversal order."""
    out = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear) and n.startswith("vit.encoder.layer.") and n.endswith(ENC_LINEARS):
            out.append((n, m))
    return out


@torch.no_grad()
def collect_grams(model, calib_x, device, batch_size=128):
    """Mean input Gram (in x in, fp32) of every encoder Linear over calib_x [n,3,H,W]."""
    layers = encoder_linears(model)
    grams = {n: torch.zeros(m.in_features, m.in_features, device=device, dtype=torch.float32) for n, m in layers}
    rows = {n: 0 for n, _ in layers}
    hooks = []

    def mk(name):
        def hook(_m, inp):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            grams[name] += x.T @ x
            rows[name] += x.shape[0]
        return hook

    for n, m in layers:
        hooks.append(m.register_forward_pre_hook(mk(n)))
    model.eval().to(device)
    for i in range(0, calib_x.shape[0], batch_size):
        model(calib_x[i:i + batch_size].to(device))
    for h in hooks:
        h.remove()
    return {n: grams[n] / rows[n] for n in grams}


def _set_module(model, name, new):
    parent = model
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


@torch.no_grad()
def to_rank_model(base, grams=None, svd_type="datasvd", r_cap_frac=1.0, device="cpu", dtype=torch.float32):
    """Deep copy of `base` with encoder Linears replaced by ODLinear (nested rank prefixes). grams=None -> plain SVD."""
    from copy import deepcopy
    model = deepcopy(base).to(device).eval()
    for name, lin in encoder_linears(model):
        inputs = None if (svd_type == "svd" or grams is None) else grams[name]
        od = decompose_linear(lin, inputs=inputs, device=device, dtype=dtype)
        r = od.max_inner_dim
        cap = max(1, int(math.ceil(r_cap_frac * r)))
        if cap < r:   # capped storage: keep only the leading `cap` rank components
            od = ODLinear.build_from_uv(od.weight_u[:cap].detach().clone(), od.weight_v[:, :cap].detach().clone(),
                                        None if od.bias is None else od.bias.detach().clone(), device=device, dtype=dtype)
        _set_module(model, name, od)
    return model


class CalibEvaluator:
    """flexrank AbstractEvaluator over a cached calibration tensor; evaluate() -> {'eval_loss': mean CE}."""

    def __init__(self, model, x, y, device, amp=True, batch_size=256):
        self.model, self.x, self.y, self.device, self.amp, self.bs = model, x, y, device, amp, batch_size
        self._distr = get_single_worker_distributed_info(device=torch.device(device))
        self.compute_metrics = None

    @property
    def distr(self):
        return self._distr

    @torch.no_grad()
    def evaluate(self, *a, **k):
        self.model.eval()
        tot, n, c1 = 0.0, 0, 0
        for i in range(0, self.x.shape[0], self.bs):
            xb, yb = self.x[i:i + self.bs].to(self.device, non_blocking=True).float(), self.y[i:i + self.bs].to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp and str(self.device) != "cpu"):
                out = self.model(xb).logits.float()
            tot += F.cross_entropy(out, yb, reduction="sum").item(); n += yb.numel(); c1 += (out.argmax(1) == yb).sum().item()
        return {"eval_loss": tot / n, "eval_top1": 100.0 * c1 / n}


def vit_macs_params(E, Qs, Fs, tokens=197, patch_in=768, classes=1000, od=None):
    """Analytic cost of a ViT whose layer l has attention width Qs[l], FFN width Fs[l] and residual width E.
    od: optional list (len 6*L, order q,k,v,o,fc1,fc2 per layer) of (in, out, r, gar) for rank layers; replaces the dense cost.
    Returns dict(params, params_stored, macs_modules, macs_total). macs_modules is the paper's FLOPs convention
    (module MACs only; verified: DeiT-B = 16.86 G); macs_total adds the two attention matmuls."""
    L = len(Qs)
    non_lin = patch_in * E + E + E + (tokens) * E + 2 * E + classes * E + classes          # patch proj, cls, pos, final LN, classifier
    params = non_lin + L * 4 * E                                                          # 2 LayerNorms per layer
    stored = params
    macs = tokens * 0 + (tokens - 1) * patch_in * E + classes * E
    attn = 0
    for l in range(L):
        Q, Fd = Qs[l], Fs[l]
        attn += 2 * tokens * tokens * Q
        if od is None:
            lin = [(E, Q), (E, Q), (E, Q), (Q, E), (E, Fd), (Fd, E)]
            for i, o in lin:
                params += i * o + o; stored += i * o + o; macs += tokens * i * o
        else:
            for (i, o, r, gar) in od[6 * l: 6 * l + 6]:
                p = (i + o) * r - (r * r if gar else 0)
                params += p + o; stored += (i + o) * r + o; macs += tokens * p
    return dict(params=params, params_stored=stored, macs_modules=macs, macs_total=macs + tokens * 0 + attn)


def od_cost_list(model):
    """[(in,out,r_active,gar_flag)] for the model's ODLinears in traversal order, using the active inner_dim."""
    return [(m.in_features, m.out_features, m.inner_dim, True) for m in model.modules() if isinstance(m, ODLayer)]


def count_total_params_rank(model):
    """Parameters of a rank model under FlexRank's accounting (GAR) for ODLayers + dense for everything else."""
    od = sum(m.get_num_parameters() for m in model.modules() if isinstance(m, ODLayer))
    od_p = {id(p) for m in model.modules() if isinstance(m, ODLayer) for p in m.parameters()}
    rest = sum(p.numel() for p in model.parameters() if id(p) not in od_p)
    return od + rest


def set_profile(model, inner_dims):
    layers = [m for m in model.modules() if isinstance(m, ODLayer)]
    assert len(layers) == len(inner_dims), (len(layers), len(inner_dims))
    for m, r in zip(layers, inner_dims):
        m.inner_dim = int(r)


def full_profile(model):
    return [m.max_inner_dim for m in model.modules() if isinstance(m, ODLayer)]


def rank_corr(a, b):
    """Spearman rho and Kendall tau-b (numpy only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    n, con, dis, ta, tb = len(a), 0, 0, 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0: ta += 1
            if db == 0: tb += 1
            if da * db > 0: con += 1
            elif da * db < 0: dis += 1
    denom = math.sqrt((con + dis + ta) * (con + dis + tb)) or 1.0
    return rho, float((con - dis) / denom)
