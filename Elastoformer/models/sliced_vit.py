"""Virtual-slicing ViT: ONE full-width parameter set that can execute every Elastoformer level.

Level L in 1..N+1 (Level_1 = core, Level_{N+1} = full model). Level L keeps, in every coupled dimension group, all indices
except the shells removed in pruning rounds 0..(N-L) (see utils/nested_metadata.py). Because the shells are disjoint and
nested by construction (E0: 0 violations) every level is a subset of the next one, so joint training of all levels shares
every weight and nothing has to be copied or frozen.

Dimension groups (verified in E0 to share indices):
  E      residual width: patch-proj out, cls_token, pos_emb, LN weights/bias, q/k/v in, attn-out out, fc1 in, fc2 out, classifier in
  Q[l]   per-layer attention width: q/k/v out, attn-out in.  head count stays 12 at every level, head_size = |Q[l]_L| / heads
  F[l]   per-layer FFN width: fc1 out, fc2 in
Equivalent to the physically pruned/rebuilt model of that level (tests/test_sliced_equivalence.py).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PFX = "vit.encoder.layer"


def build_level_indices(pruned_out, steps, sd):
    """pruned_out: list (per round) of {layer_name: [idx..]} from run_nested_pruning. sd: full model state dict.
    Returns dict group-> {level: LongTensor(sorted keep idx)} with groups 'E', ('Q', l), ('F', l)."""
    n_layers = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith(PFX))

    def keep_sets(name):
        tot = sd[f"{name}.weight"].shape[0]
        out = {}
        for L in range(1, steps + 2):
            removed = set()
            for r in range(0, steps + 1 - L):  # rounds 0..N-L removed
                removed |= set(pruned_out[r].get(name, []))
            out[L] = torch.tensor(sorted(set(range(tot)) - removed), dtype=torch.long)
        return out

    groups = {"E": keep_sets("vit.embeddings.patch_embeddings.projection")}
    for l in range(n_layers):
        qs = [keep_sets(f"{PFX}.{l}.attention.attention.{t}") for t in ("query", "key", "value")]
        for L in qs[0]:
            assert torch.equal(qs[0][L], qs[1][L]) and torch.equal(qs[0][L], qs[2][L]), f"q/k/v index mismatch layer {l} level {L}"
        groups[("Q", l)] = qs[0]
        groups[("F", l)] = keep_sets(f"{PFX}.{l}.intermediate.dense")
    # consistency of the coupled E-group members
    for nm in ("vit.layernorm", f"{PFX}.0.layernorm_before", f"{PFX}.0.layernorm_after", f"{PFX}.0.attention.output.dense", f"{PFX}.0.output.dense"):
        ks = keep_sets(nm)
        for L in ks:
            assert torch.equal(ks[L], groups["E"][L]), f"E-group mismatch at {nm} level {L}"
    return groups, n_layers


class SlicedViT(nn.Module):
    """Wraps a full-width ElasticViTForImageClassification and runs any level functionally (no weight copies)."""

    def __init__(self, base, groups, n_layers, steps):
        super().__init__()
        self.base = base
        self.steps = steps
        self.n_layers = n_layers
        self.heads = base.config.num_attention_heads
        self.eps = base.config.layer_norm_eps
        self.levels = list(range(1, steps + 2))
        for key, per_level in groups.items():
            tag = key if isinstance(key, str) else f"{key[0]}{key[1]}"
            for L, idx in per_level.items():
                self.register_buffer(f"idx_{tag}_{L}", idx, persistent=False)

    def idx(self, tag, L):
        return getattr(self, f"idx_{tag}_{L}")

    def full_level(self):
        return self.steps + 1

    @staticmethod
    def _w(w, o, i=None):
        w = w if o is None else w.index_select(0, o)
        return w if i is None else w.index_select(1, i)

    def forward(self, x, level=None):
        L = self.full_level() if level is None else level
        full = L == self.full_level()
        b = self.base
        e = None if full else self.idx("E", L)

        proj = b.vit.embeddings.patch_embeddings.projection
        pw, pb = (proj.weight, proj.bias) if full else (proj.weight.index_select(0, e), proj.bias.index_select(0, e))
        h = F.conv2d(x.to(pw.dtype), pw, pb, stride=proj.stride).flatten(2).transpose(1, 2)
        cls = b.vit.embeddings.cls_token if full else b.vit.embeddings.cls_token.index_select(2, e)
        pos = b.vit.embeddings.position_embeddings if full else b.vit.embeddings.position_embeddings.index_select(2, e)
        h = torch.cat([cls.expand(h.shape[0], -1, -1), h], dim=1) + pos
        E = h.shape[-1]

        def ln(mod, t):
            w, bb = (mod.weight, mod.bias) if full else (mod.weight.index_select(0, e), mod.bias.index_select(0, e))
            return F.layer_norm(t, (E,), w, bb, self.eps)

        for l, blk in enumerate(b.vit.encoder.layer):
            q_idx = None if full else self.idx(f"Q{l}", L)
            f_idx = None if full else self.idx(f"F{l}", L)
            a = blk.attention.attention
            y = ln(blk.layernorm_before, h)
            qkv = [F.linear(y, self._w(m.weight, q_idx, e), None if m.bias is None else (m.bias if full else m.bias.index_select(0, q_idx)))
                   for m in (a.query, a.key, a.value)]
            Wq = qkv[0].shape[-1]
            hs = Wq // self.heads
            q, k, v = (t.view(t.shape[0], t.shape[1], self.heads, hs).transpose(1, 2) for t in qkv)
            ctx = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hs))
            ctx = ctx.transpose(1, 2).reshape(h.shape[0], h.shape[1], Wq)
            o = blk.attention.output.dense
            h = h + F.linear(ctx, self._w(o.weight, e, q_idx), o.bias if full else o.bias.index_select(0, e))
            y = ln(blk.layernorm_after, h)
            f1, f2 = blk.intermediate.dense, blk.output.dense
            y = F.gelu(F.linear(y, self._w(f1.weight, f_idx, e), f1.bias if full else f1.bias.index_select(0, f_idx)))
            h = h + F.linear(y, self._w(f2.weight, e, f_idx), f2.bias if full else f2.bias.index_select(0, e))

        h = ln(b.vit.layernorm, h)
        c = b.classifier
        return F.linear(h[:, 0], c.weight if full else c.weight.index_select(1, e), c.bias)

    @torch.no_grad()
    def level_stats(self, ex):
        """params / MACs of every level via a throw-away dense count (for logging only)."""
        out = {}
        for L in self.levels:
            n = 0
            full = L == self.full_level()
            E = self.base.config.hidden_size if full else len(self.idx("E", L))
            n += 3 * 16 * 16 * E + E + (1 + 196 + 1) * E  # patch proj + bias + cls + pos
            for l in range(self.n_layers):
                Q = self.base.vit.encoder.layer[l].attention.attention.query.weight.shape[0] if full else len(self.idx(f"Q{l}", L))
                Fd = self.base.vit.encoder.layer[l].intermediate.dense.weight.shape[0] if full else len(self.idx(f"F{l}", L))
                n += 4 * E * Q + 3 * Q + E + 2 * E * Fd + Fd + E + 4 * E
            n += 2 * E + E * self.base.classifier.out_features + self.base.classifier.out_features
            out[L] = n
        return out


def slice_state_dict(sd, sliced, L):
    idx_fn = sliced.idx if hasattr(sliced, 'idx') else sliced   # SlicedViT or callable (tag, L) -> LongTensor
    """State dict of the physically-sized Level-L model, sliced from the full weights (ALL blocks, incl. off-diagonal ones).
    Used (a) as an independent reference for the equivalence test, (b) to initialise 'corrected' frozen-prefix levels."""
    out = {}
    e = idx_fn("E", L)
    for k, v in sd.items():
        if k.startswith("vit.embeddings.cls_token") or k.startswith("vit.embeddings.position_embeddings"):
            out[k] = v.index_select(2, e)
        elif k.startswith("vit.embeddings.patch_embeddings.projection"):
            out[k] = v.index_select(0, e)
        elif k.startswith("vit.layernorm"):
            out[k] = v.index_select(0, e)
        elif k.startswith("classifier"):
            out[k] = v.index_select(1, e) if k.endswith("weight") else v
        else:
            l = int(k.split(".")[3]); q, f = idx_fn(f"Q{l}", L), idx_fn(f"F{l}", L)
            w = k.endswith("weight")
            if ".layernorm_" in k or k.endswith("output.dense.bias"):   # LN params and both output-dense biases live on E
                out[k] = v.index_select(0, e)
            elif "attention.attention." in k:
                out[k] = v.index_select(0, q).index_select(1, e) if w else v.index_select(0, q)
            elif "attention.output.dense" in k:
                out[k] = v.index_select(0, e).index_select(1, q)
            elif "intermediate.dense" in k:
                out[k] = v.index_select(0, f).index_select(1, e) if w else v.index_select(0, f)
            elif "output.dense" in k:
                out[k] = v.index_select(0, e).index_select(1, f)
            else:
                raise KeyError(k)
    return out
