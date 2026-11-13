#!/usr/bin/env python
"""S0 / Phase 0 spectral diagnostics (no training). See spectral/implementation_plan.md.

For every linear layer of a causal LM, compare ways of storing the weight W (m x n) at the
same bit budget, and report two relative errors:

  err_plain = ||W - W_hat||_F^2 / ||W||_F^2
  err_gram  = ||(W - W_hat) S||_F^2 / ||W S||_F^2,  S = Sigma^(1/2), Sigma = input Gram matrix

Bases (coefficients C = T_out W T_in^T, T orthonormal):
  spatial    identity (the neuron basis)
  dct        orthonormal DCT-II on both axes, neurons in their stored order
  dct_perm   DCT after seriation: neurons re-ordered by the Fiedler vector of |correlation|
  haar_perm  unbalanced Haar wavelets after the same seriation
  klt        eigenvectors of the input Gram (in) and of the output Gram W Sigma W^T (out)

Methods per basis:
  block          leading p x q block, p/q = m/n (no index storage)
  geom           LYRA geometric index spacing per axis (dct only)
  topk           largest |C|                      (16 value bits + index bits per coefficient)
  topk_data      largest C^2 * diag(Sigma')       (diagonal Gram weighting, same bits)
  rankband       rank-r approximation of the leading (kappa m) x (kappa n) block, GAR count
                 r (k_i + k_o - r); kappa = 1 in the spatial basis is plain SVD
  rankband_data  same, block columns weighted by sqrt(diag(Sigma')); in the klt basis with
                 kappa = 1 this is exactly DataSVD
Null: a Gaussian matrix of the same shape (svd, dct block, dct topk), plain error only.

The per-layer alignment (dct_perm, haar_perm, klt) is an upper bound for what a consistent
network-wide alignment can give: here each layer gets its own permutation / rotation.

Python 3.9 compatible. Needs torch; transformers + datasets unless --selftest.
"""
import argparse
import json
import math
import os
import re
import time
from collections import defaultdict

import torch

BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
KAPPAS = [0.4, 0.6, 0.8, 1.0]
VALUE_BITS = 16
EPS = 1e-10


# ----------------------------------------------------------------------------- transforms
def dct_matrix(n, device):
    """Orthonormal DCT-II, rows = frequencies (float32)."""
    k = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(n, dtype=torch.float64)[None, :]
    c = torch.cos(math.pi * k * (2 * j + 1) / (2 * n)) * math.sqrt(2.0 / n)
    c[0] *= 1.0 / math.sqrt(2.0)
    return c.float().to(device)


def haar_matrix(n, device):
    """Orthonormal unbalanced Haar basis for any n, rows ordered coarse to fine."""
    rows = [torch.full((n,), 1.0 / math.sqrt(n), dtype=torch.float64)]
    level = [(0, n)]
    while level:
        nxt = []
        for a, b in level:
            if b - a < 2:
                continue
            mid = (a + b) // 2
            left, right = mid - a, b - mid
            v = torch.zeros(n, dtype=torch.float64)
            v[a:mid] = math.sqrt(right / (left * (left + right)))
            v[mid:b] = -math.sqrt(left / (right * (left + right)))
            rows.append(v)
            nxt += [(a, mid), (mid, b)]
        level = nxt
    return torch.stack(rows).float().to(device)


_CACHE = {}


def cached(kind, n, device):
    key = (kind, n, str(device))
    if key not in _CACHE:
        _CACHE[key] = (dct_matrix if kind == "dct" else haar_matrix)(n, device)
    return _CACHE[key]


def sym_eigh_desc(g):
    """Eigen-decomposition of a symmetric matrix in float64, eigenvalues descending."""
    evals, evecs = torch.linalg.eigh(((g + g.T) / 2).double())
    return evals.flip(0), evecs.flip(1)


def fiedler_order(g):
    """Seriation: order indices by the Fiedler vector of the |correlation| graph of Gram g."""
    d = g.diagonal().clamp_min(EPS).sqrt()
    r = (g / d[:, None] / d[None, :]).abs().double()
    r.fill_diagonal_(0)
    lap = torch.diag(r.sum(1)) - r
    _, evecs = torch.linalg.eigh(lap)
    return torch.argsort(evecs[:, 1])


def permuted(t, perm):
    """T such that T x = t (x[perm])."""
    return t[:, torch.argsort(perm)]


# ----------------------------------------------------------------------------- budgets
def gar_rank(k_o, k_i, budget_params):
    """Largest r with r (k_i + k_o - r) <= budget_params, capped at min(k_o, k_i)."""
    s = k_i + k_o
    disc = s * s - 4.0 * budget_params
    if disc <= 0:
        return min(k_o, k_i)
    return max(1, min(int((s - math.sqrt(disc)) / 2.0), min(k_o, k_i)))


def topk_count(m, n, beta):
    bits = VALUE_BITS + math.ceil(math.log2(m * n))
    return max(1, min(m * n, int(beta * m * n * VALUE_BITS / bits)))


def geom_indices(d, s, theta=3.0):
    """LYRA geometric spacing in the lower half-spectrum; duplicates filled with lowest unused."""
    if s <= 1:
        return torch.zeros(1, dtype=torch.long)
    j = torch.arange(s, dtype=torch.float64)
    idx = torch.floor(d / 2.0 * (j / (s - 1)) ** theta).long().clamp_max(d - 1)
    chosen = sorted(set(idx.tolist()))
    used = set(chosen)
    fill = 0
    while len(chosen) < min(s, d):
        if fill not in used:
            chosen.append(fill)
            used.add(fill)
        fill += 1
    return torch.tensor(sorted(chosen), dtype=torch.long)


# ----------------------------------------------------------------------------- core
def lowrank_project(b, r):
    """Best rank-r approximation of b through an eigen-decomposition of the smaller Gram."""
    if r >= min(b.shape):
        return b
    if b.shape[0] <= b.shape[1]:
        _, u = sym_eigh_desc(b @ b.T)
        u = u[:, :r].float()
        return u @ (u.T @ b)
    _, v = sym_eigh_desc(b.T @ b)
    v = v[:, :r].float()
    return (b @ v) @ v.T


class Basis:
    """Coefficients of W in one basis, plus the Gram matrix expressed in that basis."""

    def __init__(self, name, w, gram, t_out=None, t_in=None):
        self.name = name
        self.c = w if t_out is None else t_out @ w @ t_in.T
        self.gram = gram if t_in is None else t_in @ gram @ t_in.T
        self.diag = self.gram.diagonal().clamp_min(EPS)
        self.den_plain = float((self.c.double() ** 2).sum())
        self.den_gram = float(((self.c @ self.gram).double() * self.c.double()).sum())
        self._proj = {}

    def errors(self, c_hat):
        e = self.c - c_hat
        plain = float((e.double() ** 2).sum()) / max(self.den_plain, EPS)
        gram = float(((e @ self.gram).double() * e.double()).sum()) / max(self.den_gram, EPS)
        return plain, max(gram, 0.0)

    def block(self, beta):
        m, n = self.c.shape
        p = max(1, min(m, round(m * math.sqrt(beta))))
        q = max(1, min(n, round(n * math.sqrt(beta))))
        out = torch.zeros_like(self.c)
        out[:p, :q] = self.c[:p, :q]
        return out, {"p": p, "q": q}

    def geom(self, beta):
        m, n = self.c.shape
        p = max(1, min(m, round(m * math.sqrt(beta))))
        q = max(1, min(n, round(n * math.sqrt(beta))))
        io = geom_indices(m, p).to(self.c.device)
        ii = geom_indices(n, q).to(self.c.device)
        out = torch.zeros_like(self.c)
        out[io[:, None], ii[None, :]] = self.c[io[:, None], ii[None, :]]
        return out, {"p": int(io.numel()), "q": int(ii.numel())}

    def topk(self, beta, data):
        m, n = self.c.shape
        k = topk_count(m, n, beta)
        score = self.c ** 2
        if data:
            score = score * self.diag[None, :]
        idx = torch.topk(score.flatten(), k, sorted=False).indices
        out = torch.zeros_like(self.c).flatten()
        out[idx] = self.c.flatten()[idx]
        return out.view(m, n), {"k": k}

    def rankband(self, beta, kappa, data):
        m, n = self.c.shape
        k_o = max(1, min(m, round(kappa * m)))
        k_i = max(1, min(n, round(kappa * n)))
        r = gar_rank(k_o, k_i, beta * m * n)
        b = self.c[:k_o, :k_i]
        scale = self.diag[:k_i].sqrt() if data else None
        if scale is not None:
            b = b * scale[None, :]
        b_r = lowrank_project(b, r)
        if scale is not None:
            b_r = b_r / scale[None, :]
        out = torch.zeros_like(self.c)
        out[:k_o, :k_i] = b_r
        return out, {"r": r, "k_o": k_o, "k_i": k_i, "kappa": kappa}


def analyse_matrix(w, gram, budgets, kappas, skip_haar=False):
    """All methods for one weight. Returns a list of result dicts (without layer metadata)."""
    m, n = w.shape
    dev = w.device
    gram = (gram + gram.T) / 2
    g_out = w @ gram @ w.T

    perm_in, perm_out = fiedler_order(gram), fiedler_order(g_out)
    _, q_in = sym_eigh_desc(gram)
    _, q_out = sym_eigh_desc(g_out)
    c_m, c_n = cached("dct", m, dev), cached("dct", n, dev)

    def make(name):
        if name == "spatial":
            return Basis(name, w, gram)
        if name == "dct":
            return Basis(name, w, gram, c_m, c_n)
        if name == "dct_perm":
            return Basis(name, w, gram, permuted(c_m, perm_out), permuted(c_n, perm_in))
        if name == "haar_perm":
            return Basis(name, w, gram, permuted(cached("haar", m, dev), perm_out),
                         permuted(cached("haar", n, dev), perm_in))
        return Basis(name, w, gram, q_out.T.float(), q_in.T.float())  # klt

    names = ["spatial", "dct", "dct_perm", "klt"] + ([] if skip_haar else ["haar_perm"])
    out = []

    def add(method, beta, c_hat, info, basis):
        plain, g = basis.errors(c_hat)
        rec = {"method": method, "budget": beta, "err_plain": plain, "err_gram": g}
        rec.update(info)
        out.append(rec)

    for name in names:
        basis = make(name)
        for beta in budgets:
            for data in (False, True):
                suffix = "_data" if data else ""
                add(name + "_topk" + suffix, beta, *basis.topk(beta, data), basis)
            if name == "spatial":
                add("svd", beta, *basis.rankband(beta, 1.0, False), basis)
                continue
            add(name + "_block", beta, *basis.block(beta), basis)
            if name == "dct":
                add("dct_geom", beta, *basis.geom(beta), basis)
            if name == "haar_perm":
                continue
            for kappa in kappas:
                for data in (False, True):
                    suffix = "_data" if data else ""
                    c_hat, info = basis.rankband(beta, kappa, data)
                    add(name + "_rankband" + suffix, beta, c_hat, info, basis)
                    if name == "klt" and data and kappa == 1.0:
                        add("datasvd", beta, c_hat, info, basis)
        del basis
    return out


def gaussian_null(m, n, budgets, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(m, n, generator=gen).to(device)
    eye = torch.eye(n, device=device)
    out = []
    spatial = Basis("spatial", w, eye)
    dct = Basis("dct", w, eye, cached("dct", m, device), cached("dct", n, device))
    for beta in budgets:
        for method, (c_hat, info), basis in (
            ("null_svd", spatial.rankband(beta, 1.0, False), spatial),
            ("null_dct_block", dct.block(beta), dct),
            ("null_dct_topk", dct.topk(beta, False), dct),
        ):
            plain, _ = basis.errors(c_hat)
            rec = {"method": method, "budget": beta, "err_plain": plain, "err_gram": None}
            rec.update(info)
            out.append(rec)
    return out


# ----------------------------------------------------------------------------- model side
def layer_meta(name):
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return int(p), ".".join(parts[i + 1:])
    return -1, name


def target_layers(model, min_dim):
    from transformers.pytorch_utils import Conv1D

    skip = re.compile(r"lm_head|embed|wte|wpe|classifier")
    found = []
    for name, mod in model.named_modules():
        if skip.search(name):
            continue
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight
        elif isinstance(mod, Conv1D):
            w = mod.weight.T  # Conv1D stores (in, out)
        else:
            continue
        if min(w.shape) >= min_dim:
            found.append((name, mod))
    return found


def weight_of(mod):
    return mod.weight.detach().float() if isinstance(mod, torch.nn.Linear) else mod.weight.detach().T.float()


def calibration_ids(tok, n_seq, seq_len):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"][:20000] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_seq = min(n_seq, ids.numel() // seq_len)
    return ids[: n_seq * seq_len].view(n_seq, seq_len)


@torch.no_grad()
def collect_grams(model, layers, ids, batch, device):
    grams, counts, hooks = {}, defaultdict(int), []

    def make_hook(name):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            g = x.T @ x
            grams[name] = g if name not in grams else grams[name] + g
            counts[name] += x.shape[0]
        return hook

    for name, mod in layers:
        hooks.append(mod.register_forward_pre_hook(make_hook(name)))
    for i in range(0, ids.shape[0], batch):
        model(ids[i:i + batch].to(device))
    for h in hooks:
        h.remove()
    return {k: v / counts[k] for k, v in grams.items()}


# ----------------------------------------------------------------------------- summary
def summarise(records, out_dir):
    """Mean error per (layer type, budget, method); rankband uses the best kappa per layer."""
    best = {}
    for r in records:
        if r["type"].startswith("null"):
            continue
        key = (r["layer"], r["type"], r["budget"], r["method"])
        for metric in ("err_plain", "err_gram"):
            k2 = key + (metric,)
            best[k2] = min(best.get(k2, float("inf")), r[metric])
    agg = defaultdict(list)
    for (layer, typ, beta, method, metric), v in best.items():
        agg[(typ, beta, method, metric)].append(v)
        agg[("ALL", beta, method, metric)].append(v)
    mean = {k: sum(v) / len(v) for k, v in agg.items()}
    null = defaultdict(list)
    for r in records:
        if r["type"].startswith("null"):
            null[(r["budget"], r["method"])].append(r["err_plain"])
    null = {k: sum(v) / len(v) for k, v in null.items()}

    types = sorted({k[0] for k in mean})
    budgets = sorted({k[1] for k in mean})
    methods = sorted({k[2] for k in mean})
    lines = ["# S0 spectral diagnostics summary", "",
             "Relative errors, mean over layers. rankband rows use the best kappa per layer.",
             "Lower is better. `err_gram` is the output-error criterion (DataSVD's objective).", ""]
    for metric in ("err_gram", "err_plain"):
        for typ in types:
            lines += ["## %s, layer type `%s`" % (metric, typ), "",
                      "| method | " + " | ".join("b=%.2f" % b for b in budgets) + " |",
                      "|---|" + "---|" * len(budgets)]
            rows = [(m, [mean.get((typ, b, m, metric)) for b in budgets]) for m in methods]
            rows.sort(key=lambda t: sum(x for x in t[1] if x is not None))
            for m, vals in rows:
                lines.append("| %s | " % m + " | ".join("%.4f" % v if v is not None else "-" for v in vals) + " |")
            lines.append("")
    lines += ["## Gaussian null (err_plain, mean over shapes)", "",
              "| method | " + " | ".join("b=%.2f" % b for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for m in sorted({k[1] for k in null}):
        lines.append("| %s | " % m + " | ".join("%.4f" % null.get((b, m), float("nan")) for b in budgets) + " |")

    lines += ["", "## Go / no-go (plan gate H0, H1)", "",
              "A cell (type, budget) passes if a fixed-transform method (dct, dct_perm, haar_perm: no basis",
              "storage) beats DataSVD on err_gram. The klt column is an upper bound only: its per-layer",
              "rotations are not counted in the budget and are free only where they can be absorbed.",
              "A method beats the null if its err_plain is below the Gaussian value for the same rule.", "",
              "| type | budget | datasvd | best fixed-transform method | err_gram | passes | best klt method | err_gram | dct_topk plain | null dct_topk |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    n_pass = n_cells = 0

    def best_of(typ, b, prefixes):
        cands = [(mean[(typ, b, m, "err_gram")], m) for m in methods
                 if m.startswith(prefixes) and (typ, b, m, "err_gram") in mean]
        return min(cands) if cands else (float("nan"), "-")

    for typ in types:
        for b in budgets:
            ref = mean.get((typ, b, "datasvd", "err_gram"))
            v, m = best_of(typ, b, ("dct", "haar"))
            if ref is None or m == "-":
                continue
            kv, km = best_of(typ, b, ("klt",))
            ok = v < ref
            n_cells += 1
            n_pass += ok
            lines.append("| %s | %.2f | %.4f | %s | %.4f | %s | %s | %.4f | %.4f | %.4f |" % (
                typ, b, ref, m, v, "yes" if ok else "no", km, kv,
                mean.get((typ, b, "dct_topk", "err_plain"), float("nan")), null.get((b, "null_dct_topk"), float("nan"))))
    lines += ["", "Cells passed: %d of %d." % (n_pass, n_cells)]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return n_pass, n_cells


# ----------------------------------------------------------------------------- entry points
def selftest():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    for n in (7, 16, 48):
        for kind in ("dct", "haar"):
            t = cached(kind, n, dev).double()
            assert torch.allclose(t @ t.T, torch.eye(n, dtype=torch.float64), atol=1e-5), (kind, n)
    m, n = 48, 80
    w = torch.randn(m, 12) @ torch.randn(12, n) + 0.1 * torch.randn(m, n)
    x = torch.randn(4000, 20) @ torch.randn(20, n) + 0.05 * torch.randn(4000, n)
    gram = x.T @ x / x.shape[0]
    recs = analyse_matrix(w, gram, [0.1, 0.3], KAPPAS)
    by = {(r["method"], r["budget"], r.get("kappa")): r for r in recs}
    for r in recs:
        assert -1e-6 <= r["err_plain"] <= 1 + 1e-4 and -1e-6 <= r["err_gram"] <= 1.5, r
    for beta in (0.1, 0.3):
        svd, dsvd = by[("svd", beta, 1.0)], by[("datasvd", beta, 1.0)]
        # an orthonormal change of basis must not change the plain SVD result
        assert abs(by[("dct_rankband", beta, 1.0)]["err_plain"] - svd["err_plain"]) < 1e-4
        # DataSVD is optimal for the Gram-weighted error, SVD for the plain error
        assert dsvd["err_gram"] <= svd["err_gram"] + 1e-5
        assert svd["err_plain"] <= dsvd["err_plain"] + 1e-5
    full = Basis("dct", w, gram, cached("dct", m, dev), cached("dct", n, dev))
    assert full.errors(full.block(1.0)[0])[0] < 1e-8
    null = gaussian_null(m, n, [0.3], dev)
    assert all(0 < r["err_plain"] < 1 for r in null)
    for r in recs:
        r.update(layer="synthetic", type="synthetic", depth=0, m=m, n=n)
    for r in null:
        r.update(layer="null", type="null_%dx%d" % (m, n), depth=-1, m=m, n=n)
    os.makedirs("outputs/s0_selftest", exist_ok=True)
    print("summary cells passed:", summarise(recs + null, "outputs/s0_selftest"))
    print("selftest OK (%d records)" % len(recs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="outputs/s0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layer-stride", type=int, default=1, help="analyse every k-th block")
    ap.add_argument("--min-dim", type=int, default=64)
    ap.add_argument("--max-layers", type=int, default=0, help="debug: stop after this many layers")
    ap.add_argument("--skip-haar", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="only download model, tokenizer and data")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    ids = calibration_ids(tok, args.n_seq, args.seq_len)
    if args.prefetch:
        print("prefetched", args.model, "calibration tokens", tuple(ids.shape))
        return
    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    model.to(dev).eval()

    layers = [(n, m) for n, m in target_layers(model, args.min_dim) if layer_meta(n)[0] % args.layer_stride == 0]
    if args.max_layers:
        layers = layers[: args.max_layers]
    print("%d layers, calibration %s" % (len(layers), tuple(ids.shape)))
    t0 = time.time()
    grams = collect_grams(model, layers, ids, args.batch, dev)
    print("grams collected in %.0f s" % (time.time() - t0))
    weights = {n: weight_of(m).clone() for n, m in layers}
    model.to("cpu")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    records, shapes = [], set()
    with open(os.path.join(args.out, "records.jsonl"), "w") as f:
        for i, (name, _) in enumerate(layers):
            t1 = time.time()
            w, g = weights.pop(name).to(dev), grams.pop(name).to(dev)
            depth, typ = layer_meta(name)
            m, n = w.shape
            new = analyse_matrix(w, g, BUDGETS, KAPPAS, args.skip_haar)
            for r in new:
                r.update(layer=name, type=typ, depth=depth, m=m, n=n)
            if (m, n) not in shapes:
                shapes.add((m, n))
                for r in gaussian_null(m, n, BUDGETS, dev):
                    r.update(layer="null", type="null_%dx%d" % (m, n), depth=-1, m=m, n=n)
                    new.append(r)
            for r in new:
                f.write(json.dumps(r) + "\n")
            f.flush()
            records += new
            print("[%d/%d] %s %dx%d  %.0f s" % (i + 1, len(layers), name, m, n, time.time() - t1), flush=True)
    n_pass, n_cells = summarise(records, args.out)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(vars(args), budgets=BUDGETS, kappas=KAPPAS, value_bits=VALUE_BITS,
                       n_layers=len(layers), seconds=time.time() - t0), f, indent=1)
    print("done in %.0f s; go/no-go cells passed %d of %d; see %s/summary.md" % (time.time() - t0, n_pass, n_cells, args.out))


if __name__ == "__main__":
    main()
