#!/usr/bin/env python
"""S0b: shared-rotation, held-out version of the S0 diagnostics. See spectral/implementation_plan.md.

S0 showed that a per-layer data basis (KLT) halves DataSVD's error, but each layer got its own
free rotation. S0b keeps only what a real network can have:

  * ONE orthogonal rotation of the residual stream, shared by every layer (absorbed into the
    embeddings, norms and projections for RMSNorm models, so it costs no parameters).
  * The residual side of each layer (columns of readers q/k/v/gate/up, rows of writers o/down)
    can be narrowed to the leading rho_r * d rotated coordinates.
  * The other side (MLP hidden units, head dims) cannot be rotated: only whole units can be
    dropped (selection, ranked by train error), never mixed.
  * Dropped INPUT coordinates are compensated by least squares (the same trick as SliceGPT):
    W_s <- W_s + W_d S_ds S_ss^-1, which costs nothing. Dropped OUTPUT coordinates are lost.
  * Everything is fitted on calibration split A (wikitext-2 train) and all errors are reported
    on held-out split B (wikitext-2 validation) as well as on A.

Error (per layer, relative): tr(E S E^T) / tr(W S W^T), E = W - W_hat, S = input Gram of that
layer (train or test). Budget = stored parameters / (m n); candidates with cost <= budget are
allowed and the best one by TRAIN error is reported (its test error is what counts).

Methods (all costs exact, no basis storage):
  svd, datasvd                 rank-r, plain and Gram-weighted (FlexRank today)
  {shared,native,own}_slice    residual side only, dropped inputs compensated
  {shared,native,own}_sel      residual narrowing x free-side unit selection (best pair)
  reader_slice_nocomp          reader slice without compensation (shows the gain from compensation)
  shared_rank                  rank-r Gram-weighted factorisation of the narrowed matrix
Transforms: shared = pooled PCA of the residual stream (the real candidate);
            native = coordinates ranked by variance, no rotation (pruning baseline);
            own    = per-layer PCA (not absorbable, upper bound).

Python 3.9 compatible.
"""
import argparse
import json
import os
import time
from collections import defaultdict

import torch

from diagnostics import (BUDGETS, collect_grams, gar_rank, layer_meta, sym_eigh_desc,
                         target_layers, weight_of)

R_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0]   # residual-side kept fraction
H_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]                          # free-side kept fraction
RANK_RHO = [0.4, 0.7]                                            # residual fractions for rank variants
READERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "c_attn", "c_fc")
WRITERS = ("o_proj", "down_proj", "c_proj")
EIG_FLOOR = 1e-7
RIDGE = 1e-4


def role_of(name):
    last = name.split(".")[-1]
    if last in READERS:
        return "reader"
    if last in WRITERS:
        return "writer"
    return None


# ----------------------------------------------------------------------------- linear algebra
def rowerr(e, g):
    """Per-row error e_i g e_i^T for e (a x b), g (b x b)."""
    return ((e @ g) * e).sum(1)


class GF:
    """Eigen-factors of a Gram matrix: S = G^(1/2), S^-1 (floored), eigenvectors (descending)."""

    def __init__(self, g):
        g = ((g + g.T) / 2).double()
        lam, q = sym_eigh_desc(g)
        lam = lam.clamp_min(float(lam[0]) * EIG_FLOOR)
        sq = lam.sqrt()
        self.rot = q.T.float()                 # rows = eigenvectors, descending
        self.S = ((q * sq) @ q.T).float()
        self.Sinv = (q / sq) @ q.T             # double


_CACHE = {}


def gf_cached(g):
    key = (g.shape[0], round(float(g.trace()), 5), round(float(g[0, 1]), 7), str(g.device))
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = GF(g)
    return _CACHE[key]


def wsvd(b, gf):
    """SVD of B S, so that the best rank-r of B in the G-weighted norm is (U_r s_r Vh_r) S^-1."""
    return torch.linalg.svd(b @ gf.S, full_matrices=False)


def lowrank(sv, gf, r):
    u, s, vh = sv
    return (((u[:, :r] * s[:r]) @ vh[:r]).double() @ gf.Sinv).float()


def ridge_solve(g, rhs):
    g = g.double()
    lam = RIDGE * float(g.diagonal().mean())
    return torch.linalg.solve(g + lam * torch.eye(g.shape[0], dtype=torch.float64, device=g.device), rhs.double())


def reader_slice(wp, sp, k):
    """Keep the first k input coordinates (rotated basis), compensate the dropped ones.
    Returns the error matrix E (m x n, W - W_hat) and the compensated kept block (m x k)."""
    m, n = wp.shape
    if k >= n:
        return torch.zeros_like(wp), wp.clone()
    x = ridge_solve(sp[:k, :k], sp[:k, k:])                 # S_ss^-1 S_sd   (k x (n-k))
    comp = (wp[:, k:].double() @ x.T).float()               # W_d S_ds S_ss^-1  (m x k)
    e = torch.empty_like(wp)
    e[:, :k] = -comp
    e[:, k:] = wp[:, k:]
    return e, wp[:, :k] + comp


def writer_input_select(w, s_tr, order, ns):
    """Keep the ns best input units (compensated). Returns E (m x n, spatial)."""
    m, n = w.shape
    if ns >= n:
        return torch.zeros_like(w)
    keep = order[:ns].sort().values
    drop = order[ns:].sort().values
    x = ridge_solve(s_tr[keep][:, keep], s_tr[keep][:, drop])   # ns x (n-ns)
    comp = (w[:, drop].double() @ x.T).float()
    e = torch.empty_like(w)
    e[:, keep] = -comp
    e[:, drop] = w[:, drop]
    return e


# ----------------------------------------------------------------------------- candidates
def cand(tag, cost, tr, te, **extra):
    return {"tag": tag, "cost": cost, "tr": tr, "te": te, "extra": extra}


def reader_cands(w, s_tr, s_te, rot, tname, gf_full, with_rank, out):
    m, n = w.shape
    wp, sp_tr, sp_te = w @ rot.T, rot @ s_tr @ rot.T, rot @ s_te @ rot.T
    d_tr, d_te = rowerr(wp, sp_tr), rowerr(wp, sp_te)
    dsum_tr, dsum_te = float(d_tr.sum()), float(d_te.sum())
    for rho in R_GRID:
        k = max(1, min(n, round(rho * n)))
        e, _ = reader_slice(wp, sp_tr, k)
        e_tr, e_te = rowerr(e, sp_tr), rowerr(e, sp_te)
        e0 = torch.zeros_like(wp)
        e0[:, k:] = wp[:, k:]
        out.append(cand(tname + "_slice_nocomp", k / n, float(rowerr(e0, sp_tr).sum()),
                        float(rowerr(e0, sp_te).sum()), rho_r=k / n, rho_h=1.0))
        score = d_tr - e_tr
        for rh in H_GRID:
            keep = max(1, min(m, round(rh * m)))
            if keep == m:
                tr, te = float(e_tr.sum()), float(e_te.sum())
            else:
                idx = torch.topk(score, keep).indices
                tr = float(e_tr[idx].sum()) + dsum_tr - float(d_tr[idx].sum())
                te = float(e_te[idx].sum()) + dsum_te - float(d_te[idx].sum())
            cost = (k / n) * (keep / m)
            extra = dict(rho_r=k / n, rho_h=keep / m)
            out.append(cand(tname + "_sel", cost, tr, te, **extra))
            if keep == m:
                out.append(cand(tname + "_slice", cost, tr, te, **extra))
    if with_rank:
        for rho in RANK_RHO:
            k = max(1, min(n, round(rho * n)))
            _, ws = reader_slice(wp, sp_tr, k)
            gfs = GF(sp_tr[:k, :k])
            sv = wsvd(ws, gfs)
            done = set()
            for b in BUDGETS:
                r = gar_rank(m, k, b * m * n)
                cost = r * (m + k - r)
                if cost > b * m * n * (1 + 1e-6) or r in done:
                    continue
                done.add(r)
                ep = wp.clone()
                ep[:, :k] -= lowrank(sv, gfs, r)
                out.append(cand("shared_rank", cost / (m * n), float(rowerr(ep, sp_tr).sum()),
                                float(rowerr(ep, sp_te).sum()), rho_r=k / n, r=r))


def writer_cands(w, s_tr, s_te, rots, with_rank, gf_full, out):
    m, n = w.shape
    order = torch.argsort((w ** 2).sum(0) * s_tr.diagonal(), descending=True)
    wps = {t: r @ w for t, r in rots.items()}
    d = {t: (rowerr(wp, s_tr), rowerr(wp, s_te)) for t, wp in wps.items()}
    dcum = {t: (v[0].cumsum(0).cpu(), v[1].cumsum(0).cpu()) for t, v in d.items()}
    for rh in H_GRID:
        ns = max(1, min(n, round(rh * n)))
        e_sp = writer_input_select(w, s_tr, order, ns)
        for t, rot in rots.items():
            er = rot @ e_sp
            etr, ete = rowerr(er, s_tr).cumsum(0).cpu(), rowerr(er, s_te).cumsum(0).cpu()
            dtr, dte = dcum[t]
            for rho in R_GRID:
                ko = max(1, min(m, round(rho * m)))
                tr = float(etr[ko - 1] + dtr[-1] - dtr[ko - 1])
                te = float(ete[ko - 1] + dte[-1] - dte[ko - 1])
                cost = (ko / m) * (ns / n)
                extra = dict(rho_r=ko / m, rho_h=ns / n)
                out.append(cand(t + "_sel", cost, tr, te, **extra))
                if ns == n:
                    out.append(cand(t + "_slice", cost, tr, te, **extra))
    if with_rank and "shared" in wps:
        wp = wps["shared"]
        for rho in RANK_RHO:
            ko = max(1, min(m, round(rho * m)))
            b = wp[:ko]
            sv = wsvd(b, gf_full)
            done = set()
            for bud in BUDGETS:
                r = gar_rank(ko, n, bud * m * n)
                cost = r * (ko + n - r)
                if cost > bud * m * n * (1 + 1e-6) or r in done:
                    continue
                done.add(r)
                ep = wp.clone()
                ep[:ko] = b - lowrank(sv, gf_full, r)
                out.append(cand("shared_rank", cost / (m * n), float(rowerr(ep, s_tr).sum()),
                                float(rowerr(ep, s_te).sum()), rho_r=ko / m, r=r))


def baseline_cands(w, s_tr, s_te, gf_full, out):
    m, n = w.shape
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    sv = wsvd(w, gf_full)
    done_s, done_d = set(), set()
    for b in BUDGETS:
        r = gar_rank(m, n, b * m * n)
        cost = r * (m + n - r)
        if cost > b * m * n * (1 + 1e-6):
            continue
        if r not in done_s:
            done_s.add(r)
            e = w - (u[:, :r] * s[:r]) @ vh[:r]
            out.append(cand("svd", cost / (m * n), float(rowerr(e, s_tr).sum()), float(rowerr(e, s_te).sum()), r=r))
        if r not in done_d:
            done_d.add(r)
            e = w - lowrank(sv, gf_full, r)
            out.append(cand("datasvd", cost / (m * n), float(rowerr(e, s_tr).sum()), float(rowerr(e, s_te).sum()), r=r))


def pick(cands, budgets, den_tr, den_te, base):
    out = []
    for tag in sorted({c["tag"] for c in cands}):
        cs = [c for c in cands if c["tag"] == tag]
        for b in budgets:
            ok = [c for c in cs if c["cost"] <= b * (1 + 1e-6)]
            if not ok:
                continue
            c = min(ok, key=lambda x: x["tr"])
            rec = dict(base)
            rec.update(method=tag, budget=b, cost=c["cost"], err_train=c["tr"] / den_tr, err_test=c["te"] / den_te)
            rec.update(c["extra"])
            out.append(rec)
    return out


def analyse_layer(w, s_tr, s_te, role, t_shared, t_native, base):
    m, n = w.shape
    den_tr, den_te = float(rowerr(w, s_tr).sum()), float(rowerr(w, s_te).sum())
    gf_full = gf_cached(s_tr)
    cands = []
    baseline_cands(w, s_tr, s_te, gf_full, cands)
    if role == "reader":
        reader_cands(w, s_tr, s_te, t_shared, "shared", gf_full, True, cands)
        reader_cands(w, s_tr, s_te, t_native, "native", gf_full, False, cands)
        reader_cands(w, s_tr, s_te, gf_full.rot, "own", gf_full, False, cands)
    else:
        own = GF(w @ s_tr @ w.T).rot
        writer_cands(w, s_tr, s_te, {"shared": t_shared, "native": t_native, "own": own}, True, gf_full, cands)
    return pick(cands, BUDGETS, den_tr, den_te, base)


# ----------------------------------------------------------------------------- residual rotation
def pooled_rotations(infos):
    """infos: list of (role, w, s_tr) on one device. Returns (shared rotation, native rotation, d)."""
    reader_g, writer_g = [], []
    for role, w, s in infos:
        if role == "reader":
            g = s.double()
            reader_g.append(g / g.trace())
        else:
            g = (w @ s @ w.T).double()
            writer_g.append(g / g.trace())
    parts = [torch.stack(p).mean(0) for p in (reader_g, writer_g) if p]
    g_res = torch.stack(parts).mean(0)
    _, evecs = sym_eigh_desc(g_res)
    shared = evecs.T.float().contiguous()
    order = torch.argsort(g_res.diagonal(), descending=True)
    native = torch.eye(g_res.shape[0], device=g_res.device)[order].contiguous()
    return shared, native, g_res.shape[0]


# ----------------------------------------------------------------------------- summary
def summarise(records, out_dir):
    agg = defaultdict(list)
    gap = defaultdict(list)
    for r in records:
        for typ in (r["type"], "ALL"):
            agg[(typ, r["budget"], r["method"])].append((r["err_train"], r["err_test"]))
    mean_te = {k: sum(v[1] for v in vs) / len(vs) for k, vs in agg.items()}
    mean_tr = {k: sum(v[0] for v in vs) / len(vs) for k, vs in agg.items()}
    types = sorted({k[0] for k in mean_te})
    budgets = sorted({k[1] for k in mean_te})
    methods = sorted({k[2] for k in mean_te})
    lines = ["# S0b summary: shared residual rotation, held-out error", "",
             "Mean relative error over layers on HELD-OUT tokens (lower is better). Budget = stored",
             "parameters / dense parameters; no basis storage is charged because the residual rotation",
             "is shared and absorbable (RMSNorm models). `own_*` rotations are per layer and not absorbable.", ""]
    for typ in ["ALL"] + [t for t in types if t != "ALL"]:
        lines += ["## err_test, layer type `%s`" % typ, "",
                  "| method | " + " | ".join("b=%.2f" % b for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
        rows = [(m, [mean_te.get((typ, b, m)) for b in budgets]) for m in methods]
        rows.sort(key=lambda t: sum(x for x in t[1] if x is not None) / max(1, sum(1 for x in t[1] if x is not None)))
        for m, vals in rows:
            lines.append("| %s | " % m + " | ".join("%.4f" % v if v is not None else "-" for v in vals) + " |")
        lines.append("")
    lines += ["## Train vs held-out error (ALL layers)", "",
              "| method | " + " | ".join("b=%.2f tr / te" % b for b in budgets) + " |", "|---|" + "---|" * len(budgets)]
    for m in ("datasvd", "shared_slice", "shared_sel", "shared_rank", "own_sel"):
        if (("ALL", budgets[0], m)) in mean_te:
            lines.append("| %s | " % m + " | ".join(
                "%.3f / %.3f" % (mean_tr[("ALL", b, m)], mean_te[("ALL", b, m)]) if ("ALL", b, m) in mean_te else "-"
                for b in budgets) + " |")
    lines += ["", "## Go / no-go: does a shared-basis method beat DataSVD on held-out error?", "",
              "Candidates: shared_slice, shared_sel, shared_rank (all absorbable). `own_*` and `native_*` are references.", "",
              "| type | budget | datasvd | best shared method | err_test | beats | native_sel | own_sel | layers won |",
              "|---|---|---|---|---|---|---|---|---|"]
    by_layer = defaultdict(dict)
    for r in records:
        by_layer[(r["layer"], r["budget"])][r["method"]] = r["err_test"]
    n_pass = n_cells = 0
    for typ in types:
        if typ == "ALL":
            continue
        for b in budgets:
            ref = mean_te.get((typ, b, "datasvd"))
            cands = [(mean_te[(typ, b, m)], m) for m in ("shared_slice", "shared_sel", "shared_rank") if (typ, b, m) in mean_te]
            if ref is None or not cands:
                continue
            v, m = min(cands)
            ok = v < ref
            n_cells += 1
            n_pass += ok
            won = tot = 0
            for (layer, bb), d in by_layer.items():
                if bb != b or layer_type_of(layer) != typ or "datasvd" not in d:
                    continue
                sh = [d[x] for x in ("shared_slice", "shared_sel", "shared_rank") if x in d]
                if sh:
                    tot += 1
                    won += min(sh) < d["datasvd"]
            lines.append("| %s | %.2f | %.4f | %s | %.4f | %s | %.4f | %.4f | %d/%d |" % (
                typ, b, ref, m, v, "yes" if ok else "no", mean_te.get((typ, b, "native_sel"), float("nan")),
                mean_te.get((typ, b, "own_sel"), float("nan")), won, tot))
    lines += ["", "Cells where a shared-basis method beats DataSVD: %d of %d." % (n_pass, n_cells)]
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return n_pass, n_cells


def layer_type_of(name):
    return layer_meta(name)[1]


# ----------------------------------------------------------------------------- data
def split_ids(tok, split, n_seq, seq_len):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(t for t in ds["text"][:20000] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_seq = min(n_seq, ids.numel() // seq_len)
    return ids[: n_seq * seq_len].view(n_seq, seq_len)


# ----------------------------------------------------------------------------- selftest
def selftest():
    torch.manual_seed(0)
    dev = torch.device("cpu")
    d, h = 24, 40
    q, _ = torch.linalg.qr(torch.randn(d, d))
    spec = torch.tensor([10.0 * 0.7 ** i for i in range(d)])

    def sample(nn):
        return (torch.randn(nn, d) * spec.sqrt()) @ q.T

    x_tr, x_te = sample(6000), sample(6000)
    cov = lambda x: x.T @ x / x.shape[0]
    s_in_tr, s_in_te = cov(x_tr), cov(x_te)
    w_r = torch.randn(h, d)
    wr_w = torch.randn(d, h)
    proj = torch.randn(d, h)
    z_tr, z_te = x_tr @ proj + 0.1 * torch.randn(6000, h), x_te @ proj + 0.1 * torch.randn(6000, h)
    s_h_tr, s_h_te = cov(z_tr), cov(z_te)
    infos = [("reader", w_r, s_in_tr), ("writer", wr_w, s_h_tr)]
    shared, native, dd = pooled_rotations(infos)
    assert dd == d and torch.allclose(shared @ shared.T, torch.eye(d), atol=1e-4)
    assert torch.allclose(native @ native.T, torch.eye(d), atol=1e-6)

    # per-row decomposition equals the direct trace formula (reader)
    wp = w_r @ shared.T
    sp = shared @ s_in_tr @ shared.T
    k = 10
    e, ws = reader_slice(wp, sp, k)
    dvec, evec = rowerr(wp, sp), rowerr(e, sp)
    idx = torch.tensor([0, 3, 5, 7, 11, 13, 20, 30])
    what = torch.zeros_like(wp)
    what[idx, :k] = ws[idx]
    direct = float(rowerr(wp - what, sp).sum())
    mask = torch.ones(h, dtype=torch.bool)
    mask[idx] = False
    rows = float(evec[idx].sum() + dvec[mask].sum())
    assert abs(direct - rows) < 1e-3 * max(1.0, direct), (direct, rows)
    # compensation never hurts on the fitting split
    e0 = torch.zeros_like(wp)
    e0[:, k:] = wp[:, k:]
    assert float(rowerr(e, sp).sum()) <= float(rowerr(e0, sp).sum()) + 1e-4

    # rotation invariance: rank-3 Gram-weighted error is the same in the rotated basis
    gf1, gf2 = GF(s_in_tr), GF(sp)
    l1, l2 = lowrank(wsvd(w_r, gf1), gf1, 3), lowrank(wsvd(wp, gf2), gf2, 3)
    e1 = float(rowerr(w_r - l1, s_in_tr).sum())
    e2 = float(rowerr(wp - l2, sp).sum())
    assert abs(e1 - e2) < 2e-3 * e1, (e1, e2)

    for role, w, s_tr, s_te in (("reader", w_r, s_in_tr, s_in_te), ("writer", wr_w, s_h_tr, s_h_te)):
        base = dict(layer="t.%s" % role, type=role, depth=0, m=w.shape[0], n=w.shape[1], role=role)
        recs = analyse_layer(w, s_tr, s_te, role, shared, native, base)
        by = {(r["method"], r["budget"]): r for r in recs}
        for r in recs:
            assert 0 <= r["err_train"] <= 1.02 and 0 <= r["err_test"] <= 1.05, r
            assert r["cost"] <= r["budget"] * (1 + 1e-5), r
        for b in (0.2, 0.5):
            assert by[("datasvd", b)]["err_train"] <= by[("svd", b)]["err_train"] + 1e-4
            assert by[("shared_sel", b)]["err_train"] <= by[("shared_slice", b)]["err_train"] + 1e-6
            assert by[("own_sel", b)]["err_train"] > 0
            if role == "reader":
                assert by[("shared_slice", b)]["err_train"] <= by[("shared_slice_nocomp", b)]["err_train"] + 1e-4
        # aligned PCA beats an arbitrary coordinate choice on the structured reader
        if role == "reader":
            assert by[("shared_slice", 0.2)]["err_train"] < by[("native_slice", 0.2)]["err_train"]
    # writer cumulative-sum formula equals a direct evaluation
    order = torch.argsort((wr_w ** 2).sum(0) * s_h_tr.diagonal(), descending=True)
    ns, ko = 25, 9
    e_sp = writer_input_select(wr_w, s_h_tr, order, ns)
    er = shared @ e_sp
    wpw = shared @ wr_w
    et = rowerr(er, s_h_tr)
    dt = rowerr(wpw, s_h_tr)
    via_cum = float(et.cumsum(0)[ko - 1] + dt.sum() - dt.cumsum(0)[ko - 1])
    ehat = er.clone()
    ehat[ko:] = wpw[ko:]
    assert abs(float(rowerr(ehat, s_h_tr).sum()) - via_cum) < 1e-3 * max(1.0, via_cum)
    os.makedirs("outputs/s0b_selftest", exist_ok=True)
    recs = []
    for role, w, s_tr, s_te in (("reader", w_r, s_in_tr, s_in_te), ("writer", wr_w, s_h_tr, s_h_te)):
        base = dict(layer="t.%s" % role, type=role, depth=0, m=w.shape[0], n=w.shape[1], role=role)
        recs += analyse_layer(w, s_tr, s_te, role, shared, native, base)
    print("summary:", summarise(recs, "outputs/s0b_selftest"))
    print("selftest OK (%d records)" % len(recs))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--out", default="outputs/s0b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-seq", type=int, default=128)
    ap.add_argument("--n-seq-test", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--min-dim", type=int, default=64)
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    ids_tr = split_ids(tok, "train", args.n_seq, args.seq_len)
    ids_te = split_ids(tok, "validation", args.n_seq_test, args.seq_len)
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    print("calibration (fit)", tuple(ids_tr.shape), "held-out", tuple(ids_te.shape))
    model.to(dev).eval()
    layers = [(n, m) for n, m in target_layers(model, args.min_dim)
              if layer_meta(n)[0] % args.layer_stride == 0 and role_of(n)]
    if args.max_layers:
        layers = layers[: args.max_layers]
    t0 = time.time()
    g_tr = {k: v.cpu() for k, v in collect_grams(model, layers, ids_tr, args.batch, dev).items()}
    g_te = {k: v.cpu() for k, v in collect_grams(model, layers, ids_te, args.batch, dev).items()}
    weights = {n: weight_of(m).clone().cpu() for n, m in layers}
    model.to("cpu")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print("%d layers, grams collected in %.0f s" % (len(layers), time.time() - t0), flush=True)

    infos = [(role_of(n), weights[n].to(dev), g_tr[n].to(dev)) for n, _ in layers]
    shared, native, d = pooled_rotations(infos)
    for (role, w, _), (n, _) in zip(infos, layers):
        assert (w.shape[1] if role == "reader" else w.shape[0]) == d, (n, w.shape, d)
    del infos
    print("residual width d = %d; pooled rotation ready" % d, flush=True)

    records = []
    with open(os.path.join(args.out, "records.jsonl"), "w") as f:
        for i, (name, _) in enumerate(layers):
            t1 = time.time()
            w, s_tr, s_te = weights.pop(name).to(dev), g_tr.pop(name).to(dev), g_te.pop(name).to(dev)
            depth, typ = layer_meta(name)
            base = dict(layer=name, type=typ, depth=depth, m=w.shape[0], n=w.shape[1], role=role_of(name))
            new = analyse_layer(w, s_tr, s_te, role_of(name), shared, native, base)
            for r in new:
                f.write(json.dumps(r) + "\n")
            f.flush()
            records += new
            print("[%d/%d] %s %dx%d %s  %.0f s" % (i + 1, len(layers), name, w.shape[0], w.shape[1], role_of(name),
                                                  time.time() - t1), flush=True)
    n_pass, n_cells = summarise(records, args.out)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(dict(vars(args), budgets=BUDGETS, r_grid=R_GRID, h_grid=H_GRID, rank_rho=RANK_RHO, d=d,
                       n_layers=len(layers), seconds=time.time() - t0), f, indent=1)
    print("done in %.0f s; cells where a shared-basis method beats DataSVD: %d of %d; see %s/summary.md"
          % (time.time() - t0, n_pass, n_cells, args.out))


if __name__ == "__main__":
    main()
