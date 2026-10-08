"""E3: B=1 kernel microbenchmark. Does a rank-r factorised layer beat a structurally pruned DENSE layer of equal stored parameters?

Variants per (M tokens, in, out) at rank fraction f of min(in,out), r = ceil(f*min):
  dense_full      F.linear(in->out)
  dense_pruned    dense layer with BOTH dims shrunk by k so that k^2*in*out = r*(in+out)  (equal stored params to the two-GEMM form)
  lr_contig       two F.linear with contiguous factors  (in->r, r->out)
  lr_sliced       flexrank ODLinear(frwd_impl=SLICED): slices of full-rank factors each call (non-contiguous V columns), as during joint training
  gar             flexrank ODLinear after prune_weights(use_gar=True) (the deployed reparametrisation)
Shapes: ViT-B prefill (M=197) and Llama-3.2-1B prefill/decode (M=512 / M=1). fp16 and bf16. Optional CUDA-graph replay (removes launch overhead).
Reports median/p90 latency (us) over --iters iterations after --warmup warmup iterations. Writes CSV.
"""
import os, sys, csv, math, time, argparse
import torch
import torch.nn.functional as F
from flexrank.layers.linear import ODLinear
from flexrank.layers.base import ODImpl

SHAPES = {  # name: (M list, [(in,out)])
    "vitb": ([197], [(768, 768), (768, 3072), (3072, 768)]),
    "llama1b": ([1, 512], [(2048, 2048), (2048, 8192), (8192, 2048), (2048, 512)]),
}


def timeit(fn, warmup, iters, graph):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    else:
        run = fn
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); run(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)   # us
    ts.sort()
    return ts[len(ts) // 2], ts[int(0.9 * len(ts))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--dtypes", nargs="+", default=["float16", "bfloat16"])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--families", nargs="+", default=list(SHAPES))
    a = ap.parse_args()
    dev = "cuda"
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)
    rows = []
    for fam in a.families:
        Ms, mats = SHAPES[fam]
        for M in Ms:
            for (din, dout) in mats:
                for dt in a.dtypes:
                    dtype = getattr(torch, dt)
                    x = torch.randn(M, din, device=dev, dtype=dtype)
                    W = torch.randn(dout, din, device=dev, dtype=dtype) / math.sqrt(din)
                    bvec = torch.zeros(dout, device=dev, dtype=dtype)
                    mn = min(din, dout)
                    for graph in (False, True):
                        t50, t90 = timeit(lambda: F.linear(x, W, bvec), a.warmup, a.iters, graph)
                        rows.append(dict(family=fam, M=M, din=din, dout=dout, dtype=dt, graph=graph, variant="dense_full", frac=1.0, r=mn, params=din * dout, med_us=t50, p90_us=t90))
                        for f in a.fracs:
                            r = max(1, int(math.ceil(f * mn)))
                            stored = r * (din + dout)
                            k = min(1.0, math.sqrt(stored / (din * dout)))
                            pi, po = max(1, int(round(k * din))), max(1, int(round(k * dout)))
                            xp = torch.randn(M, pi, device=dev, dtype=dtype); Wp = torch.randn(po, pi, device=dev, dtype=dtype); bp = torch.zeros(po, device=dev, dtype=dtype)
                            V = torch.randn(r, din, device=dev, dtype=dtype); U = torch.randn(dout, r, device=dev, dtype=dtype)
                            od_s = ODLinear(din, dout, bias=True, device=dev, dtype=dtype, frwd_impl=ODImpl.SLICED); od_s.inner_dim = r
                            od_g = ODLinear(din, dout, bias=True, device=dev, dtype=dtype); od_g.inner_dim = r; od_g.prune_weights(use_gar=True)
                            fns = {"dense_pruned": (lambda: F.linear(xp, Wp, bp), po * pi),
                                   "lr_contig": (lambda: F.linear(F.linear(x, V), U, bvec), stored),
                                   "lr_sliced": (lambda: od_s(x), stored),
                                   "gar": (lambda: od_g(x), stored - r * r)}
                            with torch.no_grad():
                                for name, (fn, params) in fns.items():
                                    t50, t90 = timeit(fn, a.warmup, a.iters, graph)
                                    rows.append(dict(family=fam, M=M, din=din, dout=dout, dtype=dt, graph=graph, variant=name, frac=f, r=r, params=params, med_us=t50, p90_us=t90))
                            del od_s, od_g
                    print(f"{fam} M={M} {din}x{dout} {dt} done", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # crossover summary: smallest-latency variant among {dense_pruned, lr_contig, gar} at each frac
    print("\nM din dout dtype graph frac | dense_full dense_pruned lr_contig lr_sliced gar (us)")
    idx = {}
    for r in rows:
        idx[(r["family"], r["M"], r["din"], r["dout"], r["dtype"], r["graph"], r["variant"], r["frac"])] = r["med_us"]
    for (fam, M, din, dout, dt, g, v, f), t in sorted(idx.items()):
        if v == "dense_pruned" and dt == "float16":
            print(M, din, dout, dt, int(g), f, "|", *(f"{idx.get((fam, M, din, dout, dt, g, vv, f if vv != 'dense_full' else 1.0), float('nan')):.1f}" for vv in ("dense_full", "dense_pruned", "lr_contig", "lr_sliced", "gar")))
    print("E3 DONE ->", a.out)


if __name__ == "__main__":
    main()
