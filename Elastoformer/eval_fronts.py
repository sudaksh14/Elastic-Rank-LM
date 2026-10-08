"""E2: training-free accuracy-vs-cost fronts for both elasticity axes on the same pretrained DeiT-B, plus the DP additive-error check.

  prune : nested L1 pruning ladders (CR in --crs, N levels) executed through SlicedViT (== physically pruned, tests/test_sliced_equivalence.py)
  rank  : 72 encoder Linears -> ODLinear (SVD | DataSVD), stored rank capped at cap*min(in,out), profiles uniform | FlexRank-DP,
          at the SAME total parameter budgets as the pruning levels (params counted with FlexRank's GAR convention for ODLinear)
  additive : 30 random per-layer rank profiles, predicted (sum of single-layer probes) vs measured loss increase; Spearman / Kendall
Nothing is trained. Results are written incrementally to <out>/results_e2.json.
"""
import os, sys, json, time, math, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from models.elastoformer import ElasticViTForImageClassification, ElasticViTConfig
from models.sliced_vit import SlicedViT, build_level_indices
from utils.nested_metadata import run_nested_pruning, make_importance
from utils import flexrank_bridge as fb
from flexrank.profiles.dp import DPSearchAlgo
from flexrank.profiles.utils import count_od_model_params


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="facebook/deit-base-patch16-224")
    ap.add_argument("--data_path", default="/var/scratch/dchabal/quokka/data/imagenet")
    ap.add_argument("--cache_dir", default="/var/scratch/skalra/elastoSLM/data_lists")
    ap.add_argument("--stages", nargs="+", default=["prune", "rank", "additive"])
    ap.add_argument("--crs", nargs="+", type=float, default=[0.4, 0.6, 0.8])
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--decomps", nargs="+", default=["svd", "datasvd"])
    ap.add_argument("--caps", nargs="+", type=float, default=[1.0, 0.5, 0.3])
    ap.add_argument("--profiles", nargs="+", default=["uniform", "dp"])
    ap.add_argument("--calib_n", type=int, default=4096, help="train images for Gram matrices (DataSVD)")
    ap.add_argument("--probe_n", type=int, default=1024, help="subset of calib used for DP probing / additive check")
    ap.add_argument("--val_n", type=int, default=10000, help="fixed class-balanced val subset for the rank arm (0 = all 50k)")
    ap.add_argument("--dp_min_p", type=float, default=0.02)
    ap.add_argument("--dp_cuts", type=int, default=12)
    ap.add_argument("--n_random", type=int, default=30)
    ap.add_argument("--prune_full_val", action="store_true", help="also evaluate the pruning ladders on all 50k val images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dummy", action="store_true")
    return ap.parse_args()


# ----------------------------------------------------------------------------------- data
def cached_tensors(path_prefix, split_dir, n, per_class, seed, workers, tf):
    """Fixed class-balanced subset (or random n) -> fp16 tensor cache so every arm sees identical images."""
    import torchvision.datasets as tvd
    from torch.utils.data import DataLoader, Subset
    f = f"{path_prefix}.pt"
    if os.path.exists(f):
        d = torch.load(f)
        return d["x"], d["y"]
    ds = tvd.ImageFolder(split_dir, transform=tf)
    rng = np.random.RandomState(seed)
    tg = np.asarray(ds.targets)
    idx = []
    if per_class:
        for c in range(1000):
            idx += rng.choice(np.where(tg == c)[0], per_class, replace=False).tolist()
    else:
        idx = rng.choice(len(ds), n, replace=False).tolist()
    idx = sorted(idx)
    loader = DataLoader(Subset(ds, idx), batch_size=128, num_workers=workers, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.half()); ys.append(y)
    x, y = torch.cat(xs), torch.cat(ys)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f"{f}.tmp{os.getpid()}"                      # atomic publish: several jobs may build the same cache concurrently
    torch.save(dict(x=x, y=y, idx=idx), tmp); os.replace(tmp, f)
    json.dump(dict(n=len(idx), seed=seed, idx=idx), open(f + ".idx.json", "w"))
    return x, y


@torch.no_grad()
def top1(model, x, y, device, bs=256, level=None, amp=True):
    model.eval(); c = 0; loss = 0.0
    for i in range(0, x.shape[0], bs):
        xb, yb = x[i:i + bs].to(device).float(), y[i:i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            out = (model(xb, level) if level is not None else model(xb).logits).float()
        c += (out.argmax(1) == yb).sum().item(); loss += F.cross_entropy(out, yb, reduction="sum").item()
    return 100.0 * c / x.shape[0], loss / x.shape[0]


def save(res, path):
    json.dump(res, open(path, "w"), indent=1, default=str)


# ----------------------------------------------------------------------------------- stages
def stage_prune(a, base, dev, valx, valy, fullval, res):
    sd = base.state_dict()
    nl = base.config.num_hidden_layers
    base_cost = fb.vit_macs_params(base.config.hidden_size, [sd[f"vit.encoder.layer.{l}.attention.attention.query.weight"].shape[0] for l in range(nl)],
                                   [sd[f"vit.encoder.layer.{l}.intermediate.dense.weight"].shape[0] for l in range(nl)], classes=base.config.num_labels)
    b_acc, b_loss = top1(base, valx, valy, dev, amp=not a.dummy)
    res["base"] = dict(cost=base_cost, params_actual=sum(p.numel() for p in base.parameters()), top1_valsub=b_acc, loss_valsub=b_loss)   # S4: analytic vs actual
    print(f"[base] top1_valsub={b_acc:.2f} params={base_cost['params'] / 1e6:.2f}M modMACs={base_cost['macs_modules'] / 1e9:.2f}G", flush=True)
    res["prune"] = {}
    for cr in a.crs:
        t0 = time.time()
        core = deepcopy(base).cpu()
        meta = run_nested_pruning(core, deepcopy(base).cpu(), make_importance("l1"), torch.randn(1, 3, 224, 224), a.N, cr)
        groups, nl = build_level_indices(meta["pruned_index_out"], a.N, sd)
        torch.save(dict(pruned_index_out=meta["pruned_index_out"], N=a.N, CR=cr), os.path.join(a.out, f"nested_index_sets_cr{cr}.pt"))
        sl = SlicedViT(deepcopy(base), groups, nl, a.N).to(dev).eval()
        rows = []
        for L in sl.levels:
            E = len(sl.idx("E", L)) if L != sl.full_level() else base.config.hidden_size
            Qs = [(len(sl.idx(f"Q{l}", L)) if L != sl.full_level() else sd[f"vit.encoder.layer.{l}.attention.attention.query.weight"].shape[0]) for l in range(nl)]
            Fs = [(len(sl.idx(f"F{l}", L)) if L != sl.full_level() else sd[f"vit.encoder.layer.{l}.intermediate.dense.weight"].shape[0]) for l in range(nl)]
            cost = fb.vit_macs_params(E, Qs, Fs, classes=base.config.num_labels)
            acc, loss = top1(sl, valx, valy, dev, level=L, amp=not a.dummy)
            row = dict(level=L, embed=E, qkv0=Qs[0], ffn0=Fs[0], **cost, top1_valsub=acc, loss_valsub=loss)
            if fullval is not None:
                row["top1_val50k"] = top1(sl, fullval[0], fullval[1], dev, level=L, amp=not a.dummy)[0]
            rows.append(row)
            print(f"[prune CR={cr}] L{L} E={E} params={cost['params'] / 1e6:.2f}M modMACs={cost['macs_modules'] / 1e9:.2f}G top1={acc:.2f}", flush=True)
        res["prune"][str(cr)] = dict(levels=rows, wall_s=time.time() - t0)
        save(res, os.path.join(a.out, "results_e2.json"))
        del sl, core, meta
    return res


def uniform_profile(model_r, target_od, grid=None):
    """Smallest-p uniform rank fraction whose OD (GAR) parameter count <= target_od. Returns (dims, params, reachable)."""
    layers = [m for m in model_r.modules() if isinstance(m, fb.ODLayer)]
    maxd = [m.max_inner_dim for m in layers]

    def dims(p):
        return [max(1, min(md, int(math.ceil(p * md)))) for md in maxd]

    def cost(d):
        return sum(m.get_num_parameters(r) for m, r in zip(layers, d))
    # p is a fraction of the capped max rank
    if cost(dims(1.0)) <= target_od:
        return dims(1.0), cost(dims(1.0)), True
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if cost(dims(mid)) <= target_od:
            lo = mid
        else:
            hi = mid
    d = dims(lo)
    return d, cost(d), cost(d) <= target_od and lo > 0


def stage_rank(a, base, dev, calibx, caliby, valx, valy, res, budgets):
    res.setdefault("rank", {}); res.setdefault("s1", {})
    probex, probey = calibx[:a.probe_n], caliby[:a.probe_n]
    grams = None
    if "datasvd" in a.decomps:
        t0 = time.time()
        grams = fb.collect_grams(deepcopy(base), calibx.float(), dev)
        print(f"grams for {len(grams)} layers in {time.time() - t0:.0f}s", flush=True)
    for dec in a.decomps:
        for cap in a.caps:
            key = f"{dec}_cap{cap}"
            if key in res["rank"]:
                continue
            t0 = time.time()
            mr = fb.to_rank_model(base, grams if dec == "datasvd" else None, dec, cap, device=dev)
            layers = [m for m in mr.modules() if isinstance(m, fb.ODLayer)]
            od_full = count_od_model_params(mr)
            nonod = fb.count_total_params_rank(mr) - od_full
            cell = dict(max_ranks=[m.max_inner_dim for m in layers][:6], od_params_full=od_full, nonod_params=nonod, points=[])
            if dec == "datasvd" and cap == 1.0:     # S1: full-rank factorisation must reproduce the base network
                with torch.no_grad():
                    xs = valx[:64].to(dev).float()
                    d = (mr.eval()(xs).logits - base.to(dev).eval()(xs).logits).abs().max().item()
                res["s1"] = dict(max_abs_logit_diff_fp32=d, top1_rank_full=top1(mr, valx, valy, dev, amp=False)[0], top1_base=top1(base.to(dev), valx, valy, dev, amp=False)[0])
                print("S1:", res["s1"], flush=True)
            sol = None
            if "dp" in a.profiles:
                ev = fb.CalibEvaluator(mr, probex, probey, dev, amp=not a.dummy)
                algo = DPSearchAlgo(evaluator=ev, n_models=a.dp_cuts, min_p=a.dp_min_p)
                sol = algo.solve()
                pdata = sol.to_profiles_data()
                cell["dp_front_points"] = len(pdata.params)
                cell["layer_stats_path"] = os.path.join(a.out, f"layerstats_{key}.json")
                json.dump({n: dict(inner_dims=s.inner_dims, errors=s.errors, params_savings=s.params_savings) for n, s in sol.layers_stats.items()}, open(cell["layer_stats_path"], "w"))
                fb.set_profile(mr, fb.full_profile(mr))
            for (axis_tag, cr, L, total) in budgets:
                t_od = total - nonod
                for prof in a.profiles:
                    if prof == "uniform":
                        dims, got, ok = uniform_profile(mr, t_od)
                    else:
                        try:
                            dims, got = pdata.get_profile_for_params(t_od)
                            ok = True
                        except ValueError:
                            dims, got, ok = None, None, False
                    pt = dict(profile=prof, from_prune_cr=cr, from_prune_level=L, target_total_params=total, reachable=bool(ok and dims is not None))
                    if pt["reachable"]:
                        fb.set_profile(mr, dims)
                        acc, loss = top1(mr, valx, valy, dev, amp=not a.dummy)
                        calib_loss = fb.CalibEvaluator(mr, probex, probey, dev, amp=not a.dummy).evaluate()["eval_loss"]
                        odl = fb.od_cost_list(mr)
                        nl = base.config.num_hidden_layers
                        qfull = base.state_dict()["vit.encoder.layer.0.attention.attention.query.weight"].shape[0]   # attention width is not changed by rank
                        cost = fb.vit_macs_params(base.config.hidden_size, [qfull] * nl, [0] * nl, classes=base.config.num_labels, od=odl)
                        pt.update(params_gar=got + nonod, params_stored=cost["params_stored"], macs_modules=cost["macs_modules"], top1_valsub=acc,
                                  loss_valsub=loss, loss_calib=calib_loss, dims_head=dims[:12], dims_mean_frac=float(np.mean([r / m.max_inner_dim for r, m in zip(dims, layers)])))
                    cell["points"].append(pt)
                fb.set_profile(mr, fb.full_profile(mr))
            cell["wall_s"] = time.time() - t0
            res["rank"][key] = cell
            save(res, os.path.join(a.out, "results_e2.json"))
            print(f"[rank {key}] done in {cell['wall_s']:.0f}s; reachable {sum(p['reachable'] for p in cell['points'])}/{len(cell['points'])}", flush=True)
            if dec == "datasvd" and cap == 1.0 and "dp" in a.profiles:
                stage_additive(a, mr, sol, probex, probey, dev, res)
            del mr
            torch.cuda.empty_cache()
    return res


def stage_additive(a, mr, sol, probex, probey, dev, res):
    """predicted = sum of single-layer probe errors at the chosen ranks; measured = loss(profile) - loss(full), on the probe set."""
    ev = fb.CalibEvaluator(mr, probex, probey, dev, amp=not a.dummy)
    fb.set_profile(mr, fb.full_profile(mr))
    base_loss = ev.evaluate()["eval_loss"]
    stats = sol.layers_stats; names = list(stats.keys())
    rng = random.Random(a.seed)
    pred, meas, rows = [], [], []
    for k in range(a.n_random):
        sev = rng.uniform(0.15, 0.6)
        dims, p = [], 0.0
        for n in names:
            g = stats[n].inner_dims; j = len(g) - 1 - int(rng.random() * sev * len(g))
            j = max(0, j); dims.append(g[j]); p += max(stats[n].errors[j], 0.0)
        fb.set_profile(mr, dims)
        m = ev.evaluate()["eval_loss"] - base_loss
        pred.append(p); meas.append(m); rows.append(dict(pred=p, meas=m, sev=sev))
    fb.set_profile(mr, fb.full_profile(mr))
    rho, tau = fb.rank_corr(pred, meas)
    res["additive"] = dict(n=len(pred), spearman=rho, kendall=tau, pearson=float(np.corrcoef(pred, meas)[0, 1]),
                           mean_pred=float(np.mean(pred)), mean_meas=float(np.mean(meas)), rows=rows, base_loss_probe=base_loss)
    print(f"[additive] spearman={rho:.3f} kendall={tau:.3f} mean pred {np.mean(pred):.4f} meas {np.mean(meas):.4f}", flush=True)
    save(res, os.path.join(a.out, "results_e2.json"))


def main():
    a = args_()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.dummy else "cpu")
    res_path = os.path.join(a.out, "results_e2.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    res["args"] = vars(a); res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if a.dummy:
        torch.manual_seed(0)
        cfg = ElasticViTConfig(hidden_size=96, num_hidden_layers=2, num_attention_heads=6, intermediate_size=192, image_size=224, patch_size=16, num_labels=10, qkv_bias=True)
        cfg.pruned_dim = 96
        base = ElasticViTForImageClassification(cfg).eval()
        calibx, caliby = torch.randn(a.calib_n, 3, 224, 224).half(), torch.randint(0, 10, (a.calib_n,))
        valx, valy = torch.randn(a.val_n or 64, 3, 224, 224).half(), torch.randint(0, 10, (a.val_n or 64,))
        fullval = None
    else:
        import torchvision.transforms as T
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        base = ElasticViTForImageClassification.from_pretrained(a.model_name).eval()
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        calibx, caliby = cached_tensors(os.path.join(a.cache_dir, f"calib_{a.calib_n}_seed{a.seed}"), os.path.join(a.data_path, "train"), a.calib_n, 0, a.seed, a.workers, tf)
        if a.val_n:
            valx, valy = cached_tensors(os.path.join(a.cache_dir, f"val_{a.val_n}_seed{a.seed}"), os.path.join(a.data_path, "val"), a.val_n, a.val_n // 1000, a.seed, a.workers, tf)
        fullval = None
        if a.prune_full_val or not a.val_n:
            fullval = cached_tensors(os.path.join(a.cache_dir, "val_50000_all"), os.path.join(a.data_path, "val"), 50000, 50, a.seed, a.workers, tf)
            if not a.val_n:
                valx, valy = fullval
    base = base.to(dev)
    print("data:", calibx.shape, valx.shape, flush=True)

    if "prune" in a.stages:
        stage_prune(a, base, dev, valx, valy, fullval, res)
    if "rank" in a.stages:
        if "prune" not in res:
            raise SystemExit("rank stage needs the prune ladders (budgets): run stage 'prune' first (same --out)")
        budgets, seen = [], set()
        for cr, d in res["prune"].items():
            for r in d["levels"]:
                if r["params"] in seen:     # the full model appears in every ladder
                    continue
                seen.add(r["params"]); budgets.append(("prune", float(cr), r["level"], r["params"]))
        budgets.sort(key=lambda b: -b[3])
        stage_rank(a, base, dev, calibx, caliby, valx, valy, res, budgets)
    save(res, res_path)
    print("E2 DONE ->", res_path)


if __name__ == "__main__":
    main()
