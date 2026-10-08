"""Joint nested training of all Elastoformer levels on ONE shared full-width weight set (E4a "J" arms).

Differences to the frozen-prefix pipeline (main.py): no pruning-round-by-round rebuild and no gradient freezing. Every step
samples level(s) from the nested family (utils/nested_metadata.py index sets, models/sliced_vit.py), so lower levels are
revised in light of the larger ones. Optionally distills from the frozen pretrained model (KD, FlexRank default ce=0, kl=1).
Single GPU. Data/augmentation/mixup/optimizer family are the same code paths as main.py (datasets.load_imagenet, datasets.mixup_fn).

Compute matching with main.py (N+1 stages x E epochs): --total_level_epochs = (N+1)*E counts level-forwards, i.e. one batch at one
level = one pass of those samples; the sandwich sampler spends 3 level-forwards per step and runs proportionally fewer steps.
"""
import os, sys, json, time, math, random, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.elastoformer import ElasticViTForImageClassification
from models.sliced_vit import SlicedViT, build_level_indices
from utils.nested_metadata import run_nested_pruning, make_importance
from datasets import load_imagenet, mixup_fn


class RankLevels(nn.Module):
    """Levels of a nested low-rank model: level L = one per-layer rank profile (a prefix of every ODLinear's ordered factors)."""

    def __init__(self, model, profiles):
        super().__init__()
        self.model, self.profiles = model, profiles
        self.levels = sorted(profiles)

    def full_level(self):
        return self.levels[-1]

    def forward(self, x, level=None):
        from utils import flexrank_bridge as fb
        fb.set_profile(self.model, self.profiles[self.full_level() if level is None else level])
        return self.model(x).logits


def build_rank_levels(args, base, sliced, nl, device):
    """DataSVD -> ODLinear (capped rank) -> FlexRank DP profiles matched to the pruning ladder's parameter totals."""
    from utils import flexrank_bridge as fb
    from flexrank.profiles.dp import DPSearchAlgo
    from flexrank.profiles.utils import count_od_model_params
    N = args.pruning_steps
    sd = base.state_dict()
    if args.dummy:
        calibx, caliby = torch.randn(args.calib_n, 3, 224, 224).half(), torch.randint(0, 10, (args.calib_n,))
    else:
        import torchvision.transforms as T
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        from eval_fronts import cached_tensors
        tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        calibx, caliby = cached_tensors(os.path.join(args.cache_dir, f"calib_{args.calib_n}_seed0"), os.path.join(args.data_path, "train"), args.calib_n, 0, 0, args.workers, tf)
    base_dev = deepcopy(base).to(device).eval()
    grams = fb.collect_grams(base_dev, calibx.float(), device) if args.decomp == "datasvd" else None
    rank_model = fb.to_rank_model(base_dev, grams, args.decomp, args.r_cap, device=device)
    del base_dev
    od_full = count_od_model_params(rank_model)
    nonod = fb.count_total_params_rank(rank_model) - od_full
    ev = fb.CalibEvaluator(rank_model, calibx[:args.probe_n], caliby[:args.probe_n], device, amp=not args.dummy)
    t0 = time.time()
    sol = DPSearchAlgo(evaluator=ev, n_models=args.dp_cuts, min_p=args.dp_min_p).solve()
    pdata = sol.to_profiles_data()
    print(f"DP profile search in {time.time() - t0:.0f}s ({len(pdata.params)} front points)", flush=True)
    fb.set_profile(rank_model, fb.full_profile(rank_model))
    profiles, info = {}, {}
    for L in range(1, N + 1):          # budgets = total params of pruning ladder level L
        E = len(sliced.idx("E", L))
        Qs = [len(sliced.idx(f"Q{l}", L)) for l in range(nl)]
        Fs = [len(sliced.idx(f"F{l}", L)) for l in range(nl)]
        target_total = fb.vit_macs_params(E, Qs, Fs, classes=base.config.num_labels)["params"]
        try:
            dims, got = pdata.get_profile_for_params(target_total - nonod)
        except ValueError:
            dims, got = pdata.profiles[-1], pdata.params[-1]          # smallest DP profile if the budget is below the search range
        profiles[L] = [int(d) for d in dims]
        info[L] = dict(target_total_params=target_total, od_params=int(got), reached_total=int(got) + nonod)
    profiles[N + 1] = fb.full_profile(rank_model)                   # top mode = full CAPPED rank (cannot reproduce the dense model if r_cap<1)
    info[N + 1] = dict(target_total_params=None, od_params=int(od_full), reached_total=int(od_full) + nonod)
    json.dump(dict(profiles=profiles, info=info, r_cap=args.r_cap, nonod=nonod), open(os.path.join(args.out, "rank_profiles.json"), "w"))
    model = RankLevels(rank_model, profiles).to(device).train()
    nparams = {}
    for L in model.levels:
        fb.set_profile(rank_model, profiles[L])
        nparams[L] = fb.count_total_params_rank(rank_model)
    fb.set_profile(rank_model, profiles[model.full_level()])
    print("stored params (M) of top mode:", round(sum(p.numel() for p in rank_model.parameters()) / 1e6, 2),
          "vs dense", round(sum(p.numel() for p in base.parameters()) / 1e6, 2), flush=True)
    return model, rank_model, nparams


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_name", default="facebook/deit-base-patch16-224")
    ap.add_argument("--data_path", default="/var/scratch/dchabal/quokka/data/imagenet")
    ap.add_argument("--pruning_ratio", type=float, default=0.8)
    ap.add_argument("--pruning_steps", type=int, default=3)
    ap.add_argument("--pruning_type", default="l1")
    ap.add_argument("--metadata", default="", help="saved nested index sets (.pt); computed + saved if missing")
    ap.add_argument("--axis", default="prune", choices=["prune", "rank"], help="prune: nested L1 index sets (SlicedViT); rank: nested low-rank ODLinear + DP profiles")
    ap.add_argument("--r_cap", type=float, default=0.5, help="[rank] stored rank cap as a fraction of min(in,out)")
    ap.add_argument("--decomp", default="datasvd", choices=["svd", "datasvd"])
    ap.add_argument("--calib_n", type=int, default=4096)
    ap.add_argument("--probe_n", type=int, default=1024)
    ap.add_argument("--dp_cuts", type=int, default=12)
    ap.add_argument("--dp_min_p", type=float, default=0.02)
    ap.add_argument("--cache_dir", default="/var/scratch/skalra/elastoSLM/data_lists")
    ap.add_argument("--sampler", default="uniform", choices=["uniform", "sandwich", "weighted"])
    ap.add_argument("--level_probs", type=float, nargs="*", default=None, help="for --sampler weighted: prob of Level 1..N+1")
    ap.add_argument("--loss", default="ce", choices=["ce", "kd", "mix"])
    ap.add_argument("--ce_w", type=float, default=None)
    ap.add_argument("--kl_w", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--total_level_epochs", type=float, required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup_epochs", type=float, default=1.0, help="in level-epochs")
    ap.add_argument("--lr_min", type=float, default=0.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.11)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--micro_batch", type=int, default=0, help="split each batch into micro-batches of this size (0 = off); memory only")
    ap.add_argument("--train_subset", type=float, default=0.0)
    ap.add_argument("--subset_seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--val_workers", type=int, default=4)
    ap.add_argument("--eval_every_level_epochs", type=float, default=0.0, help="0 = only at the end")
    ap.add_argument("--val_limit", type=int, default=0, help="smoke tests only")
    ap.add_argument("--dummy", action="store_true", help="smoke test: random tiny ViT + random data (CPU)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print_freq", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args()


def kd_loss(s, t, T):
    return F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="batchmean") * T * T


def sample_levels(args, rng, levels):
    if args.sampler == "uniform":
        return [rng.choice(levels)]
    if args.sampler == "weighted":
        return [rng.choices(levels, weights=args.level_probs, k=1)[0]]
    return sorted({levels[-1], levels[0], rng.choice(levels)})  # sandwich: largest + smallest + one random (dedup)


@torch.no_grad()
def evaluate_levels(model, loader, device, levels, limit=0):
    model.eval()
    res = {L: dict(c1=0, c5=0, n=0, loss=0.0) for L in levels}
    for bi, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for L in levels:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x, L).float()
            r = res[L]
            top5 = out.topk(5, 1).indices
            r["c1"] += (top5[:, 0] == y).sum().item(); r["c5"] += (top5 == y[:, None]).any(1).sum().item()
            r["n"] += y.numel(); r["loss"] += F.cross_entropy(out, y, reduction="sum").item()
        if limit and (bi + 1) * x.shape[0] >= limit:
            break
    model.train()
    return {L: dict(top1=100 * r["c1"] / r["n"], top5=100 * r["c5"] / r["n"], loss=r["loss"] / r["n"], n=r["n"]) for L, r in res.items()}


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.dummy else "cpu")
    if args.loss == "kd":
        ce_w, kl_w = 0.0, 1.0
    elif args.loss == "ce":
        ce_w, kl_w = 1.0, 0.0
    else:
        ce_w, kl_w = (args.ce_w if args.ce_w is not None else 0.5), (args.kl_w if args.kl_w is not None else 0.5)
    N = args.pruning_steps
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)

    # ---------------- model, nested index sets ----------------
    if args.dummy:
        from models.elastoformer import ElasticViTConfig
        cfg = ElasticViTConfig(hidden_size=96, num_hidden_layers=2, num_attention_heads=6, intermediate_size=192, image_size=224,
                               patch_size=16, num_labels=10, qkv_bias=True); cfg.pruned_dim = 96
        base = ElasticViTForImageClassification(cfg)
    else:
        base = ElasticViTForImageClassification.from_pretrained(args.model_name)
    teacher = deepcopy(base).eval().requires_grad_(False) if kl_w > 0 else None
    sd0 = deepcopy(base.state_dict())

    meta_path = args.metadata or os.path.join(args.out, "nested_index_sets.pt")
    if os.path.exists(meta_path):
        pruned_out = torch.load(meta_path)["pruned_index_out"]
        print("loaded index sets", meta_path)
    else:
        t0 = time.time()
        core = deepcopy(base).cpu()
        meta = run_nested_pruning(core, deepcopy(base).cpu(), make_importance(args.pruning_type), torch.randn(1, 3, 224, 224), N, args.pruning_ratio)
        pruned_out = meta["pruned_index_out"]
        torch.save(dict(pruned_index_out=pruned_out, N=N, CR=args.pruning_ratio, model=args.model_name), meta_path)
        print(f"computed index sets in {time.time() - t0:.0f}s -> {meta_path}"); del core, meta
    groups, nl = build_level_indices(pruned_out, N, sd0)
    sliced = SlicedViT(deepcopy(base), groups, nl, N)

    if args.axis == "prune":
        model = sliced.to(device).train()
        core_module = model.base
        nparams = model.level_stats(None)
    else:
        model, core_module, nparams = build_rank_levels(args, base, sliced, nl, device)
        del sliced
    if teacher is not None:
        teacher = teacher.to(device)
    levels = model.levels
    print("level params (M):", {L: round(v / 1e6, 2) for L, v in nparams.items()}, flush=True)

    # ---------------- data ----------------
    if args.dummy:
        tr = [(torch.randn(args.batch_size, 3, 224, 224), torch.randint(0, 10, (args.batch_size,))) for _ in range(6)]
        va = [(torch.randn(args.batch_size, 3, 224, 224), torch.randint(0, 10, (args.batch_size,))) for _ in range(2)]
        train_loader, val_loader, mix = tr, va, None
    else:
        train_loader, val_loader, _ = load_imagenet(datapath=args.data_path, batch_size=args.batch_size, distributed=False, ra_sampler=False,
                                                    num_workers=args.workers, subset_frac=args.train_subset, subset_seed=args.subset_seed,
                                                    subset_file=os.path.join(args.out, "train_subset.json") if args.train_subset > 0 else None,
                                                    val_workers=args.val_workers)
        mix = mixup_fn
    steps_per_epoch = len(train_loader)

    # ---------------- optimisation ----------------
    levels_per_step = 3 if args.sampler == "sandwich" else 1
    total_steps = int(round(args.total_level_epochs * steps_per_epoch / (levels_per_step if args.sampler == "sandwich" else 1)))
    # sandwich: dedup can make a step cost 2 level-forwards; budget uses the nominal 3
    warm = int(args.warmup_epochs * steps_per_epoch / levels_per_step)
    norm_p = [p for n, p in model.named_parameters() if p.ndim <= 1 or n.endswith(".bias")]
    other_p = [p for n, p in model.named_parameters() if not (p.ndim <= 1 or n.endswith(".bias"))]
    opt = torch.optim.AdamW([dict(params=other_p, weight_decay=args.wd), dict(params=norm_p, weight_decay=args.wd)], lr=args.lr)

    def lr_at(s):
        if s < warm:
            return args.lr * (0.033 + (1 - 0.033) * s / max(1, warm))   # same start factor as main.py (lr_warmup_decay 0.033)
        t = (s - warm) / max(1, total_steps - warm)
        return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    ce_crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ckpt_path = os.path.join(args.out, "ckpt", "last.pt"); os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    step, counts, hist = 0, {L: 0 for L in levels}, []
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        core_module.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]; counts = ck["counts"]; hist = ck["hist"]
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        print("resumed at step", step)

    rng = random.Random(args.seed + 1000)
    for _ in range(step * 1):  # fast-forward the level RNG for exact resume
        sample_levels(args, rng, levels)

    print(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} sampler={args.sampler} loss=ce{ce_w}/kl{kl_w}", flush=True)
    eval_every = int(args.eval_every_level_epochs * steps_per_epoch / levels_per_step) if args.eval_every_level_epochs else 0
    t_start, imgs, done = time.time(), 0, False
    while not done:
        for x, y in train_loader:
            if step >= total_steps:
                done = True; break
            if mix is not None:
                x, y = mix(x, y)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.zero_grad(set_to_none=True)
            lv = sample_levels(args, rng, levels)
            mb = args.micro_batch or x.shape[0]       # gradient accumulation over micro-batches == one big batch (no BatchNorm)
            tot = 0.0
            for L in lv:
                for i0 in range(0, x.shape[0], mb):
                    xc, yc = x[i0:i0 + mb], y[i0:i0 + mb]
                    t_logits = None
                    if teacher is not None:
                        with torch.no_grad(), torch.autocast("cuda", enabled=scaler is not None):
                            t_logits = teacher(xc).logits.float()
                    with torch.autocast("cuda", enabled=scaler is not None):
                        out = model(xc, L).float()
                    loss = (ce_w * ce_crit(out, yc) if ce_w > 0 else 0.0) + (kl_w * kd_loss(out, t_logits, args.temperature) if kl_w > 0 else 0.0)
                    loss = loss * (xc.shape[0] / x.shape[0]) / len(lv)
                    (scaler.scale(loss) if scaler else loss).backward()
                    tot += loss.item()
                counts[L] += 1
            if scaler:
                scaler.unscale_(opt)
            gn = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm) if args.clip_grad_norm else torch.tensor(0.0)
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            step += 1; imgs += x.shape[0] * len(lv)
            if step % args.print_freq == 0 or step == 1:
                print(f"step {step}/{total_steps} lv={lv} loss={tot:.4f} gn={float(gn):.2f} lr={lr_at(step):.2e} {imgs / (time.time() - t_start):.0f} img/s mem={torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f}GB", flush=True)
            if eval_every and step % eval_every == 0 and step < total_steps:
                ev = evaluate_levels(model, val_loader, device, levels, args.val_limit)
                hist.append(dict(step=step, eval=ev)); print("EVAL", step, {L: round(v["top1"], 2) for L, v in ev.items()}, flush=True)
                torch.save(dict(model=core_module.state_dict(), opt=opt.state_dict(), step=step, counts=counts, hist=hist, scaler=scaler.state_dict() if scaler else None), ckpt_path)
        if args.dummy and not done:
            continue

    wall = time.time() - t_start
    ev = evaluate_levels(model, val_loader, device, levels, args.val_limit)
    res = dict(exp=args.exp_name, levels={L: dict(**ev[L], params_M=nparams[L] / 1e6) for L in levels}, mean_top1=sum(v["top1"] for v in ev.values()) / len(ev),
               counts=counts, wall_clock_s=wall, train_img_per_s=imgs / max(wall, 1e-9), steps=total_steps, args=vars(args), hist=hist,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    json.dump(res, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    torch.save(dict(model=core_module.state_dict(), step=step, counts=counts, profiles=getattr(model, "profiles", None)), os.path.join(args.out, "ckpt", "final.pt"))
    print("FINAL top1 per level:", {L: round(v["top1"], 2) for L, v in ev.items()}, "mean", round(res["mean_top1"], 2), flush=True)
    print("sampler coverage:", counts)


if __name__ == "__main__":
    main()
