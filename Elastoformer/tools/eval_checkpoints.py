"""E1: evaluate the published Elastoformer DeiT-B ladders and audit nestedness of the saved weights.

Rebuild rule (matches main.py): dims are read from the state dict (get_vit_info core_model=True) and the model is built
with create_vit_general; head count is 12 for every level (verified in E0: get_num_heads -> 12).
Eval transform matches datasets.build_transform(is_train=False): resize 256 (bicubic) -> center crop 224 -> ImageNet mean/std.
Runs on CPU or GPU; results.json is written after every level so a killed job loses nothing.
"""
import os, sys, json, time, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torchvision.datasets as tvd
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch_pruning as tp

from utils.prune_utils import get_vit_info, create_vit_general

# Elastoformer paper Table 1 (ViT-B/DeiT-B on ImageNet, base 81.74 % / 16.86 GFLOPs)
PAPER = {
    "0.4": dict(acc=[79.95, 80.13, 80.24, 80.35, 80.20, 80.13], gflops=[5.70, 7.58, 9.57, 11.38, 13.99, 16.86]),
    "0.6": dict(acc=[72.45, 74.22, 75.26, 75.79, 76.09, 76.41], gflops=[2.64, 4.44, 6.82, 9.57, 12.94, 16.86]),
    "0.8": dict(acc=[51.19, 64.14, 69.97, 72.46, 73.61, 74.65], gflops=[0.60, 1.96, 4.44, 7.58, 11.38, 16.86]),
}
FILES = ["Vit_b_16_Core_Level_1.pth"] + [f"Vit_b_16_Rebuilt_Level_{k}.pth" for k in range(2, 7)]


def load_sd(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and not any(k.startswith("vit.") for k in sd):
        sd = sd["model"]
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def build(sd, heads=12):
    info = get_vit_info(non_pruned_weights=sd, num_heads=heads, core_model=True)
    m = create_vit_general(dim_dict=info, num_classes=sd["classifier.weight"].shape[0])
    missing, unexpected = m.load_state_dict(sd, strict=False)
    return m.eval(), info, list(missing), list(unexpected)


def make_loader(root, bs, workers, limit):
    tf = T.Compose([T.Resize(int(224 / 0.875), interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    ds = tvd.ImageFolder(os.path.join(root, "val"), transform=tf)
    assert len(ds.classes) == 1000, len(ds.classes)
    if limit:  # fixed, class-spread subset for smoke tests
        g = torch.Generator().manual_seed(0)
        ds = Subset(ds, torch.randperm(len(ds), generator=g)[:limit].tolist())
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available()), len(ds)


@torch.inference_mode()
def evaluate(model, loader, device, n_total, tag):
    crit = nn.CrossEntropyLoss(reduction="sum")
    c1 = c5 = n = 0
    loss = 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        out = model(x).logits
        loss += crit(out, y).item()
        top5 = out.topk(5, 1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += y.numel()
        if i % 20 == 0:
            print(f"  [{tag}] {n}/{n_total}  top1={100 * c1 / n:.2f}  {n / (time.time() - t0):.1f} img/s", flush=True)
    return dict(top1=100 * c1 / n, top5=100 * c5 / n, loss=loss / n, n=n, eval_s=time.time() - t0)


def nestedness(sds):
    """value-level containment: fraction of each small tensor's entries that also occur (bitwise) in the next-larger level."""
    probe = ["vit.embeddings.cls_token", "vit.embeddings.position_embeddings", "vit.embeddings.patch_embeddings.projection.weight",
             "vit.encoder.layer.0.attention.attention.query.weight", "vit.encoder.layer.0.intermediate.dense.weight",
             "vit.encoder.layer.0.output.dense.weight", "vit.encoder.layer.6.attention.attention.value.weight",
             "vit.encoder.layer.11.output.dense.weight", "vit.layernorm.weight", "vit.encoder.layer.0.layernorm_before.weight",
             "classifier.weight", "classifier.bias"]
    res = {}
    for k in range(len(sds) - 1):
        small, big = sds[k], sds[k + 1]
        row = {}
        for name in probe:
            if name in small and name in big and small[name].numel() > 0:
                a, b = small[name].flatten().float(), big[name].flatten().float()
                row[name] = dict(small_shape=list(small[name].shape), big_shape=list(big[name].shape),
                                 frac_contained=torch.isin(a, b).float().mean().item(),
                                 same_shape_and_equal=bool(small[name].shape == big[name].shape and torch.equal(small[name], big[name])))
        res[f"Level_{k + 1}_in_Level_{k + 2}"] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", default="/var/scratch/skalra/elastoSLM/checkpoints")
    ap.add_argument("--crs", nargs="+", default=["0.4", "0.6", "0.8"])
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--data_path", default="/var/scratch/dchabal/quokka/data/imagenet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="evaluate a fixed random subset (smoke test only)")
    ap.add_argument("--skip_eval", action="store_true", help="nestedness/params only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    loader, n_total = (None, 0) if args.skip_eval else make_loader(args.data_path, args.batch_size, args.workers, args.limit)
    ex = torch.randn(1, 3, 224, 224)
    path = os.path.join(args.out, "results.json")
    results = {"args": vars(args), "torch": torch.__version__, "device": str(device), "crs": {}}

    for cr in args.crs:
        d = os.path.join(args.ckpt_root, f"deitb_cr{cr}")
        sds, rows = [], []
        for k in args.levels:
            f = os.path.join(d, FILES[k - 1])
            sd = load_sd(f)
            sds.append(sd)
            m, info, missing, unexpected = build(sd)
            macs, params = tp.utils.count_ops_and_params(m, ex)
            row = dict(level=k, file=os.path.basename(f), dims={x: info[x] for x in ("Embed_Dim", "QKV_Dim_out", "FFN_Intermediate_Dim", "num_heads")},
                       params_M=params / 1e6, gflops=macs / 1e9, missing=missing[:5], unexpected=unexpected[:5],
                       paper_acc=PAPER[cr]["acc"][k - 1], paper_gflops=PAPER[cr]["gflops"][k - 1],
                       file_MB=os.path.getsize(f) / 2**20)
            if missing or unexpected:
                print(f"!! CR={cr} L{k}: missing={missing[:3]} unexpected={unexpected[:3]}", flush=True)
            if not args.skip_eval:
                m = m.to(device)
                row.update(evaluate(m, loader, device, n_total, f"CR{cr}-L{k}"))
                row["delta_vs_paper_pp"] = row["top1"] - row["paper_acc"]
                m.cpu()
            print(f"CR={cr} L{k} dims={row['dims']} GFLOPs={row['gflops']:.2f}(paper {row['paper_gflops']}) "
                  f"top1={row.get('top1', float('nan')):.2f}(paper {row['paper_acc']})", flush=True)
            rows.append(row)
            results["crs"][cr] = dict(levels=rows)
            json.dump(results, open(path, "w"), indent=2)
        results["crs"][cr]["nestedness"] = nestedness(sds)
        json.dump(results, open(path, "w"), indent=2)
    print("E1 DONE ->", path)


if __name__ == "__main__":
    main()
