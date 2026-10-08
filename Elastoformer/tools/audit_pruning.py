"""E0: CPU-only audit of Elastoformer's pruning metadata and rebuild path (no data, no training).

Answers (see ELASTOFORMER_IMPLEMENTATION_PLAN.md, E0):
  H0.1 cumulative pruned fraction per round: linear CR*(i+1)/N  vs  geometric 1-(1-CR)^((i+1)/N)
  H0.2 raw (pre-dedup) index sets nested across rounds; shells disjoint
  H0.3 head-dim pruning uniform across heads; head count/size chosen per level (get_num_heads)
  H0.4 rebuilt Level-2 carries pretrained cls_token / position_embeddings slices?  (+ what params metadata skips)
"""
import os, sys, json, argparse, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from models.elastoformer import ElasticViTForImageClassification, ElasticViTConfig
from utils.prune_utils import get_vit_info, create_vit_general, update_vit_weights_global
from utils.nested_metadata import run_nested_pruning, make_importance
from copy import deepcopy

torch.manual_seed(0)


def find(names, pat):
    r = re.compile(pat)
    return [n for n in names if r.fullmatch(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/deit-base-patch16-224")
    ap.add_argument("--pruning_ratio", type=float, default=0.8)
    ap.add_argument("--pruning_steps", type=int, default=5)
    ap.add_argument("--pruning_type", default="l1")
    ap.add_argument("--out", default="outputs/E0")
    ap.add_argument("--tiny", action="store_true", help="random tiny ViT (logic smoke test, no download)")
    args = ap.parse_args()
    N, CR = args.pruning_steps, args.pruning_ratio
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    if args.tiny:
        cfg = ElasticViTConfig(hidden_size=96, num_hidden_layers=2, num_attention_heads=6, intermediate_size=192,
                               image_size=224, patch_size=16, num_labels=10, qkv_bias=True)
        cfg.pruned_dim = 96
        model = ElasticViTForImageClassification(cfg)
        heads_override, tag = 6, "tiny"
    else:
        model = ElasticViTForImageClassification.from_pretrained(args.model_name)
        heads_override, tag = None, args.model_name.replace("/", "_")
    model.eval()
    num_classes = model.config.num_labels
    orig = deepcopy(model)
    orig_sd = orig.state_dict()
    H0 = orig.config.num_attention_heads
    hidden = orig.config.hidden_size
    report = dict(model=args.model_name if not args.tiny else "tiny", CR=CR, N=N, heads=H0, hidden=hidden,
                  torch=torch.__version__)

    imp = make_importance(args.pruning_type)
    meta = run_nested_pruning(model, orig, imp, torch.randn(1, 3, 224, 224), N, CR)
    pout, pin = meta["pruned_index_out"], meta["pruned_index_in"]
    rout, rin = meta["raw_out"], meta["raw_in"]
    out_names = sorted(set().union(*[set(d) for d in pout]))

    # ---------------- H0.1: cumulative pruned fraction per round ----------------
    def total_out(name):
        return orig_sd[f"{name}.weight"].shape[0]

    reps = {
        "embed(patch_proj)": "vit.embeddings.patch_embeddings.projection",
        "qkv(layer0.query)": "vit.encoder.layer.0.attention.attention.query",
        "ffn(layer0.intermediate)": "vit.encoder.layer.0.intermediate.dense",
    }
    h01 = {}
    for label, name in reps.items():
        tot = total_out(name)
        cum, row = 0, []
        for i in range(N):
            cum += len(pout[i].get(name, []))
            row.append(cum / tot)
        h01[label] = dict(total=tot, cum_pruned_frac=row)
    exp_lin = [CR * (i + 1) / N for i in range(N)]
    exp_geo = [1 - (1 - CR) ** ((i + 1) / N) for i in range(N)]
    # spread across *all* layers' out dims, per round (uniformity of the schedule across layers)
    spread = []
    for i in range(N):
        fr = []
        for name in out_names:
            cum = sum(len(pout[j].get(name, [])) for j in range(i + 1))
            fr.append(cum / total_out(name))
        spread.append(dict(round=i, min=min(fr), max=max(fr)))
    per_dim_verdict = {}
    for label, v in h01.items():
        el = max(abs(a - b) for a, b in zip(v["cum_pruned_frac"], exp_lin))
        eg = max(abs(a - b) for a, b in zip(v["cum_pruned_frac"], exp_geo))
        per_dim_verdict[label] = dict(max_err_linear=el, max_err_geometric=eg, closer_to="linear" if el < eg else "geometric")
    err_lin = max(v["max_err_linear"] for v in per_dim_verdict.values())
    err_geo = max(v["max_err_geometric"] for v in per_dim_verdict.values())
    report["H0.1"] = dict(per_dim=h01, expected_linear=exp_lin, expected_geometric=exp_geo,
                          max_abs_err_vs_linear=err_lin, max_abs_err_vs_geometric=err_geo,
                          per_dim_verdict=per_dim_verdict, spread_over_layers=spread,
                          verdict=(sorted({v["closer_to"] for v in per_dim_verdict.values()}) if len({v["closer_to"] for v in per_dim_verdict.values()}) > 1 else per_dim_verdict["embed(patch_proj)"]["closer_to"]))

    # ---------------- H0.2: nestedness / disjointness ----------------
    viol_nested, viol_disjoint = [], []
    for name in out_names:
        for i in range(1, N):
            if name in rout[i] and name in rout[i - 1] and not set(rout[i - 1][name]).issubset(set(rout[i][name])):
                viol_nested.append(f"out:{name}@{i}")
    for name in sorted(set().union(*[set(d) for d in pin])):
        for i in range(1, N):
            if name in rin[i] and name in rin[i - 1] and not set(rin[i - 1][name]).issubset(set(rin[i][name])):
                viol_nested.append(f"in:{name}@{i}")
    for d in (pout, pin):
        for name in sorted(set().union(*[set(x) for x in d])):
            seen = set()
            for i in range(N):
                s = set(d[i].get(name, []))
                if seen & s:
                    viol_disjoint.append(f"{name}@{i}")
                seen |= s
    report["H0.2"] = dict(nested_violations=len(viol_nested), disjoint_violations=len(viol_disjoint),
                          examples=(viol_nested + viol_disjoint)[:10], verdict=not viol_nested and not viol_disjoint)

    # ---------------- H0.3: head uniformity + head counts per level ----------------
    qn = find(out_names, r"vit\.encoder\.layer\.\d+\.attention\.attention\.(query|key|value)")
    hd = hidden // H0
    bad_uniform, qkv_mismatch = [], []
    for i in range(N):
        cum = {}
        for name in qn:
            cum[name] = sorted(sum((pout[j].get(name, []) for j in range(i + 1)), []))
            cnt = torch.bincount(torch.tensor([x // hd for x in cum[name]], dtype=torch.long), minlength=H0)
            if cnt.min() != cnt.max():
                bad_uniform.append(f"{name}@{i}:{cnt.tolist()}")
        for L in sorted({n.split(".")[3] for n in qn}):
            q, k, v = (cum[f"vit.encoder.layer.{L}.attention.attention.{t}"] for t in ("query", "key", "value"))
            if not (q == k == v):
                qkv_mismatch.append(f"layer{L}@{i}")
    report["H0.3"] = dict(head_dim_orig=hd, nonuniform_per_head=len(bad_uniform), examples=bad_uniform[:5],
                          qkv_index_mismatch=len(qkv_mismatch), qkv_examples=qkv_mismatch[:5],
                          verdict=not bad_uniform and not qkv_mismatch)

    # ---------------- levels as rebuilt by main.py: head count / dims ----------------
    levels = {}
    core_info = get_vit_info(non_pruned_weights=model.state_dict(), num_heads=orig.config.num_attention_heads, core_model=True)
    levels["Level_1(core)"] = core_info
    lvl_err = None
    try:
        for i in range(N):
            lv = N + 1 - i
            info = get_vit_info(meta["pruned_weights_recorder"][f"Level_{lv}"], meta["non_pruned_weights_recorder"][f"Level_{lv}"],
                                num_heads=heads_override)
            info["head_size"] = info["QKV_Dim_out"] // info["num_heads"]
            levels[f"Level_{lv}(full-dim,shell {i} added)"] = info
    except Exception as e:  # get_num_heads can raise on odd dims
        lvl_err = repr(e)
    report["levels_rebuild_info"] = dict(levels=levels, error=lvl_err)

    # ---------------- params the metadata does not cover ----------------
    covered = set()
    for name in set(out_names) | set().union(*[set(d) for d in pin]):
        covered |= {f"{name}.weight", f"{name}.bias"}
    uncovered = [(n, tuple(p.shape)) for n, p in orig.named_parameters() if n not in covered]
    report["uncovered_params"] = uncovered

    # ---------------- metadata size ----------------
    wbytes = sum(a.nbytes for lvl in meta["pruned_weights_recorder"].values() for d in lvl.values() for a in d.values() if a is not None)
    ibytes = sum(8 * len(v) for d in (pout, pin, meta["non_pruned_index_out"], meta["non_pruned_index_in"]) for r in d for v in r.values())
    report["metadata_MB"] = dict(pruned_weight_values=wbytes / 2**20, indices_int64=ibytes / 2**20)

    # ---------------- H0.4: rebuilt Level-2 vs pretrained slices ----------------
    h04 = {}
    try:
        reb_w, pr_w = meta["non_pruned_weights_recorder"]["Level_2"], meta["pruned_weights_recorder"]["Level_2"]
        info = get_vit_info(pr_w, reb_w, num_heads=heads_override)
        rebuilt = create_vit_general(dim_dict=info, num_classes=num_classes)
        i = N - 1
        rebuilt, _, _ = update_vit_weights_global(rebuilt, [pin[i], pout[i]],
                                                 [meta["non_pruned_index_in"][i], meta["non_pruned_index_out"][i]],
                                                 pr_w, reb_w, device=torch.device("cpu"))
        proj = "vit.embeddings.patch_embeddings.projection"
        emb_idx = sorted(set(pout[i][proj]) | set(meta["non_pruned_index_out"][i][proj]))
        sl = torch.tensor(emb_idx)

        def cmp(a, b):
            a, b = a.detach().flatten().float(), b.detach().flatten().float()
            return dict(max_abs_diff=(a - b).abs().max().item(), cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item())

        r_sd = rebuilt.state_dict()
        h04["level2_embed_dim"] = len(emb_idx)
        h04["cls_token"] = cmp(r_sd["vit.embeddings.cls_token"], orig_sd["vit.embeddings.cls_token"][..., sl])
        h04["position_embeddings"] = cmp(r_sd["vit.embeddings.position_embeddings"], orig_sd["vit.embeddings.position_embeddings"][..., sl])
        h04["patch_proj_weight(sanity,should_match)"] = cmp(r_sd[f"{proj}.weight"], orig_sd[f"{proj}.weight"][sl])
        h04["classifier_weight"] = cmp(r_sd["classifier.weight"], orig_sd["classifier.weight"][:, sl])
        h04["verdict_pos_cls_pretrained"] = bool(h04["cls_token"]["max_abs_diff"] < 1e-6 and h04["position_embeddings"]["max_abs_diff"] < 1e-6)
    except Exception as e:
        h04["error"] = repr(e)
    report["H0.4"] = h04

    report["wall_clock_s"] = time.time() - t0
    path = os.path.join(args.out, f"audit_{tag}_cr{CR}_N{N}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[E0] {tag} CR={CR} N={N}")
    print("  H0.1 per-round cum pruned (embed):", [round(x, 4) for x in h01["embed(patch_proj)"]["cum_pruned_frac"]])
    print("       expected linear   :", [round(x, 4) for x in exp_lin])
    print("       expected geometric:", [round(x, 4) for x in exp_geo], "->", report["H0.1"]["verdict"])
    print("  H0.2 nested/disjoint ok:", report["H0.2"]["verdict"], f"({len(viol_nested)} nested, {len(viol_disjoint)} disjoint violations)")
    print("  H0.3 head-uniform & qkv-consistent:", report["H0.3"]["verdict"])
    print("  levels:", {k: (v["Embed_Dim"], v["QKV_Dim_out"], v["FFN_Intermediate_Dim"], v["num_heads"]) for k, v in levels.items()}, lvl_err or "")
    print("  uncovered params:", [n for n, _ in uncovered][:6])
    print("  H0.4:", json.dumps(h04)[:600])
    print("  metadata MB:", report["metadata_MB"])
    print("  wrote", path)


if __name__ == "__main__":
    main()
