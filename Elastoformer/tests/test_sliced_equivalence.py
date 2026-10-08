"""S2: sliced super-model at level L == physically pruned (Level 1) / rebuilt (Levels 2..N+1) model. CPU, tiny ViT."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
import torch
from models.elastoformer import ElasticViTForImageClassification, ElasticViTConfig, ElasticViTSelfAttention
from models.sliced_vit import SlicedViT, build_level_indices, slice_state_dict
from utils.nested_metadata import run_nested_pruning, make_importance
from utils.prune_utils import get_vit_info, create_vit_general, update_vit_weights_global


def make(hidden=96, heads=6, layers=2, inter=192, classes=10, seed=0):
    torch.manual_seed(seed)
    cfg = ElasticViTConfig(hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads, intermediate_size=inter,
                           image_size=224, patch_size=16, num_labels=classes, qkv_bias=True)
    cfg.pruned_dim = hidden
    m = ElasticViTForImageClassification(cfg).eval()
    for p in m.parameters():                       # make every tensor (incl. LN/cls/pos) non-trivial
        p.data.add_(0.05 * torch.randn_like(p))
    return m


def run(N=3, CR=0.5, **kw):
    full = make(**kw)
    orig = deepcopy(full)
    core = deepcopy(full)
    meta = run_nested_pruning(core, orig, make_importance("l1"), torch.randn(1, 3, 224, 224), N, CR)
    sd = orig.state_dict()
    groups, nl = build_level_indices(meta["pruned_index_out"], N, sd)
    sliced = SlicedViT(orig, groups, nl, N).eval()
    x = torch.randn(3, 3, 224, 224)
    res = {}
    with torch.no_grad():
        # Level N+1 == original model
        res[N + 1] = (sliced(x, N + 1) - orig(x).logits).abs().max().item()
        # Level 1 == physically pruned core (patch head attrs like main.py does)
        info = get_vit_info(non_pruned_weights=core.state_dict(), num_heads=orig.config.num_attention_heads, core_model=True)
        for m in core.modules():
            if isinstance(m, ElasticViTSelfAttention):
                m.num_attention_heads = info["num_heads"]; m.attention_head_size = m.query.out_features // m.num_attention_heads
                m.all_head_size = m.query.out_features
        res[1] = (sliced(x, 1) - core.eval()(x).logits).abs().max().item()
        # Levels 2..N: HF-module model of the right size loaded with the sliced pretrained state dict (all blocks)
        for L in range(2, N + 1):
            dims = dict(Embed_Dim=len(sliced.idx("E", L)), num_layers=nl, num_heads=orig.config.num_attention_heads,
                        FFN_Intermediate_Dim=len(sliced.idx("F0", L)), FFN_Output_Dim=len(sliced.idx("E", L)), QKV_Dim_out=len(sliced.idx("Q0", L)))
            assert all(len(sliced.idx(f"Q{l}", L)) == dims["QKV_Dim_out"] and len(sliced.idx(f"F{l}", L)) == dims["FFN_Intermediate_Dim"] for l in range(nl))
            ref = create_vit_general(dim_dict=dims, num_classes=orig.config.num_labels)
            ref.load_state_dict(slice_state_dict(sd, sliced, L), strict=True)
            res[L] = (sliced(x, L) - ref.eval()(x).logits).abs().max().item()
            # the published rebuild path only fills shell x shell and core x core blocks: record how far it is from the slice
        # diagonal-block fidelity of update_vit_weights_global (informational): off-diagonal blocks stay at random init
        i = 1; L = N
        pw, npw = meta["pruned_weights_recorder"][f"Level_{L}"], meta["non_pruned_weights_recorder"][f"Level_{L}"]
        info = get_vit_info(pw, npw, num_heads=orig.config.num_attention_heads)
        reb = create_vit_general(dim_dict=info, num_classes=orig.config.num_labels)
        reb, _, _ = update_vit_weights_global(reb, [meta["pruned_index_in"][i], meta["pruned_index_out"][i]],
                                             [meta["non_pruned_index_in"][i], meta["non_pruned_index_out"][i]], pw, npw, device=torch.device("cpu"))
        want = slice_state_dict(sd, sliced, L)["vit.encoder.layer.0.intermediate.dense.weight"]
        got = reb.state_dict()["vit.encoder.layer.0.intermediate.dense.weight"]
        res["rebuild_offdiag_frac_equal_to_pretrained_slice"] = (got == want).float().mean().item()
    # nestedness of keep sets
    for g, per in groups.items():
        for L in range(1, N + 1):
            assert set(per[L].tolist()).issubset(set(per[L + 1].tolist())), (g, L)
    return res


def test_sliced_equals_physical():
    res = run()
    print("max |logit diff| per level:", res)
    assert all(v < 1e-4 for k, v in res.items() if isinstance(k, int)), res


def test_sliced_equals_physical_other_ratio():
    res = run(N=4, CR=0.75, layers=3)
    print("max |logit diff| per level:", res)
    assert all(v < 1e-4 for k, v in res.items() if isinstance(k, int)), res


if __name__ == "__main__":
    test_sliced_equals_physical(); test_sliced_equals_physical_other_ratio(); print("OK")
