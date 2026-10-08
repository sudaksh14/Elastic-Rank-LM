"""Nested pruning-metadata computation, factored out of main.py (lines ~281-413) for auditing/reuse.

This is a *verbatim* port of the main.py logic (same BasePruner arguments, same de-duplication, same
extract_vit_weight_subset calls) with two additions that main.py does not have:
  * raw_out / raw_in: the group index lists *before* de-duplication, so nestedness can be checked.
  * no data/training: only L1/L2/random importance are supported (no gradient accumulation).
main.py is intentionally NOT modified yet; equivalence is verified before it is switched over (plan I-2).
"""
import torch
import torch_pruning as tp
import torch.nn as nn

from models.elastoformer import ElasticViTSelfAttention, ElasticViTSelfOutput
from utils.prune_utils import get_layer_size, get_unpruned_indices, extract_vit_weight_subset


def make_importance(kind):
    if kind == "random":
        return tp.importance.RandomImportance()
    if kind == "l1":
        return tp.importance.GroupMagnitudeImportance(p=1)
    if kind == "l2":
        return tp.importance.GroupMagnitudeImportance(p=2)
    raise NotImplementedError(f"importance '{kind}' needs data-driven gradient accumulation; not supported here")


def run_nested_pruning(model, orig_copy, imp, example_inputs, steps, ratio, bottleneck=False,
                       global_pruning=False, isomorphic=False):
    """Run `steps` iterative pruning rounds exactly as main.py does. `model` is pruned in place (core model)."""
    orig_dimensions = get_layer_size(orig_copy.state_dict())

    num_heads, ignored_layers = {}, [model.classifier]
    for m in model.modules():
        if isinstance(m, ElasticViTSelfAttention):
            num_heads[m.query] = m.num_attention_heads
            num_heads[m.key] = m.num_attention_heads
            num_heads[m.value] = m.num_attention_heads
        if bottleneck and isinstance(m, ElasticViTSelfOutput):
            ignored_layers.append(m.dense)

    pruner = tp.pruner.BasePruner(
        model, example_inputs, isomorphic=isomorphic, global_pruning=global_pruning, importance=imp,
        iterative_steps=steps, pruning_ratio=ratio, ignored_layers=ignored_layers, round_to=1,
        num_heads=num_heads, prune_num_heads=False, prune_head_dims=True, head_pruning_ratio=ratio,
        head_pruning_ratio_dict=None, output_transform=lambda out: out.logits.sum())

    pruned_index_out = [{} for _ in range(steps)]
    pruned_index_in = [{} for _ in range(steps)]
    non_pruned_index_out = [{} for _ in range(steps)]
    non_pruned_index_in = [{} for _ in range(steps)]
    raw_out = [{} for _ in range(steps)]   # NEW: before de-dup
    raw_in = [{} for _ in range(steps)]    # NEW: before de-dup
    pruned_weights_recorder, non_pruned_weights_recorder = {}, {}

    for i in range(steps):
        for grp in pruner.step(interactive=True):
            for dep, idxs in grp:
                layer = dep.target.module
                target_layer_name = dep.target._name
                handler = dep.handler.__name__

                if isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Conv2d)):
                    if handler == "prune_out_channels":
                        raw_out[i][target_layer_name] = list(idxs)
                        for j in range(i):
                            if target_layer_name in pruned_index_out[j].keys() and set(pruned_index_out[j][target_layer_name]).issubset(set(idxs)):
                                idxs = [item for item in idxs if item not in pruned_index_out[j][target_layer_name]]
                        pruned_index_out[i][target_layer_name] = idxs
                        idxs_non = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs)
                        for j in range(i):
                            if target_layer_name in (non_pruned_index_out[j].keys() | pruned_index_out[j].keys()):
                                idxs_non = [item for item in idxs_non if item not in pruned_index_out[j][target_layer_name]]
                        non_pruned_index_out[i][target_layer_name] = idxs_non

                    elif handler == "prune_in_channels":
                        raw_in[i][target_layer_name] = list(idxs)
                        for j in range(i):
                            if target_layer_name in pruned_index_in[j].keys() and set(pruned_index_in[j][target_layer_name]).issubset(set(idxs)):
                                idxs = [item for item in idxs if item not in pruned_index_in[j][target_layer_name]]
                        pruned_index_in[i][target_layer_name] = idxs
                        idxs_non = get_unpruned_indices(orig_dimensions[f"{target_layer_name}.weight"], idxs, dim="in")
                        for j in range(i):
                            if target_layer_name in (non_pruned_index_in[j].keys() | pruned_index_in[j].keys()):
                                idxs_non = [item for item in idxs_non if item not in pruned_index_in[j][target_layer_name]]
                        non_pruned_index_in[i][target_layer_name] = idxs_non

            if (i + 1) == steps:
                grp.prune()

        level = steps + 1 - i
        pruned_weights_recorder[f"Level_{level}"] = extract_vit_weight_subset(orig_copy, pruned_index_out[i], pruned_index_in[i])
        non_pruned_weights_recorder[f"Level_{level}"] = extract_vit_weight_subset(orig_copy, non_pruned_index_out[i], non_pruned_index_in[i])

    return dict(pruned_index_out=pruned_index_out, pruned_index_in=pruned_index_in,
                non_pruned_index_out=non_pruned_index_out, non_pruned_index_in=non_pruned_index_in,
                raw_out=raw_out, raw_in=raw_in,
                pruned_weights_recorder=pruned_weights_recorder,
                non_pruned_weights_recorder=non_pruned_weights_recorder)
