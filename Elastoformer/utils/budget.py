"""Budget utilities for the DP profile searches (E5 pruning axis, E6 rank axis).

FlexRank's DP returns the Pareto front of per-layer widths. Thresholding that front (get_profile_for_params) can leave a
large part of the parameter budget unused when the front is coarse (E5: DP spent 37 M of a 61 M budget).

greedy_fill_to_budget starts from the thresholded profile and repeatedly adds one unit of width where the estimated loss
reduction per added parameter is largest, while staying within the budget. Every integer width is a candidate, so the
budget is used almost exactly. Errors between probed widths are linearly interpolated (np.interp) from the DP probes; params
are exact (layer.get_num_parameters). Nothing here calls the model.
"""
import numpy as np


def layer_cost_table(layer_names, stats, layers_by_name):
    """Per layer: dict {width: (params, est_error)} for every integer width 1..max_inner_dim."""
    table = []
    for n in layer_names:
        s = stats[n]
        layer = layers_by_name[n]
        xs = np.asarray(s.inner_dims, dtype=float)
        ys = np.asarray(s.errors, dtype=float)
        rows = {}
        for d in range(1, int(layer.max_inner_dim) + 1):
            rows[d] = (int(layer.get_num_parameters(d)), float(np.interp(d, xs, ys)))
        table.append(rows)
    return table


def greedy_fill_to_budget(table, start_dims, budget):
    """table: per layer {width: (params, err)}. start_dims: starting width per layer. Returns (dims, total_params).
    Never exceeds budget. Repeatedly adds the single-unit step with the best error reduction per parameter."""
    dims = [int(d) for d in start_dims]
    total = sum(table[g][dims[g]][0] for g in range(len(table)))
    if total > budget:                      # DP start above budget: shrink greedily instead (least error increase per param)
        while total > budget:
            best = None
            for g, rows in enumerate(table):
                d = dims[g]
                if d <= 1:
                    continue
                dc = rows[d][0] - rows[d - 1][0]
                if dc <= 0:
                    continue
                loss = rows[d - 1][1] - rows[d][1]
                score = loss / dc
                if best is None or score < best[0]:
                    best = (score, g, dc)
            if best is None:
                break
            _, g, dc = best
            dims[g] -= 1
            total -= dc
        return dims, total
    while True:
        best = None
        for g, rows in enumerate(table):
            d = dims[g]
            if d + 1 not in rows:
                continue
            dc = rows[d + 1][0] - rows[d][0]
            if dc <= 0 or total + dc > budget:
                continue
            gain = rows[d][1] - rows[d + 1][1]
            score = gain / dc
            if best is None or score > best[0]:
                best = (score, g, dc)
        if best is None:            # no feasible step left; zero-gain steps are allowed so the budget is used
            break
        _, g, dc = best
        dims[g] += 1
        total += dc
    return dims, total


def params_of(table, dims):
    return sum(table[g][int(d)][0] for g, d in enumerate(dims))
