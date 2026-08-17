# Elastoformer × FlexRank: Design Brainstorm

Scope: (1) can nested low-rank replace structured pruning in Elastoformer, (2) what else is worth
changing, (3) extending Elastoformer to LMs, (4) model-family experiments in FlexRank.
No implementation — design notes and open questions only.

---

## 0. What each side actually does

**Elastoformer** (`main.py` → `utils/prune_utils.py` → `utils/partial_freezing.py`)

- Compress: N rounds of L1-row-saliency structured pruning via torch-pruning DepGraph, uniform
  ratio per round `p = 1 - (1-CR)^(1/N)`, applied to MHA projections, FFN, LayerNorm and patch
  embedding (i.e. it also shrinks the residual width).
- Metadata: per-round `pruned_index_in/out`, `non_pruned_index_in/out`, plus the pruned weight
  values (~1 MB/DN).
- Grow: reinsert a round's weights, freeze the inherited prefix via gradient-masking hooks
  (`freeze_*_params`, `zero_out_gradients_v{1,2,3}`), selective grad clipping on new weights only,
  retrain 50 epochs. Repeat N times → ~10N GPU-h on ImageNet.
- Select: offline lookup table (acc, latency, MB, FLOPs) + argmin-FLOPs under constraints.

**FlexRank** (`flexrank/layers` → `flexrank/profiles/dp.py` → `flexrank/samplers` → distillation)

- Decompose: per-layer `W ≈ U V^T` via **DataSVD** — minimizes output error `E‖(W−UV^T)x‖²` over a
  calibration set (~10³ samples), not weight error.
- Search: DP over per-layer `(Δcost, Δerror)` candidates, `O(L·K)`, under an additive-error
  assumption, with dominance pruning and a **nestedness filter** → a Pareto front of rank profiles.
- Train: sample a budget per step, distill from the frozen un-decomposed base
  (`ce=0, kl=1` by default in `scripts/flexrank_nlp.sh`). No freezing of the prefix.
- Deploy: GAR reparametrization, `O((m+n−r)r)` per layer, avoids storing/multiplying the `r×r` block.

The two papers are the same story told on two different axes. The theory in FlexRank §4 is
about which *training schedule* recovers the Pareto front; the decomposition is almost incidental.

---

## 1. Can low-rank replace pruning in Elastoformer? — Yes, and it removes most of the codebase

### 1.1 The structural argument

Elastoformer's hardest engineering problem is **index bookkeeping**: nestedness in pruning space is
an explicit set-membership relation that has to be recorded, deduplicated across rounds
(`idxs = [i for i in idxs if i not in pruned_index_out[j][name]]`), remapped
(`merge_and_remap_indices`), and replayed at every growth step.

In rank space nestedness is a **prefix of an ordered axis**. DN level `L` is `U[:, :r_L], V[:, :r_L]`.
Consequences:

| | pruning (today) | nested low-rank |
|---|---|---|
| per-DN metadata | index dicts + weight values, ~1 MB | one integer per layer (a profile vector) |
| mode switch | rebuild/repopulate tensors, ~50.4 ms measured | set `inner_dim`, ~0 |
| freezing | gradient hooks per layer/param | not needed (see 1.2) |
| ordering | discrete, no natural order | SVD gives it for free |

Code that would disappear: most of `prune_utils.py` (1696 lines of index/shape surgery —
`get_vit_info`, `extract_vit_weight_subset`, `update_vit_weights_global`, `get_layer_dimensions`),
`partial_freezing.py`, the three `zero_out_gradients` variants, and the per-level
`ElasticViTConfig` reconstruction in `models/elastoformer.py`. Replaced by: an `ODLinear`-style
layer with an `inner_dim` setter, a decomposition pass, a profile object, a sampler.

### 1.2 The real conflict: partial weight freezing vs nested training

This is the most important finding and it is **independent of the compression axis**.

Elastoformer grows *greedily and sequentially*: train core → freeze → train Δ → freeze → …
FlexRank Thm 4.1–4.3 characterizes exactly this family of schedules:

- PTS (train full, select post-hoc) has zero-measure chance of optimal submodels.
- ASL (train all subspaces) has strictly positive optimality gap — submodels interfere.
- NSL (train only nested submodels, **jointly**) recovers the front.

Elastoformer's frozen-prefix schedule is a *sequential greedy approximation* of NSL. The core is
optimized against the core objective alone and is never revised in light of the larger DNs. The
symptom is visible in Elastoformer Table 1: at CR = 0.8 the ladder starts at 51.19% and only reaches
74.65% at Level-6, never recovering the 81.74% base — a bad frozen core poisons every descendant.
FlexRank's ViT curves stay within ~5% of full down to 30% params.

**So the highest-value change may not be low-rank at all — it is dropping the hard freeze in favour
of joint budget sampling.** Note the freeze is not what buys the memory saving; weight *sharing* is.
Freezing only guarantees lower DNs don't drift, and joint sampling gives that too, by construction.

> **Ablation worth running on its own:** frozen-prefix growth vs jointly-sampled nested training, at
> matched total compute, same backbone, same compression axis. If joint sampling wins, that is a
> clean standalone result and it also fixes the CR=0.8 collapse.

Secondary win: design time goes from `O(N)` (10N GPU-h, one retraining per DN) to `O(1)` — a single
distillation run covers every budget. For an EdgeAI venue that is a headline number.

### 1.3 Where it gets uncomfortable — memory

Elastoformer's flagship claim is 336 MB vs 843 MB (bag-of-models). Full-rank factors of a `d×d`
matrix cost `2d²` — a **2× overhead**, which FlexRank openly acknowledges. Naively swapping the axis
would turn a 1.02× memory story into a 2× one and destroy the paper's main selling point.

Fix, and it is arguably a *better* story: **cap the maximum rank**. Store only `(m+n)·r_max`. At
`r_max = 0.3·d` for a square layer that is `0.6d² < d²` — smaller than the original model at *every*
mode, elasticity included. The trade is that the top DN no longer reproduces the pretrained network
exactly (today Elastoformer's Level-N does). Whether that matters depends on whether the target
deployment ever has budget for the uncompressed model; on Jetson-class hardware it usually does not.

Be explicit about this trade in any writeup — it is the first thing a reviewer will hit.

### 1.4 Where it gets uncomfortable — latency on edge hardware

Elastoformer's contribution is *measured* latency on Orin/Nano at batch size 1. FlexRank's savings
are demonstrated where "computation is the bottleneck" for "sufficiently large matrices and
sequences." These are not the same regime. Low-rank replaces one GEMM with two: more kernel
launches, lower arithmetic intensity, and it breaks fused QKV. **At B=1 on Jetson, a rank-r layer
may be slower than a structurally-pruned dense layer at equal FLOPs.**

This is the central empirical risk — and also the most interesting question you could answer, since
nobody has measured it: *does nested low-rank elasticity survive contact with edge hardware, or is
its Pareto advantage a datacenter-scale artifact?* A negative result is publishable at SEC/EMDL.

Important nuance that cuts the other way for LMs: **autoregressive decode is weight-bandwidth-bound**,
so a parameter reduction translates almost 1:1 into TPOT speedup, whereas prefill is compute-bound
and sees only the FLOP reduction. Low-rank should look *better* at B=1 LM decode than at ViT
inference. Report TTFT and TPOT separately — neither paper does.

Mitigations to consider:
- **Fuse at switch time.** Materialize `W_r = U_r V_r^T` once per mode → dense kernels, but you pay
  switch latency and must hold factors + dense.
- **Hybrid axis.** Low-rank only where the DP wants `r/min(m,n) ≲ 0.3` (typically deep FFN); keep
  structured pruning where it wants near-full rank. Elastoformer already has the pruning machinery.
- **Head-granular rank** so QKV fusion survives.

### 1.5 Latent-space stability — the strongest argument for the swap

Rank truncation is a **projection onto a nested subspace**: the latent space of a smaller submodel is
literally a subspace of the larger one, so the family forms a monotone filtration. Pruning gives no
such guarantee — index selection can reallocate semantics between modes, so features are not
comparable across DNs.

Two things fall out that neither paper claims:

1. **Cross-mode artefact reuse.** A KV cache, feature bank, or retrieval index built at one mode is
   meaningfully valid at another. That enables **mode switching mid-generation** without recomputing
   the cache — Elastoformer today can only switch *between* inferences (~50 ms ≈ one full cycle).
   Needs care (truncating `W_V`'s rank moves the value subspace), but if the V-factors are nested,
   cached vectors at higher rank contain the lower-rank ones as a projection. Worth checking early;
   it is the difference between "elastic model" and "elastic *serving system*".
2. **Explicit alignment loss.** FlexRank distils logits only. Add a cross-budget hidden-state
   consistency term (feature-level KD, or CKA/Procrustes alignment between DN_i and DN_N at matched
   depth). Directly targets the stable-latent-space objective and should also stabilize (1).

---

## 2. Other improvements, roughly by value/effort

**Do regardless of axis**
1. **Distil from the base model.** Elastoformer retrains on labels; the pretrained teacher is sitting
   right there and is free. Its own future-work section already asks for this.
2. **Activation-aware saliency.** L1 on weight rows → `‖W‖·‖x‖`-style (Wanda) or DataSVD-style output
   error. Near-zero cost, reliably better, and it is the same calibration data either way.
3. **DP profile search, decoupled from low-rank.** `p = 1-(1-CR)^(1/N)` uniform-per-round is exactly
   the "automated search for optimal DN count and ratios" Elastoformer lists as future work. The DP
   in `profiles/dp.py` works on any per-layer `(Δcost, Δerror)` probe — including pruning ratios.
   Lowest-risk, highest-certainty import. FlexRank Fig. 6 shows the allocation is very non-uniform
   across depth (mid-layer attention `c_proj` truncated last), so uniform allocation is leaving
   accuracy on the table.
4. **Validate the additive-error assumption at scale.** FlexRank checks it on a 4-layer MNIST net
   (App C.3). Sample random profiles at ViT-B / 8B scale and check rank correlation between predicted
   and actual error. Cheap, and reviewers will ask.

**Composition**
5. Rank for FFN + depth/layer-skip for coarse budget steps + quantization orthogonally. FlexRank
   states these are complementary but does not combine them. Watch INT8 × low-rank: factor
   conditioning after truncation is not obviously quantization-friendly.

**Adaptivity — the thesis-aligned direction**
6. **Input-dependent profile selection.** FlexRank explicitly defers "adaptive routing policies";
   Elastoformer selects on system state only. A small controller (prompt features, or early-layer
   statistics) picking a profile from the Pareto front per query is the exact intersection of both
   papers with input-dependent compute. Metric: does per-query allocation beat the best *fixed*
   profile at matched average FLOPs? This only works if switching is ~free — which favours rank over
   pruning, tying back to §1.5.

**Hazards**
7. **BN statistics for the CNN branch.** Joint budget sampling reintroduces the switchable-BN problem
   from Slimmable NN. LN-based ViTs/LLMs dodge it, which is why FlexRank never hits it — but
   Elastoformer's ResNet/VGG path will. Per-profile BN stats or a recalibration pass.

---

## 3. Extending Elastoformer to language models

Blockers in the current pruning design, and why they argue for the rank axis:

- **RoPE.** Rotary embeddings act on `(2i, 2i+1)` pairs within `head_dim`. Elastoformer prunes head
  *dimensions* (`prune_head_dims=True`, `prune_num_heads=False`) — that destroys the rotation pairing.
  Low-rank factorization preserves `head_dim` exactly, so RoPE is untouched. This alone is close to
  decisive.
- **GQA/MQA.** Llama/Qwen share K/V heads across query groups; pruning head dims breaks the grouping
  invariants. Same resolution.
- **Residual width.** Elastoformer shrinks LayerNorm and patch-embedding dims, i.e. the residual
  stream. In an LM that breaks tied embeddings and every skip connection. **Do not port this.**
  Restrict to per-layer projections, and treat embeddings / `lm_head` as excluded — FlexRank already
  has `decomposition.exclude_layers_names` for this. Note `lm_head` is a large parameter fraction in
  ≤1B models, so "% params compressed" numbers must state whether it is included.
- **Evaluation.** Perplexity + `lm-eval-harness` commonsense; FlexRank already wires this
  (`flextrain/callbacks/lm_eval_callback.py`, `config/lm_eval/commonsense.yaml`). Add IFEval if you
  touch instruct checkpoints.
- **Data.** `finewebedu_10bt_{gpt2,llama1b,llama3b,llama8b}` configs already exist with
  tokenizer-specific variants.
- **Serving metrics.** TTFT vs TPOT separately, plus switch-latency mid-generation and peak memory.
  Elastoformer's edge-measurement discipline is the thing FlexRank lacks — that is the contribution
  to protect when porting.

Pragmatic route: rather than reimplementing decomposition inside Elastoformer, treat `flexrank`
(the core package — it is deliberately standalone, no `flextrain` dependency) as the compression
backend and keep Elastoformer's runtime monitor, profiling lookup table, and edge deployment path as
the contribution layer on top.

---

## 4. Model-family experiments in FlexRank

Already present: `gpt2`, `llama1b/3b/8b`, DINOv3 `vit_base/large/huge/7b`. Axes worth adding:

1. **Within-family scaling.** Qwen3 (0.6/1.7/4/8B), Gemma 3 (1/4/12B), SmolLM3, plus existing Llama.
   Question: does the accuracy retained at budget β improve with model scale? Fig. 4 hints yes.
   A scaling-law-style fit of `retention(β, params)` across a family is a clean contribution and
   directly informs "which model should I start from for a given edge budget."
2. **Attention-architecture axis.** MHA (GPT-2) vs GQA (Llama/Qwen) vs MoE (Qwen3-MoE, OLMoE) vs
   hybrid SSM (Mamba-2, Jamba). MoE raises a genuine design question: nest *within* experts, or nest
   over expert count? For hybrids, does the DP systematically move budget away from SSM blocks?
3. **Distilled vs natively-trained siblings.** Llama-3.2-1B is distilled from 8B. Hypothesis: a
   distilled model has less exploitable redundancy and therefore *worse* rank-elasticity than a
   natively-trained model of the same size. Very testable, and it says something real about where
   elastic capacity comes from.
4. **Base vs instruct checkpoints.** Does knowledge consolidation erode instruction-following or
   safety tuning? (IFEval + a refusal set.) Nobody checks this for compression-elasticity work, and
   it is exactly the question a deployment team asks.
5. **Multimodal — the direct bridge to your thesis.** Qwen2.5-VL, SmolVLM, LLaVA-class. The DP search
   is budget-allocating across a *heterogeneous* model: does it push compression into the vision tower
   or the LM decoder, and does that flip with task (captioning vs VQA vs OCR)? Combined with §2.6
   input-dependent selection, this is "adaptive multimodal LM with input-dependent compute" almost
   verbatim.

Suggested first move: (5) is the thesis-aligned one, (1) is the cheapest credible result, (3) is the
most surprising if it holds.

---

## 5. Suggested sequencing

1. Ablate frozen-prefix vs joint nested sampling on ViT-B (§1.2) — cheapest test of the core claim,
   and it does not require changing the compression axis.
2. Import the DP profile search into the existing pruning pipeline (§2.3).
3. Prototype rank-axis Elastoformer with a capped `r_max` (§1.3) and measure Orin/Nano latency
   honestly at B=1 (§1.4). Decide the axis on that measurement, not on FLOPs.
4. LM port using `flexrank` as backend (§3), Llama-3.2-1B first.
5. Multimodal budget allocation + input-dependent profile selection (§4.5, §2.6).

## Open questions to resolve early

- Does GAR's `O((m+n−r)r)` translate into wall-clock on Jetson at B=1, or does kernel-launch overhead
  eat it?
- Can a KV cache built at rank profile A be reused at profile B without quality loss?
- Does the additive-error assumption hold at 8B scale?
- With `r_max` capped, how much accuracy does the top DN lose relative to the pretrained base — and
  is that acceptable for the edge targets Elastoformer cares about?
