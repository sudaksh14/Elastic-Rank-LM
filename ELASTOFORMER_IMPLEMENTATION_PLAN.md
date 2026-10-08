# ElastoFormer × FlexRank — Implementation & Experiment Plan

Status: **plan only, no code changed, no experiment run.**
Baseline commit: `beab276` (branch `main`, 2026-10-06).
Source design notes: [elastoformer_flexrank_brainstorm.md](elastoformer_flexrank_brainstorm.md).
Thesis context: [project/ElastoSLM_MSc_Thesis_Proposal.pdf](project/ElastoSLM_MSc_Thesis_Proposal.pdf) (Stage A = this plan's endpoint).
Results journal: [EXP-JOURNAL.md](EXP-JOURNAL.md).

Guiding rule: every speculative feature waits for the cheap experiment that validates it. All GPU work runs
on **DAS-6 over SSH only after access details are provided and approved** (§10).

---

## 0. TL;DR

| Order | ID | What | Where | Cost |
|---|---|---|---|---|
| 1 | **E0** | Env build + CPU audit of the pruning-metadata / rebuild path (resolves 4 assumptions the ablation design depends on) | local CPU | ~1 h CPU, 0 GPU-h |
| 2 | **E1** | Reproduce Elastoformer Table 1 from published checkpoints + nestedness audit of those checkpoints | DAS-6, eval only | ≤1 GPU-h |
| 3 | **E2** | Training-free fronts: L1 pruning vs SVD/DataSVD rank (uniform + DP, `r_max` caps) + additive-error check | DAS-6, eval only | 3–5 GPU-h |
| 4 | **E3** | B=1 kernel microbenchmark: dense vs pruned-dense vs 2-GEMM vs GAR | DAS-6, tiny | ≤0.5 GPU-h |
| 5 | **E4a** | **Core ablation (pilot):** frozen-prefix growth vs jointly-sampled nested training, same pruning axis, matched compute | DAS-6 | 8–12 GPU-h |
| 6 | E4b | Same at full ImageNet scale (only if E4a shows signal) | DAS-6 | 40–60 GPU-h |
| 7 | E5 | DP per-layer ratio search for the pruning axis (training-free, then one joint run) | DAS-6 | 3–6 GPU-h |
| 8 | E6 | Rank-axis ViT with joint FlexRank training and capped `r_max` | DAS-6 | 8–15 GPU-h |
| 9 | E7 | Axis fork judged on measured Jetson latency (permuted pruning vs rank/GAR) | Jetson (not DAS-6) | device time |
| 10 | E8 | LM port: reproduce FlexRank on GPT-2, then Llama-3.2-1B, via `flextrain` | DAS-6 | 3 → 120 GPU-h |
| 11 | E9 | LM diagnostics: additive error at 1B, TTFT/TPOT, KV-cache cross-profile reuse | DAS-6, eval | 3–6 GPU-h |
| 12 | E10 | Scalarization/non-convexity probe (thesis RQ1 gate) | DAS-6 | 10–20 GPU-h |

GPU-h are **A100-40GB-equivalent estimates, not measurements**. DAS-6 node GPU type is unconfirmed, so rescale
once it is known. E4a logs measured throughput, which replaces every downstream estimate.

---

## 1. Current architecture / code map

### 1.1 Repository layout

| Path | What it is | Relevance |
|---|---|---|
| `Elastoformer/` | Structured-pruning elasticity for ViT (HF) and CNN (torchvision) | Pruning axis, training loop, ImageNet loaders |
| `FlexRank/flexrank/` | Standalone core: `ODLinear/ODConv2d`, SVD/DataSVD, GAR, Gram collection, DP profile search, samplers | Rank axis backend. Depends only on torch/numpy/tqdm, but needs **Python ≥ 3.11** (`enum.StrEnum`) |
| `FlexRank/flextrain/` | Hydra + HF Trainer + Accelerate pipeline, distillation, lm-eval | LM experiments (E8–E10). Pins `transformers==4.57.5` |
| `Maestro-LoD/` | Ordered-dropout low-rank layers (Maestro, ICML'24) | Reference only (per-layer OD samplers; PufferFish sampler). Not on the execution path |
| `related papers/`, `project/` | PDFs (Elastoformer, FlexRank, MOSP, ACIP, DRONE, Maestro; thesis proposal) | Reproduction targets come from here |

No experiment logs, result CSVs, or checkpoints exist in the repo. `Elastoformer/.gitignore` excludes `saves/`,
`outputs/`, `jobs/`, `wandb/`. Published DN checkpoints are linked from `Elastoformer/README.md` (surfdrive).

### 1.2 Elastoformer ViT execution path (`Elastoformer/main.py`)

```
main()                                                     main.py:189
├─ data: load_imagenet / load_imagenette / load_cifar      datasets.py:99 / :299 / :338
├─ model: ElasticViTForImageClassification.from_pretrained models/elastoformer.py:208
├─ importance: tp.importance.GroupMagnitudeImportance(p=1) main.py:199-210
├─ pruner = tp.pruner.BasePruner(iterative_steps=N,        main.py:292-308
│     prune_head_dims=True, prune_num_heads=False, round_to=1, ignored=[classifier])
├─ for i in range(N): for grp in pruner.step(interactive=True):   main.py:358-405
│     record pruned/non-pruned in/out indices per layer, de-dup vs earlier rounds
│     grp.prune() ONLY when i == N-1   ← model is physically pruned once, at the end
│     extract_vit_weight_subset(orig_copy, …) → metadata .pth per level   main.py:408-457
├─ patch attention head sizes on the core model             main.py:467-474
├─ fine_tuner_core(core, core_epochs)                       trainer.py:370
└─ rebuild loop, Level-2 … Level-N+1                        main.py:516-589
      get_vit_info → create_vit_general(dim_dict)           prune_utils.py:573, :15
      update_vit_weights_global(…)  (Linear/LN/Conv only)    prune_utils.py:906
      freeze_partial_weights (grad-mask hooks)               prune_utils.py:1196 → partial_freezing.py
      fine_tuner(rebuild=True) → train_one_epoch_freeze      trainer.py:171, :64
         selective_gradient_clipping_norm (new weights only) prune_utils.py:1234
      extract_vit_core_weights(rebuilt) → frozen set for next level
      save per-level state_dict (full copy per level)
```

Loss everywhere is CE with timm `Mixup` (`datasets.py:32`). There is no teacher or distillation.

### 1.3 FlexRank execution path (`FlexRank/train.py`)

```
train.py → load_model_from_hf / load_dataset_from_hf
get_flexrank()                                        flextrain/utils/flexrank_utils.py:236
├─ calib evaluator (HF Trainer, lr=0, max_steps=1)
├─ init_flexrank_model → SVDTrainer (Gram collection → DataSVD per Linear) → FlexRankModel
├─ get_submodel_profiles → DPSearchAlgo: per-layer probe (Δparams, Δloss) → minimize_error_with_any_drops
│     (dominance pruning + nested filter)            flexrank/profiles/dp.py:204, :132
└─ DistillationTrainer (ce=0, kl=1) + SamplerCallback (PredefinedModelsSampler over DP profiles)
```

Core primitives that are reusable without `flextrain`:
- `ODLinear` (`flexrank/layers/linear.py`): factors `weight_u [r_max×in]`, `weight_v [out×r_max]`, active rank via
  `inner_dim` setter (`layers/base.py:74`). `r_max` is set by the factor shape passed to `build_from_uv`, so a
  **rank cap is just truncating U/V before construction**. No new layer is needed.
- `decompose_linear(layer, inputs=gram)` (`layers/decomposition.py:196`): SVD or DataSVD.
- `minimize_error_with_any_drops(savings_errors)` (`profiles/dp.py:204`): pure numpy over per-layer
  `(savings, errors)` lists, so it is **axis-agnostic** and works for pruning ratios as-is.
- `BaseSampler` and subclasses (`samplers/`): discover `ODLayer`s in any `nn.Module`; `PredefinedModelsSampler`
  supports `sandwich`.
- `SVDTrainer` + `BaseEvaluator` (`trainers/`): single-worker decomposition driver; needs an evaluator exposing
  `.model`, `.distr`, `.evaluate()`.

### 1.4 Inconsistencies between the brainstorm and the code (resolved or flagged)

| # | Brainstorm says | Code / paper shows | Status |
|---|---|---|---|
| C1 | Uniform per-round ratio `p = 1-(1-CR)^(1/N)` | Model is **not** pruned between rounds (`grp.prune()` only at `i==N-1`, main.py:404). torch-pruning's iterative scheduler therefore yields a **linear** cumulative keep fraction, Level-j keeps `1 − CR·(N+1−j)/N` of each prunable dim. Paper Table 1 FLOPs fit linear (CR=0.8: predicted 0.67/2.18/4.56/7.79/11.9 G vs reported 0.60/1.96/4.44/7.58/11.38 G). Geometric predicts 0.67/1.28/2.45/4.65/8.86 G | **Resolved: linear.** E0 confirms on real metadata |
| C2 | "All N profiles computed from pretrained weights before retraining; nested by construction" | Confirmed (main.py:358-405). The de-dup uses an `issubset` guard (main.py:374, :388). If a later round's set were not a superset, the de-dup silently skips. Deterministic L1 on unchanged weights should be nested | Confirmed; **E0 asserts nestedness explicitly** |
| C3 | DNs share one weight set and lower DNs never drift | `freeze_partial_weights` skips `classifier` (prune_utils.py:1207), so **each level retrains the full classifier**. `cls_token`/`position_embeddings` are raw `Parameter`s, never handled by `update_vit_weights_global` (iterates Linear/LN/Conv only, prune_utils.py:933-935), and so are **apparently freshly initialised in each rebuilt level and not frozen**. If true, DN_k ≠ slice of DN_{k+1} for these tensors, and rebuilt levels lose pretrained position embeddings | **Flagged. Verify in E0 (CPU) and E1 (on published checkpoints).** Confound for E4 |
| C4 | Pruned-weight metadata ~1 MB/DN | Metadata stores pruned **weight values**. At CR=0.5/N=5 that is ~10% of ViT-B weights per level (≈8–9 M params ≈ 35 MB fp32). Index-only metadata is small | Flagged; E0 measures |
| C5 | "~10N GPU-h" design time | Paper: 8×RTX 3090, 50 epochs/DN with repeated augmentation ×3. 50 ImageNet epochs of ViT-B cannot take 10 GPU-h. Not reproducible from the repo | **Unverified.** E4a measures throughput |
| C6 | Table 1 is "ViT-B" | Paper says ViT-B, base 81.74%. README/run.sh use `facebook/deit-base-patch16-224` | **Ask user** which checkpoint produced Table 1 |
| C7 | Elastoformer and FlexRank can share one process | Elastoformer pins Python 3.10 + `transformers==4.28.0` (ElasticViT subclasses 4.28 internals). flexrank needs Python ≥ 3.11. flextrain pins `transformers==4.57.5` | Plan: two envs (§3, I-0). `flexrank` core installs into the Elastoformer env with `--no-deps` |
| C12 | Rebuilt levels reuse pretrained weights for the shell | `update_vit_weights_global` copies only shell×shell and core×core blocks; off-diagonal blocks keep random init (E0/I-5 test). F-faithful therefore starts from a partly random matrix, J from the full pretrained one | **Resolved:** E4a runs F-faithful and F-corrected (`slice_state_dict` init, A11) so the schedule is the only difference |

Other defects found that affect experiment validity (to fix behind flags before E4, defaults unchanged):
- `trainer.py` always applies the module-level timm `Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.11,
  num_classes=1000)` from `datasets.py:32`. **CLI `--mixup-alpha/--cutmix-alpha` are ignored.** For CIFAR,
  `load_cifar` also mixes in its collate (`datasets.py:376-381`), so mixup runs twice with the wrong class count.
  **Do not use the CIFAR path until fixed.**
- `fine_tuner*` logs `test_ema_acc1` even when `--model-ema` is off, which raises `NameError` with `--log_wandb` (trainer.py:338).
- The non-AMP clipping branch calls the CNN clipper for ViTs (trainer.py:102).
- `wandb.init` runs once per training stage, so each level becomes a separate run.
- FlexRank `cv.yaml` references `profile_algo: uniform`, which has no config file (CLI override masks it).
  `sampler/predefined_models_largest.yaml` passes `sample_only_largest`, which no sampler accepts. Avoid both.
- No edge-latency harness exists in the repo, even though the paper reports Orin/Nano numbers. It is needed for E7.

Housekeeping (not touched): `.gitignore` entries `./project` / `./related papers` have a `./` prefix and CRLF
endings, so they match nothing. Those PDFs, including third-party papers and the thesis proposal, are committed.

---

## 2. Target architecture mapped to files

Two nested-elasticity axes over one shared training scheme, with Elastoformer's runtime and measurement layer on
top. New code lives next to the code it reuses. Nothing in the default `main.py` path changes, so the paper
baseline stays reproducible.

| Concern | Today | Target | File (new = ★) |
|---|---|---|---|
| Pruning metadata | Inline loop in `main()` | `compute_nested_keep_sets(model, imp, args) -> {level: {layer: (in_idx, out_idx)}}`, cumulative keep sets. Plus a `--prune_only` flag that saves and exits | ★`Elastoformer/utils/nested_metadata.py`; `main.py` calls it |
| Level materialisation | Physical rebuild per level (`create_vit_general` + copy) | **Virtual slicing**: one full-width super-model, `set_level(k)` gathers rows/cols per Linear, gathers LN params and normalises over active dims only, gathers patch-embed/cls/pos channels, uses `head_size_k = |I_k^qkv| / heads` and scale `1/sqrt(head_size_k)` | ★`Elastoformer/models/sliced_vit.py` (wraps `ElasticViT*` modules) |
| Growth schedule | Frozen-prefix, sequential (`freeze_partial_weights`) | Joint nested sampling: per-step level sampling (uniform / sandwich / weighted) | ★`Elastoformer/train_nested.py` reusing `trainer.py` optimiser/scheduler/EMA setup |
| Supervision | CE + mixup | Optional KD from the frozen pretrained teacher (`ce_w`, `kl_w`, `T`) | same ★`train_nested.py` |
| Rank axis in ViT | — | `ElasticViT` encoder Linears → `ODLinear` via `flexrank.decompose_linear` (DataSVD), cap `r_max`, flexrank samplers | ★`Elastoformer/utils/flexrank_bridge.py` (adapter implementing flexrank `AbstractEvaluator` over Elastoformer loaders, logits wrapper, decomposition, `r_max` truncation) |
| Profile search | Uniform linear schedule | `minimize_error_with_any_drops` over per-layer probes (pruning ratios for internal dims, ranks for ODLinear) | ★`Elastoformer/utils/profile_search.py` (thin, calls flexrank DP) |
| Training-free fronts | — | Evaluate levels/profiles without training → CSV | ★`Elastoformer/eval_fronts.py` |
| Contiguous deploy (axis fork) | Scattered indices | Offline permutation by introduction level, head- and DepGraph-consistent, so level k becomes a `W[:d_k,:d_k]` view | ★`Elastoformer/utils/permute.py` (only after E4/E6) |
| Measurement | Paper-only | B=1 kernel bench; edge latency (50 trials, 10 warm-up, as in the paper); switch latency | ★`Elastoformer/tools/bench_kernels.py`, ★`tools/bench_latency.py` |
| Audit | `compare_tensor.py` (hard-coded paths) | Metadata/rebuild/nestedness audit | ★`Elastoformer/tools/audit_pruning.py` |
| Tests | none in Elastoformer | Slicing equivalence, nestedness, freeze integrity | ★`Elastoformer/tests/` |
| LM | — | Use `flextrain` directly (no port of the pruning axis to LMs: RoPE/GQA/residual-width blockers, brainstorm §3) | configs only |
| Jobs | ad hoc | Slurm templates writing to the DAS-6 run directory (§10) | ★`slurm/` (repo root) |

Code that the brainstorm says "would disappear" (index surgery, freezing hooks) is **kept** until E4 decides.
It is the frozen-prefix baseline arm.

---

## 3. Ordered implementation steps

Each step names the experiment it unblocks. Nothing after I-4 starts before E4a has a result.

| Step | Work | Unblocks | Done when |
|---|---|---|---|
| **I-0** | Envs. (a) `elasto`: Python 3.11, `torch==2.7.0`, `torchvision==0.22.0`, `transformers==4.28.0`, `tokenizers==0.13.3` (**verify a cp311 wheel exists, otherwise needs Rust**), `torch-pruning==1.6.1`, `timm==1.0.19`, rest of `Elastoformer/requirements.txt`, then `pip install -e FlexRank/flexrank --no-deps`, `pytest`. (b) `flextrain`: Python 3.11 per `FlexRank/requirements.txt`. Record `pip freeze` per env | E0 | `python -c "import flexrank, torch_pruning, transformers"` works in `elasto`; `pytest FlexRank/flexrank/tests` passes |
| **I-1** | `tools/audit_pruning.py`: runs the main.py pruning loop on pretrained ViT/DeiT-B on CPU with `example_inputs` only (L1 needs no data), no training. Dumps per level: cumulative keep fraction per layer, nestedness assertion, per-head uniformity of head-dim pruning, list of params not covered by metadata, metadata size (indices vs weights), and builds Level-2 via `create_vit_general` + `update_vit_weights_global` to check `cls_token`/`position_embeddings` against the pretrained slice | E0 | JSON report written |
| **I-2** | `utils/nested_metadata.py` extracted from main.py:358-405 (pure refactor, unit test: identical index dicts to the inline loop) + `--prune_only` | E2, E4 | Test passes on a 2-layer tiny ViT on CPU |
| **I-3** | `eval_fronts.py` + `utils/flexrank_bridge.py` + `utils/profile_search.py` (training-free) | E2, E5 | Full-rank DataSVD reproduces base logits (fp32 max-abs-diff < 1e-3) on CPU with 8 random images |
| **I-4** | Small fixes behind flags: honour `--mixup-alpha/--cutmix-alpha`, guard `test_ema_acc1`, single wandb run, save `results.json` per level, `--train_subset <frac>` (class-balanced, seeded, index list saved), `--freeze_pos_cls` (corrected F arm, A11). Defaults stay paper-faithful | E4 | Old CLI produces the same behaviour as before |
| **I-5** | `models/sliced_vit.py` + `tests/test_sliced_equivalence.py` | E4 | For every level k: sliced logits == physically pruned Level-k logits (fp32, ≤1e-5) on a tiny ViT and on ViT-B (CPU, 2 images) |
| **I-6** | `train_nested.py`: level sampler (uniform / sandwich / weighted), CE or KD, shared sliced classifier (default) or per-level heads (ablation), DDP, per-level eval, sampler-coverage logging, throughput logging | E4 | 20-step CPU smoke on dummy data; GPU smoke inside the E4a job |
| — | **Gate: E4a result** | | |
| I-7 | DP for pruning ratios: per-layer probes on internal dims (FFN intermediate, QKV head-dims). Residual-stream width stays uniform because it is one DepGraph group across all blocks. Level keep sets come from fixed per-layer L1 rankings, which keeps them nested | E5 | Profiles nested; params match target ±1% |
| I-8 | Rank-axis ViT training path: `flexrank_bridge` + flexrank sampler inside `train_nested.py` (`--axis rank`) | E6 | Sampler toggles `inner_dim` on all ODLinears; eval per profile |
| — | **Gate: E4 + E6 results decide the axis** | | |
| I-9 | `utils/permute.py` (contiguous levels) + `tools/bench_latency.py` (Jetson) | E7 | Permuted model output == unpermuted at every level |
| I-10 | LM: flextrain configs/scripts for DAS-6 (no new model code) | E8–E10 | GPT-2 pilot runs end-to-end |

---

## 4. Experiment 0 — local CPU audit (first experiment)

**ID** E0 · **Where** local CPU only · **GPU** none · **DAS-6** not needed.

**Purpose.** Before any GPU spend, settle four facts that change the design of the core ablation (E4) and the
interpretation of the paper baseline: C1 schedule, C2 nestedness, C3 what is actually shared, and per-head
uniformity (a precondition for virtual slicing, I-5).

**Hypotheses.**
- H0.1: Cumulative keep fraction per level is linear, `1 − CR·(N+1−j)/N`, for every pruned dim.
- H0.2: Level keep sets are strictly nested for all layers (`I_1 ⊂ I_2 ⊂ … ⊂ I_{N+1}`).
- H0.3: `prune_head_dims=True` removes the same number of dims from every head, so a per-level head size exists.
- H0.4 (expected to be **rejected**): rebuilt Level-2 carries pretrained `cls_token`/`position_embeddings`.
  The code suggests they are re-initialised.

**Variables.** Model ∈ {`google/vit-base-patch16-224`, `facebook/deit-base-patch16-224`}; CR ∈ {0.4, 0.6, 0.8}; N = 5;
importance = L1. No data, no training.

**Expected outputs** (`Elastoformer/outputs/E0/`, gitignored): `audit_<model>_cr<CR>.json` with per-level keep
fractions, nestedness booleans, head-uniformity booleans, uncovered parameter names, metadata MB (indices vs
weights), and pos-emb/cls comparison (cosine + max-abs-diff vs pretrained slice); `pytest` log of flexrank tests.

**Success criteria.** Script runs for all 6 configs; `flexrank/tests` pass (FlexRank's `refactor.md` reports
39 passed / 1 skipped for the full two-package suite);
each of H0.1–H0.4 gets a clear true/false with evidence.

**Decision.**
- H0.2 or H0.3 false → I-5 must gather per-head index sets (no common head size); redesign before E4.
- H0.4 false (as expected) → user decision (journal): run the frozen baseline **faithful** (as published) and/or
  **corrected** (copy + freeze pretrained pos/cls slices). Recommendation: report the faithful one as the paper
  reproduction (E1) and use corrected for E4, so the ablation isolates the schedule.
- H0.1 false → recompute level parameter targets used by E2/E4 from the measured fractions.

**Commands** (once I-0 and I-1 exist; Git Bash on Windows):
```bash
cd "Elastic Rank LM"
py -3.11 -m venv .venv-elasto && source .venv-elasto/Scripts/activate
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.28.0 tokenizers==0.13.3 torch-pruning==1.6.1 timm==1.0.19 \
            fastdownload==0.0.8 matplotlib seaborn scikit-learn tqdm wandb pytest
pip install -e FlexRank/flexrank --no-deps
python -m pytest FlexRank/flexrank/tests -q                       # E0.1
cd Elastoformer
python tools/audit_pruning.py --model_name google/vit-base-patch16-224 \
       --pruning_ratio 0.8 --pruning_steps 5 --out outputs/E0/      # E0.2/E0.3, repeat per config
```
Requires a local Python 3.11 (currently only 3.14 is installed, without torch) and ~2 GB of downloads
(CPU torch + two ~350 MB HF checkpoints).

**Cost.** ~1 h wall clock on CPU (pruning ViT-B with torch-pruning on CPU is seconds to minutes per config).

---

## 5. Subsequent experiments (dependency/priority order)

Common to all DAS-6 runs: each writes `config.yaml`, `cmd.sh`, `git_sha.txt`, `pip_freeze.txt`, Slurm stdout/err,
`results.json` and checkpoints into its run directory (§10). ImageNet-1k = ILSVRC-2012 train/val (path on DAS-6 to
be provided). Calibration data is always drawn from **train**, never val.

### E1 — Reproduce Table 1 from published checkpoints + nestedness audit
- **Objective.** Establish the frozen-prefix reference without training. Test C3 on real checkpoints.
- **Depends on.** E0 (for the audit tooling). Published checkpoints (surfdrive link in `Elastoformer/README.md`).
- **Code/config.** No model changes. Small eval wrapper: load each `Vit_b_16_Rebuilt_Level_k_state_dict_*.pth`,
  build via `get_vit_info(core_model=True)` + `create_vit_general`, call `trainer.evaluate`. Generalise
  `compare_tensor.compare_weights_multilevel` (paths are hard-coded) and extend it to `cls_token`,
  `position_embeddings`, `classifier`.
- **Command.** `python tools/eval_checkpoints.py --ckpt_dir <ckpts> --data_path <imagenet> --out <run>/` (single GPU).
- **Data/ckpts.** ImageNet val (50k). Published Level-1…6 checkpoints for CR ∈ {0.4, 0.6, 0.8}.
- **Metrics.** Top-1/Top-5 per level; params; MACs (`tp.utils.count_ops_and_params`); per-tensor nestedness
  (max-abs-diff between DN_k and the slice of DN_{k+1}); per-level classifier/pos-emb deltas.
- **Expected.** Top-1 within ±0.3 pp of Table 1 (e.g. CR=0.8: 51.19 / 64.14 / 69.97 / 72.46 / 73.61 / 74.65). Encoder
  Linear/LN weights exactly nested. Classifier (and likely pos/cls) not nested.
- **Success.** Reproduction within ±0.3 pp at ≥5/6 levels per CR.
- **Failure → decision.** Mismatch → check checkpoint (C6) and eval transforms (`datasets.build_transform`,
  crop 0.875, bicubic). If unresolved, the E4a frozen arm becomes the reference at matched budget, and the paper
  numbers are cited, not compared against.
- **Cost.** 18 evals × ~1–2 min ≈ **≤1 GPU-h**, one GPU.

### E2 — Training-free Pareto fronts: pruning vs rank, uniform vs DP, `r_max` caps, additive error
- **Objective.** Get the untrained starting fronts for both axes on the same backbone, validate the flexrank
  bridge, quantify the top-mode loss from capping `r_max` (brainstorm §1.3 open question), and test the DP's
  additive-error assumption at ViT-B scale (§2.4).
- **Depends on.** I-2, I-3. E1 fixes the backbone.
- **Variables.**
  - Pruning arm: one-shot L1, CR ∈ {0.4, 0.6, 0.8}, N=5 → 6 levels each, no fine-tuning.
  - Rank arm: all 72 encoder Linears (q,k,v,o,fc1,fc2 × 12) → ODLinear; decomposition ∈ {SVD, DataSVD (4,096 train
    images, val transform)}; profiles ∈ {uniform p, DP}; `r_max/min(m,n)` ∈ {1.0, 0.5, 0.3}; budgets matched to the
    pruning arm's level parameter counts. Classifier and patch embedding are excluded.
  - Additive error: 30 random per-layer profiles; predicted Σ per-layer Δloss vs measured Δloss.
- **Command.** `python eval_fronts.py --axis {prune,rank} --model_name <E1 model> --data_path <imagenet> --calib_n 4096 --out <run>/`.
- **Metrics.** Top-1 val; calibration loss; params with/without embeddings+classifier; stored params
  `(m+n)·r_max`; MACs; Spearman ρ and Kendall τ (predicted vs measured Δloss, calib and val).
- **Expected.** Full-rank SVD/DataSVD = base accuracy (sanity S1). DataSVD ≥ SVD at every truncated budget. Both
  axes degrade sharply at aggressive budgets without training. DP ≥ uniform at matched params. ρ unknown: FlexRank
  validated it only on a 4-layer MNIST net.
- **Success.** S1 passes; fronts produced for all cells; ρ reported with CI.
- **Decision.** ρ ≥ 0.8 → trust DP profiles downstream (E5, E6). ρ < 0.6 → DP profiles get a measured-loss
  re-ranking pass (evaluate top-k DP candidates directly) before use. `r_max` cap: pick the cap(s) for E6 from the
  untrained top-mode loss and stored-parameter trade-off. The front shapes go into the paper as the
  "no-consolidation" baseline (thesis WP1 gate).
- **Cost.** ~40 val evals (~1.5 h) + DP probing (72 layers × ~10 ranks × ~5 s on a 4k calib set ≈ 1 h) ×2 decompositions
  ≈ **3–5 GPU-h**, one GPU.

### E3 — B=1 kernel microbenchmark (datacenter proxy)
- **Objective.** First look at the brainstorm's §1.4 risk: does a rank-r layer (two GEMMs / GAR) beat a
  structurally pruned dense layer at equal FLOPs, at batch 1?
- **Depends on.** I-0 only (synthetic tensors). Can run in parallel with E2.
- **Variables.** Shapes: ViT-B prefill (197×768 @ 768×768, 768×3072, 3072×768); Llama-3.2-1B decode (1×2048 @
  2048×2048, 2048×8192, 8192×2048, 2048×512 GQA kv). Variants: dense full, dense pruned to matched params, sliced
  two-GEMM at r/min(m,n) ∈ {0.1…1.0}, GAR (`prune_weights(use_gar=True)`), materialised `U_r V_r^T`. fp16 and bf16.
  CUDA graphs on/off.
- **Command.** `python tools/bench_kernels.py --out <run>/bench.csv` (one GPU, ~15 min).
- **Metrics.** Median/p90 latency over 200 iters after 50 warm-up; crossover rank where low-rank beats dense-pruned.
- **Expected (hypothesis, unmeasured).** Prefill: low-rank wins only below r/min ≈ 0.3–0.4. Decode
  (bandwidth-bound): latency tracks bytes read, so the gap to dense-pruned is small.
- **Success.** Crossover curves for all shapes.
- **Decision.** Sets realistic `r_max` and profile constraints for E6. If low-rank never wins at B=1, raise the
  priority of the permuted-pruning path (E7) and the hybrid axis.
- **Cost.** **≤0.5 GPU-h.** Not a substitute for Jetson (E7).

### E4a — Core ablation, pilot: frozen-prefix vs joint nested sampling (pruning axis)
- **Objective.** Test the brainstorm's central claim (§1.2): dropping the hard freeze in favour of joint budget
  sampling improves the ladder (fixes the CR=0.8 collapse) at matched compute, with axis, backbone, index sets and
  loss all held fixed.
- **Depends on.** E0 (C3 decision), I-4, I-5, I-6. E1 (backbone/reference).
- **Arms** (same index sets from I-2, same init = pretrained, same data/augmentation, same optimiser family):
  - **F** frozen-prefix: existing `main.py` pipeline with `--core_epochs E --epochs E`, faithful or corrected
    pos/cls per the E0 decision.
  - **J** joint: `train_nested.py --sampler uniform`. Total samples processed = F's total
    (`(N+1)·E` epochs worth). Shared sliced classifier.
  - **J+KD**: J with `ce_w=0, kl_w=1, T=1` from the frozen pretrained model (FlexRank default). CR=0.8 only.
- **Variables.** CR ∈ {0.5, 0.8}; N = 3 (4 levels) for the pilot; E = 10 epochs per stage on a **10% class-balanced
  ImageNet-1k train subset** (~128k images, fixed seed, list saved) → 40 subset-epochs per arm; full 50k val.
  1 seed.
- **Command (sketch, finalised after DAS-6 details).**
  ```bash
  # F
  torchrun --nproc_per_node=$G main.py --exp_name E4a_F_cr0.8 --model_name <E1 model> --dataset_name imagenet \
     --data_path <imagenet> --pruning_type l1 --pruning_ratio 0.8 --iterative --pruning_steps 3 \
     --core_epochs 10 --epochs 10 --train_subset 0.1 --rebuild --test_accuracy --amp --distributed \
     --save_as <run>/ckpt/
  # J
  torchrun --nproc_per_node=$G train_nested.py --exp_name E4a_J_cr0.8 --metadata <run_E2>/meta_cr0.8_N3.pth \
     --sampler uniform --total_epochs 40 --train_subset 0.1 --loss ce --amp --out <run>/
  ```
  (`--train_subset` is new in I-4, `--metadata` in I-6. The N=3 metadata comes from `main.py --prune_only`
  (I-2), which is a CPU-cheap step.)
- **Metrics.** Top-1 per level (val 50k); mean Top-1 over levels; area under the acc–MACs curve; Top-1 gap to base
  at the top level; measured GPU-h and img/s per arm; sampler coverage per level; for F, frozen-weight hash
  unchanged across stages (S10).
- **Expected.** J ≥ F at every level, with the largest gain at the top level (F cannot revise the core) and at
  CR=0.8. KD ≥ CE.
- **Success.** At CR=0.8: J beats F by **≥ +2.0 pp mean Top-1 over levels** and is not worse than F by more than
  0.5 pp at any level. At CR=0.5: J ≥ F − 0.3 pp everywhere. (Seed noise for ImageNet fine-tuning is typically
  ~0.1–0.3 pp. Margins are set above that.)
- **Failure → decision.**
  - J ≈ F: run ablations A1 (sandwich) and A9 (frequency-weighted sampling) before concluding.
  - J < F at the top levels only: update-frequency imbalance (§1.6c) → weighted sampling.
  - J < F overall: frozen schedule stays the default for the pruning axis. Theory (FlexRank Thm 4.3) only
    covers the rank axis, so move straight to E6 and treat this as a reported negative.
  - Success → E4b, and joint sampling becomes the default for E5/E6.
- **Cost.** 5 runs × ~1.5–2.5 GPU-h (5.1M samples each at ~1k img/s for DeiT-B, smaller levels faster, KD +~30%)
  ≈ **8–12 GPU-h**.

### E4b — Core ablation at full scale
- **Objective.** Confirm E4a on full ImageNet-1k with the paper's N=5.
- **Depends on.** E4a success.
- **Variables.** CR ∈ {0.5, 0.8} (add 0.4 if budget allows, to match Table 1); N=5; E=5 epochs per stage (30 epochs
  total per arm); arms F, best-of-{J, J+KD}; 2 seeds for CR=0.8.
- **Command.** As E4a without `--train_subset`, `--pruning_steps 5`, `--core_epochs 5 --epochs 5`.
- **Metrics/expected/success.** As E4a. Also compare against the published 50-epochs/stage ladder (E1) as an
  unmatched upper reference.
- **Decision.** Success → headline result ("joint nested training fixes the frozen-core collapse at 1/N design
  time"). Proceed to E5/E6 with joint training.
- **Cost.** ~10–14 GPU-h per run × 5–6 runs ≈ **40–60 GPU-h**, multi-GPU DDP.

### E5 — DP profile search for the pruning axis
- **Objective.** Replace the uniform linear schedule with DP-allocated per-layer ratios (brainstorm §2.3). This is
  the lowest-risk import from FlexRank.
- **Depends on.** E2 (ρ), I-7. The training step depends on E4a (uses the winning schedule).
- **Variables.** Profiles ∈ {uniform (paper), DP}; importance ∈ {L1, activation-aware |W|·‖x‖ (Wanda-style, same calib
  data)} (A4); CR=0.8, N=5 targets matched in params.
- **Command.** `python eval_fronts.py --axis prune --profiles dp --importance {l1,act} …`, then one
  `train_nested.py --profiles <dp.json>` run with the E4a recipe.
- **Metrics.** Training-free Top-1 per level at matched params; after training: as E4a.
- **Expected.** DP > uniform at aggressive levels. Allocation is non-uniform across depth (FlexRank Fig. 6
  pattern). Activation-aware ≥ L1.
- **Success.** Training-free: DP ≥ uniform + 1.0 pp at ≥3 of the lowest 4 levels. Trained: gain persists ≥ 0.5 pp.
- **Decision.** Adopt DP profiles (and act-aware importance if it wins) as the default pruning-axis config.
- **Cost.** 2–3 GPU-h training-free + ~2 GPU-h pilot training ≈ **3–6 GPU-h**.

### E6 — Rank-axis ViT with joint FlexRank training and capped `r_max`
- **Objective.** The other side of the axis fork (brainstorm §1.6 table): same backbone, same budget levels, nested
  low-rank with DataSVD + DP profiles + joint distillation, with storage capped below dense.
- **Depends on.** E2 (`r_max`, ρ), E3 (crossover), E4a (training recipe), I-8.
- **Variables.** `r_max/min(m,n)` ∈ {1.0, 0.5} (+0.3 if E2/E3 favour it); sampler ∈ {PredefinedModels, sandwich};
  KD as in FlexRank (ce=0, kl=1). Same E4a budget and data subset. Levels matched to E4a/E5 level params.
- **Command.** `torchrun … train_nested.py --axis rank --decomp datasvd --r_max 0.5 --profiles dp --loss kd …`.
- **Metrics.** Top-1 per level; stored params/MB vs dense; params and MACs per level; predicted latency per level
  from E3 tables.
- **Expected.** Rank ≥ best pruning-axis J at aggressive levels (FlexRank reports ViT within ~5% of full down to
  ~30% params). With `r_max<1` the top mode loses accuracy vs base. E2 quantified the untrained loss; training
  should recover part of it.
- **Success.** Rank-J ≥ prune-J at ≥4/6 levels on Top-1 at matched params, with storage ≤ 1.0× dense at
  `r_max=0.5`.
- **Decision.** Feeds E7. Rank wins on accuracy but loses on E3 latency → hybrid axis (rank where the DP picks
  r/min ≲ 0.3, permuted pruning elsewhere) becomes the E7 candidate.
- **Cost.** 3–4 runs × 2.5–4 GPU-h ≈ **8–15 GPU-h**.

### E7 — Axis fork on edge hardware
- **Objective.** Decide the deployment axis on **measured** Jetson Orin/Nano latency at B=1, not FLOPs
  (brainstorm §5.3).
- **Depends on.** E4/E5/E6 checkpoints, I-9. **Needs Jetson access, which is not DAS-6** (to be requested).
- **Variables.** Candidates: permuted-pruning (dense GEMMs, contiguous views); rank with GAR; hybrid. Per level:
  latency (50 trials, 10 warm-up, as in the paper), peak memory, mode-switch latency (paper baseline 50.4 ms),
  total storage.
- **Metrics/expected.** Permuted pruning: switch ≈ 0, dense kernels, ~1× memory. Rank: switch ≈ 0, possibly slower
  per level at B=1. A negative result for rank is publishable (brainstorm §1.4).
- **Success.** A clear Pareto plot (Top-1 vs measured latency) per device, with switch latency.
- **Decision.** Selects the axis for the ViT paper. For LMs the decision is made separately in E9 (decode is
  bandwidth-bound).
- **Cost.** No GPU-h; ~1–2 days of device time.

### E8 — LM port via `flextrain`: GPT-2, then Llama-3.2-1B (thesis WP0)
- **Objective.** Reproduce FlexRank's LM front with the released pipeline before any LM changes.
- **Depends on.** I-0(b), I-10. HF token for gated Llama. Independent of E4–E7, so it can start once DAS-6 is
  available.
- **Variables.** Pilot: `model=gpt2 dataset=finewebedu_10bt_gpt2`, `steps=1000` (script default 6000, global batch 512
  × 1024 tokens). Full: GPT-2 6000 steps; then `model=llama1b dataset=finewebedu_10bt_llama1b`. DataSVD, DP, ce=0/kl=1
  (released defaults). `lm_head` excluded (default).
- **Command.** `bash scripts/flexrank_nlp.sh model=gpt2 dataset=finewebedu_10bt_gpt2 steps=1000 project=<wandb> …`
  with `BASE_DIR` overridden to the DAS-6 run directory (the script defaults to `/tmp`, so it must be changed).
- **Metrics.** Held-out PPL per profile; lm-eval commonsense (`config/lm_eval/commonsense.yaml`); params with and
  without embeddings/`lm_head`.
- **Expected.** Front within tolerance of FlexRank's published curves. **Read the target numbers from
  `related papers/FlexRank.pdf` and record them in the journal before running.**
- **Success.** GPT-2 full run within ±3% PPL of the paper at each reported budget.
- **Decision.** Reproduced → Llama-1B, then E9/E10. Not reproduced → debug before any extension.
- **Cost.** GPT-2 pilot **~2–3 GPU-h**; GPT-2 full ~9–12 GPU-h; Llama-1B pilot (500 steps) ~8–10 GPU-h; Llama-1B full
  ~90–120 GPU-h (≈3.1B tokens × (6+2)·1.24B FLOPs).

### E9 — LM diagnostics (eval only, on E8 checkpoints)
- **Objective.** (a) Additive-error assumption at 1B (thesis RQ5). (b) TTFT vs TPOT per profile on a datacenter GPU
  (E3 harness extended to full model generate). (c) KV-cache reuse across profiles (brainstorm §1.5.1): build cache at
  profile A, continue decoding at profile B, compare PPL/accuracy to recompute-at-B.
- **Depends on.** E8 Llama-1B (GPT-2 is acceptable for a first pass).
- **Metrics.** ρ/τ; TTFT, TPOT, peak memory at B=1; ΔPPL for cache reuse at A→B for adjacent and distant profiles.
- **Expected.** ρ lower at 1B than at ViT-B (more interacting layers). TPOT tracks active params. Cache reuse works for
  adjacent profiles but degrades for distant ones (V-subspace shift).
- **Success/decision.** ρ < 0.6 → DP needs a correction term (thesis contribution). Cache reuse ΔPPL < 1% for
  adjacent profiles → "elastic serving" claim is viable and justifies the alignment-loss ablation (A10).
- **Cost.** **3–6 GPU-h.**

### E10 — Scalarization / non-convexity probe (thesis RQ1 gate)
- **Objective.** Before building MOO machinery (thesis Stage B), test cheaply whether weighted-sum training leaves
  budgets unreachable: sweep the per-budget weights (sampling probabilities) and look for non-convex regions.
- **Depends on.** E8 GPT-2 pipeline.
- **Variables.** 4–5 α settings (uniform, small-heavy, large-heavy, two mixed) on GPT-2, 1000-step pilots.
- **Metrics.** Per-budget PPL; hypervolume of the union front; convex-hull gap.
- **Expected.** Unknown. This is the thesis pivot.
- **Decision.** Non-convex kink found → Stage B (MGDA/CAGrad/hypernetwork) justified. Front convex → thesis
  re-centres on RQ3/RQ4 as planned.
- **Cost.** 5 × 2–3 GPU-h ≈ **10–20 GPU-h**.

---

## 6. Per-experiment summary table

| ID | Objective | Code changes | Data / ckpts | Key metric | Success | Next if success / fail | GPU-h |
|---|---|---|---|---|---|---|---|
| E0 | Audit schedule, nestedness, sharing, head uniformity | I-0, I-1 | HF ViT-B / DeiT-B weights | booleans + MB | all answered | design I-5 / redesign | 0 |
| E1 | Reproduce Table 1, audit published ckpts | eval wrapper | ImageNet val, surfdrive ckpts | Top-1/level | ±0.3 pp | E2 / use E4a F as reference | ≤1 |
| E2 | Untrained fronts, r_max, additive error | I-2, I-3 | ImageNet train (calib) + val | Top-1 vs params, ρ | S1 + ρ reported | E5/E6 inputs | 3–5 |
| E3 | B=1 kernel crossover | bench script | synthetic | latency | curves | constrain E6 / prioritise E7 | ≤0.5 |
| E4a | Freeze vs joint (pilot) | I-4, I-5, I-6 | 10% ImageNet train, full val | mean Top-1 over levels | +2 pp @CR0.8 | E4b / ablate → E6 | 8–12 |
| E4b | Freeze vs joint (full) | — | ImageNet-1k | same | same | headline / report negative | 40–60 |
| E5 | DP ratios for pruning | I-7 | as E2/E4a | Top-1 @ matched params | +1 pp untrained | default config | 3–6 |
| E6 | Rank axis, joint, capped r_max | I-8 | as E4a | Top-1, storage | ≥4/6 levels | E7 candidates | 8–15 |
| E7 | Axis on Jetson | I-9 | E4–E6 ckpts | measured latency | Pareto per device | choose axis | 0 (device) |
| E8 | FlexRank LM reproduction | configs | FineWeb-Edu, GPT-2, Llama-1B | PPL, lm-eval | ±3% PPL | E9/E10 / debug | 3 → 120 |
| E9 | LM additive error, TTFT/TPOT, KV reuse | eval scripts | E8 ckpts | ρ, TPOT, ΔPPL | reported | A10 / DP correction | 3–6 |
| E10 | Non-convexity probe | sampler weights | GPT-2 | hypervolume gap | reported | Stage B / pivot RQ3-4 | 10–20 |

---

## 7. Ablations and sanity checks

### Sanity checks (cheap, gate every run that depends on them)
| ID | Check | Where | Pass criterion |
|---|---|---|---|
| S1 | Full-rank SVD/DataSVD reproduces base | E2, CPU test in I-3 | Top-1 Δ ≤ 0.05 pp; logits max-abs-diff < 1e-3 fp32 |
| S2 | Sliced super-model at level k == physically pruned Level-k | I-5 test | logits ≤ 1e-5 fp32, all levels |
| S3 | Nestedness integrity after training: Level-k sliced from the final top model == Level-k evaluated | E1, E4, E6 | identical Top-1 (J by construction; F expected to fail on classifier/pos-emb) |
| S4 | Param/MAC counts: `tp.utils.count_ops_and_params` vs analytic | E2 | ≤ 0.5% diff |
| S5 | Eval determinism | E1 | same ckpt twice → identical Top-1 |
| S6 | Distributed eval covers exactly 50,000 val images (warning in `trainer.evaluate`) | all | no warning; else evaluate single-process |
| S7 | Calibration ⊂ train, disjoint from val | E2, E5, E6 | index lists saved and checked |
| S8 | Teacher frozen (`requires_grad=False`, `eval()`) | KD runs | param hash unchanged |
| S9 | Sampler coverage per level logged | E4, E6 | within ±5% of the intended distribution |
| S10 | Frozen-prefix weights unchanged across F stages | E4 F arm | hash equality of frozen slices |
| S11 | Pilot subset is class-balanced and fixed | E4a | list file hashed in journal |

### Ablations (run only when the parent experiment needs them to interpret its result)
| ID | Factor | Levels | Parent |
|---|---|---|---|
| A1 | Sampler | uniform vs sandwich (largest+random+smallest) | E4a (if J≈F), E6 |
| A2 | Loss | CE vs KD (ce=0,kl=1) vs mix | E4a |
| A3 | Classifier | shared sliced head vs per-level heads (Elastoformer-like, removes C3 confound) | E4a |
| A4 | Importance | L1 vs activation-aware | E5 |
| A5 | Profiles | uniform vs DP | E2, E5, E6 |
| A6 | `r_max/min(m,n)` | 1.0 / 0.5 / 0.3 | E2, E6 |
| A7 | Decomposition | SVD vs DataSVD | E2 |
| A8 | Number of levels | N = 3 vs 5 | E4a → E4b |
| A9 | Update-frequency compensation | uniform vs size-weighted sampling vs per-param LR scaling | E4a (if top levels lag) |
| A10 | Cross-budget hidden-state alignment loss | off vs feature-KD vs CKA (brainstorm §1.5.2) | after E9 cache-reuse result |
| A11 | Frozen baseline variant | faithful vs corrected pos/cls (C3) | E4a |

Out of scope for this plan: CNN branch (`elastic_cnn.py`), whose joint training reintroduces the switchable-BN
problem (brainstorm §2.7); quantization composition (§2.5); input-dependent routing (§2.6). Each needs its own gate.

---

## 8. Final evaluation experiments

Run only on configurations that won their gates. Report mean ± std over seeds where stated.

| ID | What | Setup | Metrics |
|---|---|---|---|
| F1 | ViT final fronts | ViT-B (E1 model), ImageNet-1k, CR ∈ {0.4, 0.6, 0.8}, N=5; arms: published frozen ladder (E1), F at matched budget, best joint pruning-axis, rank-axis (r_max chosen in E6); 2 seeds for the main arms | Top-1/Top-5 per level, params (with and without embeddings), MACs, total storage vs bag-of-models (paper: 336 MB vs 843 MB), design GPU-h |
| F2 | Edge deployment | Jetson Orin + Nano, B=1, fp16 | per-level latency, switch latency, peak memory, energy if available |
| F3 | LM final fronts | Llama-3.2-1B (+ Qwen3-1.7B if budget allows), FineWeb-Edu | PPL, lm-eval commonsense, params with and without `lm_head`, TTFT/TPOT separately |
| F4 | Serving properties | best LM checkpoint | mid-generation profile switch: latency + quality (KV reuse), memory at every mode ≤ dense |

---

## 9. Risks, unknowns, fallbacks

| Risk / unknown | Signal | Fallback |
|---|---|---|
| `tokenizers==0.13.3` has no cp311 wheel → `elasto` env fails | I-0 install error | Install Rust toolchain to build it, or use Python 3.10 for Elastoformer and vendor flexrank with a `StrEnum` backport shim (one line in `layers/base.py`, kept local) |
| C3 confirmed: rebuilt DNs re-init pos/cls and retrain classifier | E0/E1 | Report the published baseline faithfully; E4 uses the corrected F arm (A11) so the ablation isolates the schedule |
| Per-head pruning not uniform (H0.3 false) | E0 | Per-head index sets in `sliced_vit.py` (gather per head, pad-free) at extra compute |
| Joint sampling does not beat freezing on the pruning axis | E4a | A1/A9/A2 first; then report the negative and rely on the rank axis, where Thm 4.3 holds |
| Additive-error assumption fails (low ρ) | E2/E9 | Re-rank top DP candidates by measured loss; propose a pairwise-interaction correction (thesis contribution) |
| Low-rank slower than dense at B=1 | E3/E7 | Hybrid axis; fuse `U_r V_r^T` at switch time; head-granular rank to keep QKV fusion |
| `r_max` cap costs too much top-mode accuracy | E2/E6 | Raise the cap for attention, keep it low for FFN (DP decides); state the trade explicitly |
| Paper Table 1 not reproducible from published ckpts | E1 | Use the E4a/E4b F arm as reference; cite paper numbers separately |
| Pilot (10% subset) conclusions don't transfer | E4b disagrees with E4a | E4b is authoritative; the pilot only gates spending |
| DAS-6 GPU type slower than assumed | E4a throughput log | Rescale budgets; drop to N=3 / fewer CRs; keep 2 seeds only for CR=0.8 |
| Gated models / dataset access (Llama, ImageNet on DAS-6) | E8 / E1 | Start LM work on GPT-2 (ungated); ask for ImageNet path |
| Jetson unavailable | E7 | Report datacenter B=1 (E3) clearly labelled as a proxy; defer edge claims |
| `transformers` 4.28 ↔ 4.57 drift if a single env is attempted | import errors | Keep two envs; do not port ElasticViT to 4.57 unless forced |

---

## 10. DAS-6 execution protocol (mandatory)

1. **No SSH, no Slurm submission, no remote file changes, no GPU training** until the user provides: (a) SSH
   access/host/path, (b) the remote output/checkpoint directory, and explicitly approves the job.
2. On approval, create one dedicated experiment root (placeholder: `<REMOTE_ROOT>/elastorank/`) with:
   ```
   <REMOTE_ROOT>/elastorank/
     envs/                      # pip freeze per env
     data_lists/                # calib / pilot subset index files (hashed)
     runs/<EXP-ID>_<YYYYMMDD>_<gitsha7>/
        cmd.sh  config.yaml  git_sha.txt  pip_freeze.txt
        slurm-%j.out  slurm-%j.err
        logs/  ckpt/  results.json  plots/
   ```
3. Every job: pinned commit pushed (or bundle copied), seed recorded, `results.json` with all metrics, journal
   entry updated with paths **only after** the job finishes. Results are never written into the journal from memory.
4. Local machine: CPU-only checks (E0, unit tests) only.

---

## 11. Open questions for the user

1. Which checkpoint produced Table 1: `google/vit-base-patch16-224` or `facebook/deit-base-patch16-224` (C6)?
2. Are the published DN checkpoints (surfdrive) already on DAS-6, and for which CRs?
3. ImageNet-1k location on DAS-6; GPU type/partition and per-job limits.
4. For the E4 frozen arm: faithful-to-paper, corrected pos/cls, or both (A11)? Decide after E0.
5. Jetson Orin/Nano access for E7/F2.
6. HF token availability for gated Llama checkpoints (E8).
