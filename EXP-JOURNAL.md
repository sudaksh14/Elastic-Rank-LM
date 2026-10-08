# Experiment Journal — ElastoFormer × FlexRank

Plan: [ELASTOFORMER_IMPLEMENTATION_PLAN.md](ELASTOFORMER_IMPLEMENTATION_PLAN.md) (section numbers below refer to it).

Rules
- Statuses: `PLANNED` → `READY` (code + config done, awaiting approval/resources) → `RUNNING` → `DONE` | `FAILED` | `ABANDONED`.
- Fill **Result / Metrics** only from `results.json` or logs of a finished run. Never from memory or expectation.
- GPU jobs run on DAS-6 only, after access + output directory are provided and the job is approved (plan §10).
- One entry per run. Re-runs get a suffix (`E4a-r2`). Superseded entries stay, with a pointer to the newer one.
- Record the exact commit. Uncommitted code is not a valid run.

---

## Index

| ID | Title | Status | Depends on | Where | Est. GPU-h | Last update |
|---|---|---|---|---|---|---|
| E0 | Env build + CPU audit of pruning metadata / rebuild path | DONE | I-0, I-1 | DAS-6 fatq (CPU) | 0 | 2026-10-08 |
| E1 | Reproduce Table 1 from published ckpts + nestedness audit | READY (queued, job 27682) | E0 | DAS-6 defq GPU | ≤1 | 2026-10-08 |
| E2 | Training-free fronts: pruning vs rank, DP, r_max, additive error | READY (queued, job 27698) | I-2, I-3, E1 | DAS-6 defq GPU | 3–5 | 2026-10-08 |
| E3 | B=1 kernel microbenchmark | READY (queued, job 27699) | I-0 | DAS-6 defq GPU | ≤0.5 | 2026-10-08 |
| E4a | Frozen-prefix vs joint nested sampling (pilot) | READY (queued: smoke 27690, arms 27691-27697) | E0, I-4, I-5, I-6 | DAS-6 defq GPU | ~7 runs × 2–4 | 2026-10-08 |
| E4b | Frozen-prefix vs joint nested sampling (full ImageNet) | PLANNED | E4a | DAS-6 | 40–60 | 2026-10-06 |
| E5 | DP per-layer ratio search, pruning axis | READY (training-free part queued, job 27706; trained part deferred to E4a gate) | E2, I-7, E4a | DAS-6 defq GPU | 3–6 | 2026-10-08 |
| E6 | Rank-axis ViT, joint FlexRank training, capped r_max | READY (queued, jobs 27703-27705: r_cap 1.0/0.5/0.3) | E2, E3, E4a, I-8 | DAS-6 defq GPU | 3 × 3–5 | 2026-10-08 |
| E7 | Axis fork on Jetson (measured latency) | PLANNED | E4–E6, I-9 | Jetson | 0 (device) | 2026-10-06 |
| E8 | FlexRank LM reproduction: GPT-2 → Llama-3.2-1B | PLANNED | I-0(b), I-10 | DAS-6 | 3 → 120 | 2026-10-06 |
| E9 | LM diagnostics: additive error, TTFT/TPOT, KV reuse | PLANNED | E8 | DAS-6 | 3–6 | 2026-10-06 |
| E10 | Scalarization / non-convexity probe (RQ1 gate) | PLANNED | E8 | DAS-6 | 10–20 | 2026-10-06 |
| F1 | Final ViT fronts | PLANNED | E4b, E5, E6 | DAS-6 | TBD | 2026-10-06 |
| F2 | Final edge deployment | PLANNED | E7 | Jetson | 0 | 2026-10-06 |
| F3 | Final LM fronts | PLANNED | E8–E10 | DAS-6 | TBD | 2026-10-06 |
| F4 | Final serving properties (switch, KV reuse, memory) | PLANNED | E9 | DAS-6 / Jetson | TBD | 2026-10-06 |

## Decision log

| Date | Decision | Based on | By |
|---|---|---|---|
| 2026-10-06 | Brainstorm's geometric per-round schedule is replaced by the linear cumulative schedule (plan §1.4 C1) | code (main.py:404) + paper Table 1 FLOPs | analysis, to confirm in E0 |
| 2026-10-08 | Model is **DeiT-B** (`facebook/deit-base-patch16-224`); ViT-B (google) dropped. Resolves plan C6 | user | user |
| 2026-10-08 | Schedule is **linear** (C1 confirmed on real DeiT-B): embed/FFN cumulative pruned fraction within 0.1% of CR·(i+1)/N, geometric off by up to 15% | E0 | analysis |
| 2026-10-08 | Head **count is 12 at every level**; head *size* varies (CR0.8: 12/23/33/43/53/64). Virtual slicing therefore needs a per-level head size, not per-head index sets | E0 | analysis |
| 2026-10-08 | Local env = `elastoformer` (option C); DAS-6 env = shared `/var/scratch/skalra/prune_llm` (already has the pins). Extra pkgs go to `/var/scratch/skalra/elastoSLM/pylibs`, never into the shared env | user + E0 | user |
| 2026-10-08 | GPU nodes node205-208 fully held by another user's 4×15-day jobs (started ~2026-10-06, ~13 d left). Rule from user: GPU-needing jobs are never run on CPU nodes; queue behind and wait. Only CPU-only work (E0, downloads) runs on `fatq` | user | user |
| 2026-10-08 | **Finding C12:** `update_vit_weights_global` fills only shell×shell and core×core blocks of each rebuilt weight matrix. Off-diagonal blocks (shell rows × core cols and vice versa) keep random init (only 64-68% of a rebuilt FFN matrix equals the pretrained slice, tiny-model test). Joint training starts from the full pretrained matrices, so an unmodified F arm vs J would not be a fair init comparison. E4a therefore runs F-faithful (paper) AND F-corrected (all blocks, cls/pos from pretrained slice; `slice_state_dict`) | tests/test_sliced_equivalence.py | analysis |
| 2026-10-08 | I-5 done: `models/sliced_vit.py` (SlicedViT) equals physically sized models at every level (max logit diff < 6e-7, two configs, CPU) | tests/test_sliced_equivalence.py | analysis |
| — | Frozen baseline variant for E4: faithful / corrected / both. E0 shows cls/pos are NOT carried into rebuilt levels (F arm "faithful" inherits that) | E0 | user |

---

## Entry template (copy for every run)

```markdown
### <EXP-ID> — <short title>
- **Status:** PLANNED | READY | RUNNING | DONE | FAILED | ABANDONED
- **Date:** started YYYY-MM-DD · finished YYYY-MM-DD
- **Hypothesis:**
- **Code/config commit:** `<sha>` (branch `<branch>`), config file: `<path>`
- **Command:**
  ```bash
  <exact command / sbatch line>
  ```
- **Hardware:** <local CPU | DAS-6 node/partition, GPU type × count>, wall-clock: <h>, GPU-h: <measured>
- **Dataset:** <name, split, subset list file + hash, calib list file + hash>
- **Settings:** <model, CR, N, axis, sampler, loss, epochs/steps, LR, batch, seed(s), r_max, profiles>
- **Result:** <one-paragraph factual summary>
- **Metrics:**
  | Level / profile | Params | MACs | Top-1 / PPL | Notes |
  |---|---|---|---|---|
- **Logs/checkpoint paths:** `<REMOTE_ROOT>/elastorank/runs/<run-dir>/` (logs/, ckpt/, results.json), wandb: <url>
- **What worked:**
- **What failed:**
- **Interpretation:**
- **Decision:**
- **Next experiment:**
```

---

## Entries

### E0 — Env build + CPU audit of the pruning-metadata / rebuild path
- **Status:** DONE
- **Date:** 2026-10-08 (job 27676, ~4 min on node201)
- **Hypothesis:** H0.1 cumulative keep fraction per level is linear. H0.2 level keep sets are strictly nested. H0.3 head-dim
  pruning removes the same number of dims per head. H0.4 (expected to be rejected) rebuilt Level-2 carries pretrained
  `cls_token`/`position_embeddings`. Also: flexrank core imports/tests pass in the Elastoformer env (py3.10 + shim).
- **Code/config commit:** `beab276-dirty` (uncommitted: `Elastoformer/utils/nested_metadata.py`, `tools/audit_pruning.py`, `slurm/*`)
- **Command:** `sbatch slurm/e0_audit.sbatch` (pytest on FlexRank/flexrank/tests, then `tools/audit_pruning.py --model_name facebook/deit-base-patch16-224 --pruning_ratio {0.4,0.6,0.8} --pruning_steps 5`)
- **Hardware:** DAS-6 node201 (fatq, CPU only), 16 CPUs. Env `/var/scratch/skalra/prune_llm` (py3.10.18, torch 2.7.0+cu126, transformers 4.28.0, torch-pruning 1.6.x, timm 1.0.26)
- **Dataset:** none (L1 importance; random `example_inputs`)
- **Settings:** DeiT-B, CR {0.4, 0.6, 0.8}, N=5, L1, `prune_head_dims=True`, `round_to=1`
- **Result:** All four hypotheses answered. H0.1 true (linear). H0.2 true. H0.3 true. H0.4 rejected as predicted: not carried over.
- **Metrics:**
  | Check | CR 0.4 | CR 0.6 | CR 0.8 |
  |---|---|---|---|
  | H0.1 max abs err vs linear (embed / qkv / ffn) | 0.001 / 0.014 / 0.0003 | 0.001 / 0.015 / 0.0003 | 0.0008 / 0.0125 / 0.0003 |
  | H0.1 max abs err vs geometric (embed / qkv / ffn) | 0.025 / 0.014 / 0.025 | 0.066 / 0.057 / 0.067 | 0.154 / 0.147 / 0.154 |
  | H0.2 nested / disjoint violations | 0 / 0 | 0 / 0 | 0 / 0 |
  | H0.3 per-head uniform, Q=K=V indices | yes | yes | yes |
  | Head count at every level | 12 | 12 | 12 |
  | Head size, Level 1 (core) → Level 6 | 38,43,48,53,58,64 | 25,33,40,48,56,64 | 12,23,33,43,53,64 |
  | Core dims Level-1 (embed/qkv/ffn) | 460/456/1843 | 307/300/1228 | 153/144/614 |
  | H0.4 rebuilt Level-2 `cls_token` cosine vs pretrained slice | 0.10 | 0.05 | 0.12 |
  | H0.4 rebuilt Level-2 `position_embeddings` cosine | ≈0.003 | 0.0016 | −0.0003 |
  | Rebuilt patch-proj & classifier vs pretrained slice | identical | identical | identical |
  | Metadata MB: pruned weight values / int64 indices | 12.7 / 6.0 | 26.9 / 5.4 | 46.3 / 4.8 |
  flexrank core tests under py3.10 + StrEnum shim: **44 passed, 1 skipped** (perf test, needs `RUN_PERF_TESTS=1`).
- **Logs/checkpoint paths:** `/var/scratch/skalra/elastoSLM/runs/E0_20261008_beab276-dirty/` (audit_*.json, logs/)
- **What worked:** audit script and pytest in the shared env; all outputs reproducible from the run dir.
- **What failed:** first submission (27675) died on `set -u` with an unset `PYTHONPATH`; fixed in `slurm/common.sh`.
- **Interpretation:** (1) Brainstorm's geometric schedule is wrong for the code; linear it is. (2) Nestedness holds by
  construction, so virtual slicing is valid. (3) Constant head count + uniform per-head pruning means level k only needs
  `head_size_k`. (4) `cls_token`/`position_embeddings` are fresh random tensors in every rebuilt level (max abs diff ≈ 6.2,
  larger than the pretrained values' scale), so the frozen-prefix ladder is NOT a weight-sharing slice for these tensors.
  (5) Pruned-weight metadata is 5–46 MB per CR, not ~1 MB.
  Unverified: whether *trained* published checkpoints later learned usable pos/cls (E1 checks this directly).
- **Decision:** I-5 uses per-level `head_size` slicing. E4 F-arm: needs the user's call on faithful vs corrected (A11).
- **Next experiment:** E1 (running).

### E1 — Reproduce Elastoformer Table 1 from published checkpoints + nestedness audit
- **Status:** READY (queued, pending GPU)
- **Date:** queued 2026-10-08 as job 27682 (defq, 1 GPU, 4 h limit, all three CRs sequentially). CPU jobs 27678-27680 were cancelled after ~52 min on the user's rule that GPU-needing work is never run on CPU nodes; their partial output is not used.
- **Hypothesis:** Published Level-1…6 checkpoints reproduce Table 1 Top-1 within ±0.3 pp (CR=0.8 targets: 51.19 /
  64.14 / 69.97 / 72.46 / 73.61 / 74.65; CR=0.6: 72.45 / 74.22 / 75.26 / 75.79 / 76.09 / 76.41; CR=0.4: 79.95 / 80.13 /
  80.24 / 80.35 / 80.20 / 80.13; base 81.74). Encoder Linear/LN weights are exactly nested across levels. Classifier
  (and probably pos/cls) are not.
- **Code/config commit:** `beab276-dirty` (`Elastoformer/tools/eval_checkpoints.py`, `slurm/e1_eval.sbatch`)
- **Command:** `sbatch slurm/e1_gpu.sbatch`
- **Hardware:** DAS-6 defq, 1 GPU (A40/A10, whichever frees first); node205-208 currently held by another user's jobs
- **Dataset:** ImageNet-1k val (50k)
- **Settings:** DeiT-B, checkpoints from the SURFdrive release downloaded to `/var/scratch/skalra/elastoSLM/checkpoints/deitb_cr{0.4,0.6,0.8}` (job 27674, 18 files, SHA256SUMS.txt); eval transform resize 256 / crop 224, bicubic
- **Result:** pending. Smoke test (job 27677, 256 val images, CR0.8, L1+L6): rebuilt GFLOPs 0.60 / 16.86 = paper exactly; top-1 48.8 / 71.5 vs paper 51.19 / 74.65 (n=256, noise ≈ ±3 pp, not evidence either way). Published CR0.8 Level-1 dims are 152/96/608 (embed/qkv/ffn); the current code's core is 153/144/614, so the released checkpoints came from a slightly different code version.
- **Metrics:** Top-1/Top-5, params, MACs per level; per-tensor nestedness deltas
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E2

### E2 — Training-free Pareto fronts: pruning vs rank, uniform vs DP, r_max, additive error
- **Status:** READY (queued, job 27698; includes an in-job real-data smoke that aborts on failure)
- **Date:** queued 2026-10-08
- **Hypothesis:** Full-rank SVD/DataSVD reproduces base accuracy (S1). DataSVD ≥ SVD and DP ≥ uniform at matched params.
  Additive-error Spearman ρ at ViT-B is reported (no prior expectation beyond FlexRank's 4-layer MNIST check).
- **Code/config commit:** `beab276-dirty` (`Elastoformer/eval_fronts.py`, `utils/flexrank_bridge.py`, `utils/nested_metadata.py`, `slurm/e2_run.sbatch`). CPU-tested on a tiny ViT: S1 full-rank DataSVD == base (max logit diff 1.2e-6), analytic params == actual.
- **Command:** `sbatch slurm/e2_run.sbatch` → `python eval_fronts.py --out $RD --crs 0.4 0.6 0.8 --N 5 --calib_n 4096 --probe_n 1024 --val_n 10000 --prune_full_val --decomps svd datasvd --caps 1.0 0.5 0.3 --profiles uniform dp --dp_cuts 12 --dp_min_p 0.02 --n_random 30`
- **Hardware:** DAS-6 defq, 1 GPU
- **Dataset:** ImageNet-1k train (4,096 random-image calib set, seed 0, index list saved next to the cached fp16 tensor in `/var/scratch/skalra/elastoSLM/data_lists/`; first 1,024 used for DP probing and the additive check) + val (fixed class-balanced 10k subset for the rank arm; full 50k for the pruning ladders)
- **Settings:** prune: L1, CR {0.4,0.6,0.8}, N=5, no FT. Rank: 72 encoder Linears, {SVD, DataSVD} × {uniform, DP} ×
  r_max/min(m,n) {1.0, 0.5, 0.3}; 30 random profiles for the additive-error check
- **Result:** —
- **Metrics:** Top-1 vs params/MACs/storage; calib loss; ρ, τ
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E4a (recipe), E5/E6 (profiles, r_max)

### E3 — B=1 kernel microbenchmark (datacenter proxy)
- **Status:** READY (queued, job 27699)
- **Date:** queued 2026-10-08
- **Hypothesis:** For ViT-B prefill shapes, two-GEMM low-rank beats matched-param dense-pruned only below
  r/min ≈ 0.3–0.4. For LM decode shapes, latency tracks bytes read.
- **Code/config commit:** `beab276-dirty` (`Elastoformer/tools/bench_kernels.py`, `slurm/e3_bench.sbatch`); CPU-checked that ODLinear SLICED/GAR constructions work and GAR == masked forward (8e-6)
- **Command:** `sbatch slurm/e3_bench.sbatch`
- **Hardware:** DAS-6 defq, 1 GPU (A40 or A10; record which: it changes the crossover)
- **Dataset:** synthetic tensors
- **Settings:** shapes per plan §5 E3; dense / dense-pruned / sliced / GAR / materialised; fp16, bf16; CUDA graphs on/off
- **Result:** —
- **Metrics:** median/p90 latency, crossover rank
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E6 constraints / E7 priority

### E4a — Frozen-prefix vs joint nested sampling (pilot, pruning axis)
- **Status:** READY (all jobs pending GPU; `afterok` gated on the GPU smoke test 27690)
- **Date:** queued 2026-10-08
- **Hypothesis:** At matched compute, joint nested sampling (J) beats frozen-prefix growth (F). At CR=0.8: ≥ +2.0 pp
  mean Top-1 over levels and no level worse than −0.5 pp. At CR=0.5: J ≥ F − 0.3 pp everywhere. KD ≥ CE.
- **Code/config commit:** `beab276-dirty` (`Elastoformer/train_nested.py`, `models/sliced_vit.py`, `main.py` flags `--train_subset/--corrected_init/--eval_every/--results_dir`, `trainer.py --eval_every`, `slurm/e4_smoke.sbatch`, `slurm/e4a_run.sbatch`, `slurm/submit_e4a.sh`)
- **Command:** see plan §5 E4a (`main.py … --pruning_steps 3 --core_epochs 10 --epochs 10 --train_subset 0.1` vs
  `train_nested.py … --total_epochs 40 --train_subset 0.1`)
- **Hardware:** DAS-6, multi-GPU DDP (TBD)
- **Dataset:** 10% class-balanced ImageNet-1k train subset (list hashed), full val
- **Settings:** 7 runs: F-faithful, F-corrected (C12) and J at CR 0.5 and 0.8, plus J+KD at CR 0.8. N=3 (4 levels); 1 seed; pretrained DeiT-B init; same L1 index sets. Common: 10% class-balanced train subset (seed 0, list saved), batch 128, AdamW lr 5e-5, fp16 AMP + GradScaler, grad-clip 1.0, timm Mixup/CutMix with label smoothing 0.11 (module-level, same object in both arms), no EMA, no repeated-aug sampler, **no stochastic depth in F** (J has none), warmup 1 epoch. F: core wd 0.05 then 0; J: wd 0. Sampler uniform (one level per step). Hyper-parameters are copied from the README recipe, not tuned for either arm
- **Result:** —
- **Metrics:** Top-1 per level, mean over levels, acc–MACs AUC, GPU-h and img/s per arm, sampler coverage, frozen-hash check
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E4b if success; ablations A1/A9/A2 if J≈F; E6 if J<F

### E4b — Frozen-prefix vs joint nested sampling (full ImageNet-1k)
- **Status:** PLANNED
- **Date:** —
- **Hypothesis:** E4a's ordering holds at full scale with N=5.
- **Code/config commit:** —
- **Command:** as E4a without `--train_subset`, `--pruning_steps 5 --core_epochs 5 --epochs 5`
- **Hardware:** DAS-6, multi-GPU
- **Dataset:** ImageNet-1k full train/val
- **Settings:** CR {0.5, 0.8} (+0.4 if budget allows); arms F, best-of-{J, J+KD}; 2 seeds at CR 0.8
- **Result:** —
- **Metrics:** as E4a, plus comparison to the published 50-epochs/stage ladder (E1)
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E5, E6

### E5 — DP per-layer ratio search for the pruning axis
- **Status:** READY for the training-free comparison (job 27706, in-job smoke first). The trained comparison (DP profiles through `train_nested.py`) is NOT built: it needs nested DP profiles and depends on E4a deciding the schedule.
- **Date:** queued 2026-10-08
- **Hypothesis:** DP-allocated internal-dim ratios beat the uniform linear schedule at matched params (≥ +1.0 pp
  training-free at ≥3 of the lowest 4 levels; ≥ +0.5 pp after training). Activation-aware importance ≥ L1.
- **Code/config commit:** `beab276-dirty` (`Elastoformer/eval_prune_dp.py`, `slurm/e5_run.sbatch`). FlexRank's DP is reused unchanged by wrapping each prunable width group as an `ODLayer` (24 groups: 12 FFN + 12 attention head-dim widths; residual width E fixed per ladder level). CPU-tested on a tiny ViT.
- **Command:** `sbatch slurm/e5_run.sbatch` → `python eval_prune_dp.py --crs 0.8 0.6 --N 5 --calib_n 4096 --probe_n 1024 --val_n 10000 --dp_cuts 12 --dp_min_p 0.05`. Importance is group-L1 only; the activation-aware variant (A4) is not implemented yet.
- **Hardware:** DAS-6
- **Dataset:** as E2 / E4a
- **Settings:** CR 0.8, N=5 targets; residual width uniform; internal dims (FFN intermediate, QKV head-dims) per layer
- **Result:** —
- **Metrics:** Top-1 per level at matched params (untrained, trained)
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E6 / F1

### E6 — Rank-axis ViT with joint FlexRank training and capped r_max
- **Status:** READY (queued: jobs 27703/27704/27705 for r_cap 1.0/0.5/0.3; each runs an in-job smoke first)
- **Date:** queued 2026-10-08
- **Hypothesis:** At matched params, nested low-rank (DataSVD + DP + KD, joint) beats the best pruning-axis joint
  model at ≥4/6 levels, with stored params ≤ 1.0× dense at r_max=0.5.
- **Code/config commit:** `beab276-dirty` (`train_nested.py --axis rank`, `utils/flexrank_bridge.py`, `slurm/e6_run.sbatch`); CPU smoke on tiny ViT passes for both axes
- **Command:** `torchrun … train_nested.py --axis rank --decomp datasvd --r_max <0.5|1.0> --profiles dp --loss kd …`
- **Hardware:** DAS-6
- **Dataset:** as E4a
- **Settings:** r_cap {1.0, 0.5, 0.3} of min(in,out); DataSVD (4,096 calib images, Gram matrices); DP over 72 encoder Linears (12 cuts, min_p 0.02, 1,024 probe images); uniform sampler; KD from the frozen dense DeiT-B (kl=1). Params follow FlexRank's GAR accounting; stored params are reported separately. With r_cap<1 the top mode cannot reproduce the dense model
- **Result:** —
- **Metrics:** Top-1 per level; stored MB; params; MACs; predicted latency (E3)
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E7

### E7 — Axis fork on Jetson (measured latency)
- **Status:** PLANNED
- **Date:** —
- **Hypothesis:** Permuted contiguous pruning gives the lower B=1 latency per level. Rank/GAR is competitive only at
  aggressive levels. Both switch in ≈0 ms vs the paper's 50.4 ms.
- **Code/config commit:** — (needs I-9)
- **Command:** `python tools/bench_latency.py --ckpt <…> --device orin --trials 50 --warmup 10`
- **Hardware:** Jetson Orin / Nano (access TBD, not DAS-6)
- **Dataset:** synthetic inputs (latency); accuracy taken from E4–E6
- **Settings:** fp16, B=1; candidates: permuted pruning, rank-GAR, hybrid
- **Result:** —
- **Metrics:** per-level latency (median/p90), switch latency, peak memory, storage
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** F1/F2

### E8 — FlexRank LM reproduction: GPT-2 → Llama-3.2-1B
- **Status:** PLANNED
- **Date:** —
- **Hypothesis:** The released `flextrain` pipeline reproduces FlexRank's published LM front within ±3% PPL per budget
  (record the target numbers from `related papers/FlexRank.pdf` here **before** running).
- **Code/config commit:** — (FlexRank as-is; I-10 scripts for DAS-6)
- **Command:** `bash scripts/flexrank_nlp.sh model=gpt2 dataset=finewebedu_10bt_gpt2 steps=1000 …` (BASE_DIR → run dir)
- **Hardware:** DAS-6, multi-GPU
- **Dataset:** FineWeb-Edu sample-10BT (streaming), seq len 1024
- **Settings:** DataSVD, DP, ce=0/kl=1, PredefinedModels sampler, lm_head excluded; pilot 1000 steps → full 6000 → Llama-1B
- **Result:** —
- **Metrics:** PPL per profile; lm-eval commonsense; params with and without lm_head
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** E9, E10

### E9 — LM diagnostics: additive error at 1B, TTFT/TPOT, KV-cache reuse
- **Status:** PLANNED
- **Date:** —
- **Hypothesis:** Additive-error ρ is lower at 1B than at ViT-B. TPOT scales with active params, TTFT with FLOPs.
  A KV cache built at profile A is reusable at an adjacent profile B with < 1% ΔPPL.
- **Code/config commit:** —
- **Command:** TBD (eval scripts on E8 checkpoints)
- **Hardware:** DAS-6, 1 GPU
- **Dataset:** FineWeb-Edu held-out
- **Settings:** 30 random profiles (ρ); B=1 generate (TTFT/TPOT); A→B cache reuse for adjacent and distant profiles
- **Result:** —
- **Metrics:** ρ, τ; TTFT, TPOT, peak memory; ΔPPL
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** A10 / F3 / F4

### E10 — Scalarization / non-convexity probe (thesis RQ1 gate)
- **Status:** PLANNED
- **Date:** —
- **Hypothesis:** Open. Either the per-budget front under weighted-sum training shows a non-convex region at aggressive
  budgets (Stage B justified), or it is convex (pivot to RQ3/RQ4).
- **Code/config commit:** —
- **Command:** 4–5 × `flexrank_nlp.sh … steps=1000` with different per-budget sampling weights (mechanism TBD in I-10)
- **Hardware:** DAS-6
- **Dataset:** FineWeb-Edu
- **Settings:** GPT-2; α ∈ {uniform, small-heavy, large-heavy, 2 mixed}
- **Result:** —
- **Metrics:** per-budget PPL; hypervolume; convex-hull gap
- **Logs/checkpoint paths:** —
- **What worked:** —
- **What failed:** —
- **Interpretation:** —
- **Decision:** —
- **Next experiment:** thesis Stage B or pivot

### F1 — Final ViT fronts
- **Status:** PLANNED — scope fixed after E4b/E5/E6 (plan §8).

### F2 — Final edge deployment
- **Status:** PLANNED — scope fixed after E7 (plan §8).

### F3 — Final LM fronts
- **Status:** PLANNED — scope fixed after E8–E10 (plan §8).

### F4 — Final serving properties
- **Status:** PLANNED — scope fixed after E9 (plan §8).
