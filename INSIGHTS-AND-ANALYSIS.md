# Insights and Analysis: ElastoFormer x FlexRank

Last updated: 2026-10-13. Read this first. It records what was measured, what it means, what is still unknown, and the next steps. The per-experiment record (dates, job IDs, settings) is in [EXP-JOURNAL.md](EXP-JOURNAL.md). The plan is in [ELASTOFORMER_IMPLEMENTATION_PLAN.md](ELASTOFORMER_IMPLEMENTATION_PLAN.md).

## 1. Where things stand

| Area | State | Confidence |
|---|---|---|
| Published Elastoformer ladder (DeiT-B) reproduces | Yes, within 0.30 pp (E1) | High |
| Pruning schedule is linear, not geometric | Yes (E0) | High |
| Trained ladder levels are nested for encoder weights; cls and position embeddings are not | Yes (E0, E1) | High |
| Joint nested training (E4a) vs frozen-prefix | **Not yet measured** (smoke test failed; fixed, awaiting rerun) | — |
| Low-rank (DataSVD) beats pruning without training | Yes at every budget below ~70% (E2) | High for the untrained comparison |
| Trained low rank (E6) beats the pruning ladder | Suggested at mid and large budgets, **not a matched comparison** | Low |
| DP per-layer allocation helps | **Unknown.** Previous run was not budget-matched; fill fix now in place (E5 rerun pending) | — |
| Additive-error assumption (RQ5) | Not supported at ViT-B scale in the untrained probe (rho 0.09). Probe is noisy | Low |
| Low rank speeds up single-token decode | Yes on A40 for large layers at rank up to ~0.5; parameter-matched dense pruning is as fast | Medium (one GPU, one run, no Jetson) |
| Edge (Jetson) latency | Not measured | — |

## 2. Measured results and what they mean

### 2.1 Reproduction of the published ladder (E1)
- All 18 levels (CR 0.4/0.6/0.8, levels 1–6) match Table 1 within 0.30 pp. Every level is slightly low (median -0.2 pp). FLOPs match exactly.
- **Interpretation:** the released checkpoints are the paper's models. The small negative offset is most likely an evaluation-transform detail (resize/crop/interpolation). It is not verified.
- **Use:** E1 is the reference for every paper comparison. Report the offset.

### 2.2 Nestedness of the published models (E0, E1)
- Encoder projections, patch embedding and FFN weights are nested by value: 1.00 of the smaller level's values appear in the next level.
- `cls_token` (0.00) and `position_embeddings` (~0.00) are **not** carried across levels. In the rebuild they start fresh (E0 max abs diff ~6).
- **Consequence:** the frozen-prefix pipeline is not a pure weight-sharing family. The "faithful" frozen arm inherits this. The "corrected" arm (copy pretrained cls/pos and all off-diagonal blocks) isolates the schedule. This is why E4a has both F arms.

### 2.3 Untrained fronts: pruning vs low rank (E2)
- Untrained pruning ladders collapse: top-1 0.1–7% at every level below the full model.
- Untrained low rank with DataSVD degrades gracefully: 81.5% at 73 M (cap 1.0), 79.6% at 54.7 M (cap 0.5), 66.4% at 36 M (cap 0.3).
- DataSVD beats SVD at every capped budget (66.4% vs 54.4% at cap 0.3).
- **Interpretation:** input-aware decomposition matters most where truncation hurts. Low rank is a far better starting point than structured pruning when no training is done.
- **Caveats:** single seed, 10k class-balanced validation subset.

### 2.4 Checks in E2
- **S1 (full-rank DataSVD reproduces the base):** top-1 identical (81.74). Max logit difference 0.0099, above the 1e-3 threshold. Likely TF32 on the GPU, not verified. **Action:** rerun with TF32 disabled.
- **S4 (analytic parameter count):** 86,567,656 = actual.
- **Additive error (RQ5):** Spearman 0.09, Kendall 0.07 on 30 random profiles. The probe set is 1,024 images and the base loss difference is small, so the estimate is noisy. Treat as "not supported yet", not "refuted".

### 2.5 Kernel benchmark (E3, A40, fp16)
- **Single token with CUDA graphs:** for large layers, two-GEMM low rank is faster than dense up to rank ~0.5. 2048x8192: dense 79.9 µs; low rank 42.0 µs (r=0.3), 54.3 µs (r=0.5), 103.4 µs (full rank). 8192x2048 shows the same pattern.
- **Parameter-matched dense pruning is as fast as low rank** (41.0 vs 42.0 µs). GAR beats the two-GEMM form at full rank (90 vs 103 µs).
- **Prefill (197 tokens, no graphs):** low rank is 1.2–1.5x slower than dense at rank 0.3–0.5 for 768x3072.
- **Small shapes (2048x512) are launch-bound** at ~20 µs. Use CUDA-graph rows for kernel comparisons.
- **Interpretation:** the brainstorm's worry that low rank is slower at batch 1 does not hold on A40 for decode. But on speed alone it does not beat structured pruning, so the case for low rank rests on accuracy (2.3), not on latency. Edge numbers (Jetson) are still missing.

### 2.6 Single-level DP allocation (E5, not yet budget-matched)
- Uniform widths with the same per-layer ranking reproduce the Elastoformer ladder (sanity check passes).
- **The previous DP run was not budget-matched.** It spent 37 M of a 61 M budget, so its accuracy comparison with uniform is meaningless.
- **Fixed in code (not yet run on GPU):** `utils/budget.py` fills unused budget greedily, one unit at a time, using the DP's own probe estimates. On the tiny CPU test the gap is <0.1% at every level.

### 2.7 Trained low rank (E6, 40 level-epochs on a 10% ImageNet subset, KD, one seed)
| Cap | L1 (3.6 M) | L2 (~19 M) | L3 (36–46 M) | L4 (full capped) |
|---|---|---|---|---|
| 1.0 | 10.96 | 64.11 | 78.29 (45.1 M) | 81.17 (86.6 M) |
| 0.5 | 11.38 | 68.65 | 80.19 (46.4 M) | 80.77 (54.7 M) |
| 0.3 | 12.55 | 70.65 | 78.76 (36.1 M) | 78.76 (36.1 M, same as L3) |

- **Training matters most where truncation hurts:** at cap 0.3 untrained 66.4% (36 M) becomes 78.8% trained. At cap 0.5 the gain is ~1 pp.
- **Mid budgets look strong:** at ~45 M, rank reaches 78–80%, against ~72% for the published pruning ladder at 39–59 M. Not like-for-like (see §3).
- **The smallest level collapses, and this is structural, not a search failure.** The rank model keeps the residual width at 768. Its fixed parts (classifier 0.77 M, patch embedding ~0.6 M, layer norms) use ~1.55 M of the 3.6 M budget. The pruning ladder can shrink the residual width to 152 at that level. Even with every rank at its minimum the rank model is over budget at 3.6 M (the code now reports `reached_total > target_total` for this case).
- **At cap 0.3 the top mode duplicates level 3**, because the cap limits the full model to 36.1 M.

## 3. Caveats that limit every comparison above
1. **Budgets and training differ.** E1 is the published 50-epoch CE ladder on full ImageNet. E6 is 40 level-epochs on 10% of ImageNet with KD. Only E4a (pruning, joint vs frozen, same budget) gives a fair comparison, and it has not run yet.
2. **Single seed everywhere.** Differences under ~0.5 pp should be treated as noise.
3. **Validation sets differ.** E2 uses a 10k class-balanced subset. E1 and E6 use the full 50k.
4. **Smoke-test numbers are meaningless for accuracy.** They use 1% of the data and a debug path.
5. **Hardware:** A40 and A10 only. No Jetson measurements.

## 4. Bugs found and fixed (2026-10-12 to 2026-10-13)
- **No job used the GPU before 2026-10-12.** The CUDA toolkit module was not loaded, so torch silently fell back to CPU. Fixed: every batch script sources `slurm/common.sh`, which loads `cuda12.1/toolkit/12.1` and activates `prune_llm`. Verified with `env_check`.
- **Missing plot folder** in `main.py` crashed the F arm. Fixed by creating `./saves/plots`.
- **Corrected-init device mismatch.** `slice_state_dict` mixed CPU index tensors with CUDA weights. Fixed by moving indices to the weights' device (`models/sliced_vit.py`). Not yet verified on GPU; the E4a smoke test is the check.
- **E5 under-spend.** Fixed with `utils/budget.py` (greedy fill). Verified on CPU; GPU rerun pending.
- **Time limits.** Smoke and E2 time limits were too short for CPU runs. Current rule: each job's limit is 2x its pessimistic expected runtime, max 150 h.

## 5. Open questions (ordered by importance)
1. **Does joint nested training beat frozen-prefix on the pruning axis?** This is the central claim of the brainstorm (§1.2). Test: E4a, after the smoke test passes.
2. **Does DP allocation beat uniform at matched budgets?** E5 rerun.
3. **Does rank beat pruning at matched budgets, with the same training?** Needs a matched pruning arm trained the same way as E6. E4a's J arm plays that role for the pruning axis.
4. **Is the additive-error assumption true at ViT-B and at 1B+ (RQ5)?** Needs a larger probe set and a second run with more probe images.
5. **Does low rank help on edge hardware?** Jetson measurement (E7). Needed before any edge claim.
6. **Does the TF32 effect explain the S1 logit gap?** One short rerun with TF32 disabled.

## 6. How to reproduce or continue
- **Repository:** `slurm/common.sh` sets the environment. Every batch script sources it. Scripts live in `slurm/`, Python code in `Elastoformer/` and `FlexRank/`.
- **DAS-6 layout:** code is synced to `/var/scratch/skalra/elastoSLM/repo` by `slurm/sync.sh`. Runs go to `/var/scratch/skalra/elastoSLM/runs/<EXP>_<date>_<sha>`. Checkpoints go to `/var/scratch/skalra/elastoSLM/checkpoints` (DeiT-B ladders from SURFdrive are there). Home has a small quota; do not store large files there.
- **Node facts (probe 27711, GPU node):** 128 GB RAM, `/dev/shm` 63 GB (RAM), `/local` 1.5 TB NVMe SSD but **not writable by users** (root-owned), NFS about 330 MB/s writes. Data staging is deferred until the admins or the user decide how to use `/local` or `/dev/shm`.
- **Job rules:** GPU work is always queued, never run on CPU nodes. One job owns a whole node (`#SBATCH --exclusive`). Time limit = 2x pessimistic runtime.
- **Local development:** the `elastoformer` conda env (Python 3.10) on Windows runs the CPU smoke tests with `--dummy` (tiny model, random data).

## 7. Next steps
See the "Next steps" section at the end of [EXP-JOURNAL.md](EXP-JOURNAL.md), which is kept up to date.
