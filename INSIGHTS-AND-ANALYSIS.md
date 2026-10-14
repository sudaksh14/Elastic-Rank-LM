# Insights and Analysis: ElastoFormer x FlexRank

Last updated: 2026-10-14. Read this first. It records what was measured, what it means, what is still unknown, and the next steps. The per-experiment record (dates, job IDs, settings) is in [EXP-JOURNAL.md](EXP-JOURNAL.md). The plan is in [ELASTOFORMER_IMPLEMENTATION_PLAN.md](ELASTOFORMER_IMPLEMENTATION_PLAN.md).

## 1. Where things stand

| Area | State | Confidence |
|---|---|---|
| Published Elastoformer ladder (DeiT-B) reproduces | Yes, within 0.30 pp (E1) | High |
| Pruning schedule is linear, not geometric | Yes (E0) | High |
| Trained ladder levels are nested for encoder weights; cls and position embeddings are not | Yes (E0, E1) | High |
| Joint nested training (E4a) vs frozen-prefix | **Yes on the pruning axis**: +9.5 pp mean at CR 0.8, +12.9 pp at CR 0.5 vs the corrected baseline. Exception: top level −0.5 to −1.2 pp. Strict criteria not met | Medium (pilot, one seed, 10% data) |
| Low-rank (DataSVD) beats pruning without training | Yes at every budget below ~70% (E2) | High for the untrained comparison |
| Trained low rank (E6) beats pruning | Yes against the matched JKD arm: +30 to +37 pp at 19 M, +4.5 to +6.4 pp at 46 M; both collapse at 3.6 M. Hyper-parameter equality to be confirmed | Medium-low (one seed) |
| DP per-layer allocation helps | **Unknown.** Budgets now match (<0.01%), but untrained pruning is at chance for levels 1-4, so the test says nothing. Needs trained comparison (E5b) | — |
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

### 2.6 Single-level DP allocation (E5)
- Uniform widths with the same per-layer ranking reproduce the Elastoformer ladder (sanity check passes).
- **Rerun with budget fill (job 27785):** DP params match the budget to <0.01% at all 10 level/CR points (the unfixed run spent 37 M of 61 M).
- **Training-free top-1 is at chance for levels 1-4** (0.1-0.4%) for uniform, DP and the published schedule alike. Only level 5 (61-67 M) is above chance: DP 1.52 vs 0.65 (CR 0.8) and 2.88 vs 1.15 (CR 0.6), still ≤3%.
- DP lowers validation loss at all 10 points, but every loss is near ln 1000 = 6.91, so this is weak evidence.
- **Interpretation:** the test cannot decide DP vs uniform for pruning. DP profiles are extreme (some layers at full width, others at width 1), which may or may not train well. **Action:** E5b, train DP profiles with the E4a JKD recipe.

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

### 2.8 Joint nested training vs frozen-prefix (E4a pilot, pruning axis, DeiT-B, 10% ImageNet, 40 level-epochs equivalent)
Mean top-1 over the 4 levels (val 50k, base 81.80). Level params at CR 0.8: 3.6 / 19.1 / 46.5 / 86.6 M; at CR 0.5: 22.1 / 38.6 / 60.2 / 86.6 M.

| CR | F (faithful) | Fc (corrected) | J (CE) | JKD (KD) |
|---|---|---|---|---|
| 0.8 | 8.2 | 36.3 | 45.8 | 47.5 |
| 0.5 | 21.8 | 55.8 | 68.7 | not run |

Per level, J (CR 0.8): 0.5 / 31.0 / 71.9 / 79.7; Fc: 1.5 / 8.4 / 55.2 / 80.2. J (CR 0.5): 49.8 / 69.3 / 76.2 / 79.6; Fc: 16.5 / 50.6 / 75.1 / 80.8.

- **Joint sampling wins everywhere except the top level.** Gains are largest at the small and middle levels (+17 to +33 pp), where frozen-prefix leaves the shared core at what it learned in the first stage.
- **The top level is the exception:** J is 0.5 pp (CR 0.8) and 1.2 pp (CR 0.5) below Fc. Likely cause: J gives L4 a quarter of the steps while Fc's last stage trains only L4. Not tested; the weighted or sandwich sampler is the test.
- **Strict criteria fail** (CR 0.8: L1 −1.0 pp, L4 −0.5 pp; CR 0.5: L4 −1.2 pp). L1 at CR 0.8 is at chance in every arm, so that miss carries no information.
- **F-faithful is not a usable baseline.** Its top level is 14.6% (CR 0.8) and 27.3% (CR 0.5), because of random off-diagonal blocks and cls/pos (C12). Compare against Fc only.
- **KD helps:** JKD − J = +1.7 pp mean, +1.9 pp at the top level, at 1.5x wall time.
- **The smallest level (3.6 M) does not train under any recipe here** (loss near ln 1000). That is the same level that fails on the rank axis (E6).
- **Rank vs pruning:** the JKD arm gives pruning points at 19.05 M (33.7%) and 46.5 M (73.8%) under the E6 recipe. Rank (E6) gets 64-71% and 78.3-80.2% at about the same budgets. This supports rank over pruning at mid budgets, subject to confirming that the recipes are identical and adding seeds.
- **Caveats:** one seed; 10% data; hyper-parameters copied from the README, not tuned for either arm; F arms run through a different pipeline and GPU type, so compute is matched by epochs.

## 3. Caveats that limit every comparison above
1. **Budgets and training differ.** E1 is the published 50-epoch CE ladder on full ImageNet. E6 is 40 level-epochs on 10% of ImageNet with KD. E4a JKD (pruning, same subset, schedule, KD and sampler as E6) is the closest match, but it is one seed and its hyper-parameters need a diff against E6.
2. **Single seed everywhere.** Differences under ~0.5 pp should be treated as noise.
3. **Validation sets differ.** E2 uses a 10k class-balanced subset. E1 and E6 use the full 50k.
4. **Smoke-test numbers are meaningless for accuracy.** They use 1% of the data and a debug path.
5. **Hardware:** A40 and A10 only. No Jetson measurements.

## 4. Bugs found and fixed (2026-10-12 to 2026-10-13)
- **No job used the GPU before 2026-10-12.** The CUDA toolkit module was not loaded, so torch silently fell back to CPU. Fixed: every batch script sources `slurm/common.sh`, which loads `cuda12.1/toolkit/12.1` and activates `prune_llm`. Verified with `env_check`.
- **Missing plot folder** in `main.py` crashed the F arm. Fixed by creating `./saves/plots`.
- **Corrected-init device mismatch.** `slice_state_dict` mixed CPU index tensors with CUDA weights. Fixed by moving indices to the weights' device (`models/sliced_vit.py`). Verified on GPU by smoke 27786 (2026-10-14).
- **E5 under-spend.** Fixed with `utils/budget.py` (greedy fill). Verified on GPU by the E5 rerun (DP params within 0.01% of budget).
- **Time limits.** Smoke and E2 time limits were too short for CPU runs. Current rule: each job's limit is 2x its pessimistic expected runtime, max 150 h.

## 5. Open questions (ordered by importance)
1. **Can the top-level gap be closed without losing the joint-training gain?** J is 0.5-1.2 pp below Fc at L4. Test: weighted or sandwich sampler.
2. **Does the E4a ordering hold at full scale and across seeds?** E4b, N=5, two seeds. Only after question 1 fixes the recipe.
3. **Does DP allocation beat uniform after training?** E5b. Training-free evidence is empty.
4. **Does rank beat pruning with identical recipes and seeds?** Diff E6 and JKD configs, then add seeds.
5. **Can the smallest level (3.6 M) be trained at all?** It fails on both axes. Higher lr, longer schedule, or a narrower residual width on the rank axis.
6. **Is the additive-error assumption true at ViT-B and at 1B+ (RQ5)?** Needs a larger probe set.
7. **Does low rank help on edge hardware?** Jetson measurement (E7).
8. **Does TF32 explain the S1 logit gap?** One short rerun.

## 6. How to reproduce or continue
- **Repository:** `slurm/common.sh` sets the environment. Every batch script sources it. Scripts live in `slurm/`, Python code in `Elastoformer/` and `FlexRank/`.
- **DAS-6 layout:** code is synced to `/var/scratch/skalra/elastoSLM/repo` by `slurm/sync.sh`. Runs go to `/var/scratch/skalra/elastoSLM/runs/<EXP>_<date>_<sha>`. Checkpoints go to `/var/scratch/skalra/elastoSLM/checkpoints` (DeiT-B ladders from SURFdrive are there). Home has a small quota; do not store large files there.
- **Node facts (probe 27711, GPU node):** 128 GB RAM, `/dev/shm` 63 GB (RAM), `/local` 1.5 TB NVMe SSD but **not writable by users** (root-owned), NFS about 330 MB/s writes. Data staging is deferred until the admins or the user decide how to use `/local` or `/dev/shm`.
- **Job rules:** GPU work is always queued, never run on CPU nodes. One job owns a whole node (`#SBATCH --exclusive`). Time limit = 2x pessimistic runtime.
- **Local development:** the `elastoformer` conda env (Python 3.10) on Windows runs the CPU smoke tests with `--dummy` (tiny model, random data).

## 7. Next steps
See the "Next steps" section at the end of [EXP-JOURNAL.md](EXP-JOURNAL.md), which is kept up to date.
