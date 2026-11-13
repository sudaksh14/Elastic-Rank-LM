# Implementation plan: rank × bandwidth factorization in FlexRank

Date: 2026-11-13. Implements idea A (and the fallback idea B) from [paper_plan.md](paper_plan.md). Nothing here is implemented yet.

All paths are relative to `FlexRank/`. `core/` below stands for `flexrank/src/flexrank/`.

## 0. Design in one page

**Layer math.** `C_n` is the orthonormal DCT-II matrix. `P` is a permutation or rotation chosen by alignment (identity when unused).

```
x_hat = (C_n P_in x)[:k_i]            # band-limited input
h     = B[:r, :k_i] x_hat             # r × k_i
y_hat = A[:k_o, :r] h                 # k_o × r
y     = P_out^T C_m^T pad(y_hat) + b  # back to the neuron basis
```

This forward is LYRA's factored pass (its Eqs. 3–5 with `S = A·B`), applied to the whole weight and with an alignment step in front. Use the LYRA paper in `related papers/spectral/` as the reference for the algebra and for the cost `O(b(n·k_i + k_i·r + r·k_o + k_o·m))`.

**Key choice: keep the `ODLayer` contract.** The DP, the samplers and save/load all talk to a layer through one integer, `inner_dim` ([base.py](../FlexRank/flexrank/src/flexrank/layers/base.py)). The new layer keeps that contract. Its `inner_dim` indexes a per-layer **chain**: a monotone list of `(r, k_i, k_o)` triples from smallest to largest. Everything downstream then runs unchanged.

**Reuse, do not rewrite.**

| Need | Existing code |
|---|---|
| Factor storage, rank mask, `from_uv`, pruning | `ODLinear` in `core/layers/linear.py` |
| Forward kernels | `od_dense_linear`, `od_sliced_linear` in `core/layers/functional.py` |
| Input Gram matrices | `collect_model_grams` in `core/layers/gram.py` |
| SVD / DataSVD of a matrix | `compute_decomp` in `core/layers/decomposition.py` |
| Layer replacement loop | `SVDTrainer._decompose_layers` in `core/trainers/svd_trainer.py` |
| Profiling and DP | `profile_model`, `fast_profile_model` in `core/profiles/profiling.py`; `minimize_error_with_any_drops` in `core/profiles/dp.py` |
| GAR deployment form | `make_gar` in `core/layers/gar.py` |

## Phase 0: offline diagnostics (S0, no training)

New script `scripts/spectral_diagnostics.py`. It decides whether the rest is built.

- **Input:** a model name, a calibration split, a list of layers.
- **Per layer, compute the error-vs-bits curve for:**
  - SVD and DataSVD truncation (`compute_decomp`).
  - DCT low-frequency block, LYRA's geometric index spacing, DCT top-k by magnitude, MaCP-style banded selection, WaveletFT-style random Haar locations.
  - The same three after alignment (permutation by seriation; shared rotation from the pooled Gram).
  - Top-k magnitude in the weight domain.
  - Haar and Hadamard bases.
  - A Gaussian matrix of matched shape and variance.
- **Error:** plain Frobenius and Gram-weighted `‖(W − Ŵ) Σ^(1/2)‖_F`.
- **Bits:** 16 per stored value plus `ceil(log2(m·n))` per stored index, zero for block or prefix selections.
- **Output:** one JSON per layer and summary plots by layer type (q, k, v, o, FFN up, FFN down) and depth. These plots answer RQ1 and the reconstruction half of RQ3 from `DCT_papers.md` §6.
- **Go/no-go:** continue only if some aligned transform beats both the Gaussian null and DataSVD at equal bits on at least one layer type.

Run on GPT-2 locally on CPU for a smoke test, then on DAS-6 for Llama-3.2-1B.

## Phase 1: core layer

**New file `core/layers/spectral.py`.**
- `dct_matrix(n, device, dtype)`: orthonormal DCT-II, cached per size. Start with the explicit matrix; a fast FFT-based version is an optimization for Phase 5.
- `seriate(gram)`: order from the Fiedler vector of the correlation matrix. Returns a permutation.
- `shared_rotation(grams)`: eigenvectors of the pooled Gram, sorted by eigenvalue.
- `build_chain(errors_grid, params_grid)`: greedy path through the `(r, k)` grid, at each step taking the move with the smallest error increase per parameter saved. Returns the list of triples.

**New class `SpectralODLinear(ODLinear)` in `core/layers/spectral_linear.py`.**
- Stores `weight_u` as `B` (`r_max × n`, transform coordinates) and `weight_v` as `A` (`m × r_max`), exactly like `ODLinear`.
- Buffers: `t_in`, `t_out` (the composed `C·P`, non-persistent if rebuildable from a stored permutation), `_band_mask_in`, `_band_mask_out`, and `_chain`.
- Overrides:
  - `_on_inner_dim_changed`: set the rank mask and both band masks from `_chain[inner_dim]`.
  - `_forward`: transform, mask, call the existing dense kernel, mask, inverse transform.
  - `get_num_parameters(inner_dim)`: `r(k_i + k_o − r)` when `r ≤ k_o`, plus bias. This matches the GAR-style count used by `ODLinear.get_num_parameters`.
  - `get_weight(inner_dim)`: dense weight in the neuron basis.
  - `prune_weights`: slice `A` and `B` to the active triple.
  - `export_flexrank_state` / `set_extra_state`: add the chain and the permutation.
- `max_inner_dim` becomes `len(chain) − 1`. Check every caller that treats `max_inner_dim` as a rank: `_sample_inner_dims` and `_p_to_inner_dim` in `profiling.py`, `get_submodel_profiles` in `flextrain/src/flextrain/utils/flexrank_utils.py`.
- `eigs`: set to the square root of the per-chain-step energy increments, so `_tail_energy_by_rank` and the fast DP path work as they are.

**Stage-1 shortcut.** Start with masks on full-size factors (the `ODImpl.DENSE` pattern). This gives correct accuracy numbers with no speed benefit, which is enough for S1 and S2. Sliced and absorbed forwards come in Phase 5.

## Phase 2: decomposition path

- `core/types/decomposition.py`: add `DCT` and `DATA_DCT` to `SVDType`. Renaming the enum is out of scope.
- `core/layers/decomposition.py`:
  - New `decompose_linear_spectral(layer, inputs, device, dtype, align)`: transform `W`, call `compute_decomp` on the transformed (and, for `DATA_DCT`, Gram-weighted) matrix, build the error grid, call `build_chain`, return a `SpectralODLinear`.
  - `decompose(...)` gains a `svd_type` argument and dispatches. `Conv2d` keeps the SVD path.
- `core/trainers/svd_trainer.py`:
  - `_collect_layer_inputs` currently skips Gram collection only for `SVDType.SVD`; `DATA_DCT` and any aligned mode need the Grams.
  - `_decompose_layers` passes `args.svd_type` to `decompose`.
  - Alignment must be consistent across layers that share an axis. Add a pre-pass that computes one rotation for the residual stream and one permutation per FFN hidden axis, then hands them to each layer's decomposition.
- Exclusions: keep `exclude_layers_names: ['lm_head']` from `config/decomposition/default.yaml`.

**Scope of alignment in the first version.** Per-layer transforms only, kept as explicit buffers in each layer (correct but not free at run time). Absorbing a global residual rotation into embeddings, norms and all projections is a model-level rewrite; it is deferred to Phase 5 and done first for Llama (RMSNorm).

## Phase 3: profiles and sampling

- No change to `dp.py`. Each layer supplies `(params_savings, errors)` along its chain through the existing `_profile_layer` / `_fast_profile_layer`.
- Verify that `_filter_nested_solutions` still means nesting: a larger chain index must contain a smaller one on all three of `r`, `k_i`, `k_o`. `build_chain` guarantees this by construction; add a test.
- Samplers in `core/samplers/` operate on profile vectors of `inner_dim` and need no change.

## Phase 4: training-side wiring and tests

- `flextrain/src/flextrain/config/decomposition/dct.yaml` and `datadct.yaml`, following `datasvd.yaml`.
- `flextrain/src/flextrain/utils/args.py`: `DecompositionArguments` gains `align: none | permute | rotate`.
- `flextrain/src/flextrain/model/flexrank_model.py`: reconstruct `SpectralODLinear` from the `"class"` field of `export_flexrank_state`; add a round-trip test next to `flextrain/tests/test_flexrank_model_save_load.py`.
- Core tests in `flexrank/tests/test_spectral.py`, following `test_layers_core.py`:
  1. `dct_matrix` is orthonormal.
  2. Full rank and full band reproduce the dense layer output (tolerance 1e-4 in float32; run with TF32 off, see INSIGHTS §2.4).
  3. With full band, outputs equal `ODLinear` built from the same SVD at every rank.
  4. Analytic `get_num_parameters` equals the count of unmasked entries.
  5. Chain triples are monotone.
  6. Permuting hidden units and inverting the permutation leaves the model output unchanged.
  7. Save and load round trip.

## Phase 5: speed and deployment (only after S2 is positive)

- Sliced forward: store and multiply only `B[:r, :k_i]` and `A[:k_o, :r]`.
- Fast DCT through `torch.fft` for the non-absorbed transforms.
- Global residual rotation absorbed into the weights for Llama.
- Single small dense GEMM when `r = min(k_i, k_o)`.
- Kernel benchmark following E3: single token with CUDA graphs, and prefill.

## Idea B (fallback or add-on): mode-specific spectral patches

- New module `core/layers/patches.py`: per layer and per level, a coefficient vector and a location index into the DCT of the active factor.
- Locations chosen once from the DCT of the trained factor, MaCP-style (energy plus stratified random), shared across levels.
- Training: freeze the shared factors of a trained FlexRank checkpoint; train only the patches, one level per step, with the same distillation loss.
- Switching: write `factor + iDCT(patch)` into a scratch buffer; never modify the shared factors in place.
- Test: with zero patches the model equals the checkpoint at every level.

## Experiments on DAS-6

Follow the existing job rules: every batch script sources `slurm/common.sh`; GPU work is queued, never run on CPU nodes; `#SBATCH --exclusive`; time limit is twice the pessimistic runtime, at most 150 h.

| Job | Content | Depends on |
|---|---|---|
| `s0_diag.sbatch` | Phase 0 on GPT-2 and Llama-3.2-1B | Phase 0 |
| `s1_fronts.sbatch` | Untrained fronts: SVD, DataSVD, DCT, DataDCT with each alignment | Phases 1–3 |
| `s2_train.sbatch` | Joint nested training, two seeds, best S1 variant vs FlexRank | Phase 4 |
| `s3_patches.sbatch` | Idea B on a trained FlexRank checkpoint | Idea B module |
| `s4_bench.sbatch` | Kernel benchmark | Phase 5 |

Record each run in [EXP-JOURNAL.md](../EXP-JOURNAL.md) and results in [INSIGHTS-AND-ANALYSIS.md](../INSIGHTS-AND-ANALYSIS.md).

## Open points to settle while coding

- The body of `_sample_inner_dims` was not read in full; confirm it only needs `max_inner_dim` and `get_num_parameters`.
- Whether GPT-2's LayerNorm allows a clean residual rotation without the SliceGPT-style conversion.
- Whether attention `q`/`k` need a RoPE-compatible alignment on the head dimension (output side) for Llama; the first version leaves `k_o` full for `q` and `k`.
- Order of work: Phase 0 first. Do not start Phase 1 before the go/no-go.
