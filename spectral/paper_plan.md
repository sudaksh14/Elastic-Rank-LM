# Paper plan: a spectral replacement for FlexRank's SVD factorization

Date: 2026-11-13. Companion files: [implementation_plan.md](implementation_plan.md), [brainstorm.md](brainstorm.md). Project state is in [INSIGHTS-AND-ANALYSIS.md](../INSIGHTS-AND-ANALYSIS.md).

## 1. Starting point

**What FlexRank does.** Each linear layer becomes `W ≈ V·U` with `U` of size `r × n` and `V` of size `m × r`, initialized by SVD or DataSVD ([decomposition.py](../FlexRank/flexrank/src/flexrank/layers/decomposition.py)). A submodel is a prefix of the rank axis. A dynamic program picks per-layer ranks, and all nested submodels are trained jointly by distillation. FlexRank has no DCT today; "replacing the DCT-based factorization" is read here as "replacing the SVD-based factorization with a spectral one".

**Three weaknesses already measured in this project.**

| Weakness | Evidence |
|---|---|
| Factors cost `r(m + n)`: 2× the dense matrix at full rank | Brainstorm §1.3; cap experiments in E6 |
| The smallest budget collapses because the residual width stays fixed | INSIGHTS §2.7: over budget at 3.6 M even at minimum rank |
| Two GEMMs per layer; no speed advantage over parameter-matched dense pruning | INSIGHTS §2.5 (E3, A40) |

**What the frequency-domain literature offers.** A weight (update) can be stored as a few coefficients in a fixed transform. The basis then costs nothing to store and can be applied in `O(n log n)`.

## 2. Literature: what was read and what it supports

| Paper | Read | Finding used here |
|---|---|---|
| MaCP (ACL 2025; same work as sDCTFT, arXiv 2410.09103) | Full text | Takes the DCT of the pretrained `W`, splits the spectrum into low/mid/high by distance from the origin, picks `n` locations per layer (70% by energy, the rest random, stratified), trains `ΔW` only there. `n` = 700 for a 768² layer. "Low only" is worse than the three-band mix (Table 8) |
| FourierFT (ICML 2024) | Full HTML | `n` random locations shared by all layers. A Fourier basis beats a random orthogonal basis for training (RTE 79.1 vs 75.6). A fixed frequency bias does not help reliably |
| LoCA (ICLR 2025) | Main text | Theorem 1, for an i.i.d. Gaussian matrix at equal parameter count: random locations are worse than low rank; top-magnitude locations are better than low rank. Locations are learned by finite differences, alternating with coefficients |
| FreqPrune, Zeng et al. (ACML 2021) | Method and results | The only paper here that compresses the pretrained weights. DCT on each `d × d` conv kernel, L1 penalty in the frequency domain, then data-aware low rank on the frequency matrices, repeated. The DCT runs along the spatial kernel axis, which has a real order |
| FourierMoE (arXiv 2604.01762) | Main text | In RoBERTa-large, early layers show high-frequency spikes and deeper layers are smoother. Band-specific experts beat random locations |
| LFMA (PMLR v282) | Abstract | Top-k Fourier components by magnitude, per layer |
| FDA (Findings of EMNLP 2025) | Abstract | Modulates features, not weights, in the Fourier domain |
| F-Adapter (NeurIPS 2025) | Abstract | More adapter capacity for low frequencies; theory for Fourier neural operators |
| LYRA, Muhsin and Ko (IEEE SPL 2026) | Full text | `ΔW = γ · C_m[:p]^T · S · C_n[:q]`, with a `p × q` low-frequency block and a three-matmul forward that never forms the dense update. Optional `S = A·B` with rank `r`, cost `r(p + q)`. **Its own spectral analysis finds no low-frequency concentration in full fine-tuning updates (about 0.04% of energy, the same as random selection).** The authors call the block a smoothness regularizer, not a recovery of a concentrated optimum. Encoders only (BERT, RoBERTa) |
| WaveletFT, Hu et al. (Neurocomputing 2025) | Full text | One-level Haar transform of `ΔW`; `4n` trainable coefficients per layer at **random** locations shared by all layers, one part low-frequency to three parts high-frequency. Db2 and Sym2 are slightly better than Haar. The update is reconstructed densely |
| Freq-LoRA, Xia et al. (Remote Sensing 2026) | Method and main results | Acts on 64 × 64 encoder **feature maps**, not weights. Four learned Gaussian frequency bands and a 140-parameter gate driven by the input image's DCT statistics. The DCT axis here is spatial, so it has a real order |
| Tensor-factorization survey, He et al. (Neurocomputing 2026) | Sections 1–3, 6–7 | Taxonomy: truncated SVD, CP, Tucker, tensor train, tensor ring. Lists DRONE, LASER and matrix-product operators for transformers. Benchmarks are ResNet-20 only. Future directions are hardware co-design and adaptive rank selection |

**Citation hygiene.** [DCT_papers.md](../related%20papers/spectral/DCT_papers.md) is a chatbot-generated list, but all four papers that could not be retrieved earlier exist and match its descriptions. Still abstract-only: LFMA, FDA, F-Adapter. Found by search and not read: WaveFT (arXiv 2505.12532), DWTSG (AAAI 2026), SpectralLoRA (arXiv 2604.10649). These six must be read in full before they are cited in a paper.

## 3. Three observations that decide the design

1. **The gap is real.** Every frequency-domain PEFT paper compresses a trained update `ΔW`, at a single budget, with the dense pretrained `W` kept. None builds nested submodels, none compresses `W` itself in a transformer, and none selects coefficients by output error on calibration data.
2. **A raw DCT of a transformer weight has no reason to compact energy.** The DCT compacts signals that are smooth along their index. Neuron indices are arbitrary: permuting hidden units changes nothing in the function but scrambles the spectrum. FreqPrune works because conv kernels have a spatial order. For transformer weights the order has to be created. LYRA's measurement supports this directly: even fine-tuning updates carry about 0.04% of their energy in the low-frequency block.
3. **Reported compaction is close to the noise floor.** For pure Gaussian noise, the largest 33% of coefficients already hold about 81% of the energy, and 90% needs about 45% (my calculation). The search snippet for SpectralLoRA reports 90% of energy in about 33% of DCT coefficients for LoRA updates. That is better than noise, but not by much. A Gaussian matrix is also rotation-invariant, so LoCA's "top-magnitude beats low rank" result holds equally for top-magnitude entries in the ordinary weight domain (my argument, not a claim from the paper). Any DCT claim therefore needs two controls: the Gaussian null, and plain magnitude sparsity at equal bits including index storage.

## 4. Recommended idea (A): two-axis elastic factorization, rank × bandwidth

**Representation.** Let `C_n` be the orthonormal DCT-II matrix of size `n`. For each layer:

```
W(r, k_i, k_o) = C_m^T[:, :k_o] · A[:k_o, :r] · B[:r, :k_i] · C_n[:k_i, :]
```

- `r` is the rank, as in FlexRank. `k_i` and `k_o` are the input and output **bandwidths**: how many leading transform coordinates the layer reads and writes.
- Stored parameters: `r(k_i + k_o)`, against `r(n + m)` for FlexRank. `C` is never stored.
- `k_i = n, k_o = m` is exactly FlexRank in a rotated basis, so the method can only match or improve FlexRank's starting point.
- `r = min(k_i, k_o)` is LYRA's dense block; `r < min(k_i, k_o)` is LYRA's factored mode.
- A submodel is a prefix on both axes, so nesting needs no index bookkeeping.

**Relation to LYRA (prior art for the algebra).** LYRA already uses this exact form, including the rank-`r` core, for a trainable update `ΔW` on encoders at one fixed budget. The representation alone is therefore not a contribution. What is new here: applying it to the pretrained weight itself, creating the order by alignment, selecting by output error, nesting on both axes, and per-layer allocation. LYRA lists "decoder-only and larger models" and "adapting the frequency band during training" as future work; both are covered by this plan.

**Contribution 1: gauge alignment.** A "gauge" is a change of basis inside the network that leaves its function unchanged. Two are available:
- FFN hidden units can be permuted freely. Order them by spectral seriation of their activation correlation, so that neighbours are similar and the DCT sees a smooth signal.
- The residual stream of an RMSNorm model (Llama) can be rotated by one global orthogonal matrix. With this, the transform on the residual side is absorbed into the weights and costs nothing at run time. GPT-2 uses LayerNorm and needs the mean subtraction folded in first, as SliceGPT does (from memory; verify).

FlexRank already collects the Gram matrices needed for both ([gram.py](../FlexRank/flexrank/src/flexrank/layers/gram.py), `collect_model_grams`).

**Contribution 2: DataDCT.** Rank coefficients and bandwidths by output error `‖(W − Ŵ) Σ^(1/2)‖`, where `Σ` is the input Gram matrix. This is the DataSVD criterion applied to spectral selection. MaCP, LFMA and FourierFT select by weight energy, magnitude, or at random.

**Contribution 3: nested training on two ordered axes.** Reuse FlexRank's sampler and distillation unchanged. Each layer's (rank, bandwidth) grid is reduced to one monotone chain, so a submodel is still one integer per layer.

**Contribution 4: allocation.** The existing DP chooses a chain position per layer. FourierMoE's observation that spectral content varies with depth predicts a non-uniform bandwidth profile, which the DP can exploit.

**Why this targets the measured weaknesses.**
- Memory: basis storage shrinks from `r(m + n)` to `r(k_i + k_o)`.
- Small-budget floor: on the residual side, bandwidth is a residual-width reduction in a rotated basis. This is the missing degree of freedom from INSIGHTS §2.7.
- Latency: where the transform is absorbed, the layer is a smaller two-GEMM; where `r` reaches `min(k)`, it is a single small dense GEMM.

**An honest fork.** Where the rotation can be absorbed (the residual stream), any orthogonal basis is free, and the data-optimal basis (PCA of the activations, the KLT) is at least as good as a DCT. The DCT is needed only where absorption is impossible: FFN hidden units and per-layer bases. So the experiments compare a ladder of transforms: raw DCT, DCT after seriation, and the shared KLT as the upper bound. If only the KLT helps, the paper becomes "rank × width in a rotated basis" and drops the DCT claim. If nothing helps, idea B is the fallback.

## 5. Alternates

**Idea B: mode-specific spectral patches (low risk).**
- Keep FlexRank's shared factors. For each budget level `j`, add a small DCT-domain delta to the active factors, of MaCP size (hundreds to a few thousand coefficients per layer).
- Motivation: joint training leaves the top level 0.5–1.2 pp below the frozen-prefix baseline (INSIGHTS §2.8). FlexRank's theory attributes such gaps to interference between submodels.
- Scientific question: is cross-budget interference low-dimensional enough for a few spectral coefficients to remove it?
- Cost: negligible storage. Switching needs one sparse inverse transform per layer. Submodels stop being exact prefixes of each other, which weakens the cache-reuse argument.
- This uses the PEFT methods as designed, so it is the most likely to work.

**Idea C: low-rank head plus spectral-sparse bulk.**
- `W ≈ V_r U_r + C^T S C`, with `S` sparse and its coefficients ordered by importance so a prefix defines a submodel.
- Motivation: the SVD residual is the unstructured part of the weight, the case where LoCA's theorem favours coefficient selection over more rank.
- Required control: magnitude sparsity in the weight domain at equal bits. By observation 3 this may tie, in which case the DCT adds nothing.

## 6. Hypotheses and gates

| ID | Hypothesis | Test | Gate |
|---|---|---|---|
| H0 | After alignment, a transformer weight compacts better in the transform basis than the Gaussian null | S0: offline curves, no training | If no layer type beats the null, stop idea A's DCT variant |
| H1 | At equal stored bits, rank × bandwidth has lower Gram-weighted error than rank alone | S0 | Needed to continue idea A |
| H2 | The untrained front (perplexity vs parameters) improves over DataSVD | S1 | Needed before any training |
| H3 | After joint training, the trained front improves, most at small budgets | S2 | Paper claim |
| H4 | Per-level spectral patches close the top-level and low-level gaps | S3 | Idea B claim |
| H5 | Bandwidth allocation is non-uniform across depth and layer type | S2 analysis | Supporting analysis |

**Mapping to the three research questions in `DCT_papers.md` §6.** All three are covered; the plan adds controls they lack.

| Question in `DCT_papers.md` | Where it is in this plan | What this plan adds or changes |
|---|---|---|
| RQ1, spectral compressibility: are pretrained LLM weights concentrated in low-frequency DCT components, and does it differ between attention and MLP layers? | H0, experiment S0, reported per layer type and depth | The Gaussian null, and the same curves after alignment. Expected answer for the raw DCT is "no" (observation 2 and LYRA's measurement) |
| RQ2, nested spectral representations: can coefficients be ordered so smaller models are strict subsets of larger ones? | Contribution 3, H3, experiment S2 | The bandwidth prefix gives the nesting by construction; joint training with FlexRank's sampler is the test |
| RQ3, spectral versus rank at equal parameter budget, on reconstruction error and perplexity | H1 (reconstruction, S0) and H2 (perplexity, S1) | Budgets counted in bits including index storage; error weighted by the input Gram matrix |

The comparison list in that section (SVD truncation, DCT low-frequency truncation, DCT top-k, DCT followed by low rank, wavelet truncation) is the S0 baseline set. One correction: "DCT, then SVD of the full spectrum" is identical to plain SVD, because the DCT is orthogonal and leaves singular values unchanged. It only differs once frequencies are truncated, which is idea A. Its approach C (different budgets for Q, K, V and MLP) is the DP allocation, tested as H5.

## 7. Experiments

**Models.** GPT-2 first (fast, FlexRank config exists), then Llama-3.2-1B (RMSNorm, the thesis target). DeiT-B optionally, for continuity with E2 and E6.

**Baselines, all at equal stored bits (index bits counted).**
- SVD and DataSVD truncation; FlexRank trained.
- Raw DCT low-frequency block (LYRA contiguous); LYRA's geometric index spacing; raw DCT top-k by magnitude; MaCP-style banded selection; WaveletFT-style random Haar locations.
- Top-k magnitude sparsity in the weight domain.
- Gaussian matrix with matched variance (the null).
- Haar wavelet and Hadamard transforms as alternative fixed bases.

**Metrics.** Gram-weighted reconstruction error; perplexity; lm-eval commonsense (already wired in FlexRank); parameters and bits; peak memory; TTFT and TPOT at batch 1.

**Sequence.**
- S0, diagnostics: compaction curves per layer type and depth. About one day, no training.
- S1, untrained fronts on GPT-2 and Llama-1B.
- S2, joint nested training with the E6/E4a recipe, two seeds.
- S3, idea B on an existing trained FlexRank checkpoint.
- S4, kernel benchmark in the style of E3 for the band-limited layer.

## 8. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| No compaction after alignment | Idea A's DCT variant fails | S0 is cheap; fall back to the KLT variant or idea B |
| Seriation is too weak a gauge (permutation only) | DCT loses to KLT | Report the ladder; consider learned fast transforms ([brainstorm.md](brainstorm.md) §3) |
| Transform cost cancels the saving | No latency gain | Absorb on the residual side; fast DCT elsewhere; measure in S4 |
| Two-axis grid breaks the DP's additive-error assumption | Poor profiles | Chain reduction per layer; the assumption is already weak (INSIGHTS §2.4) |
| Overlap with LYRA (same algebraic form), SliceGPT and basis-sharing work | Novelty challenge | Cite LYRA as the source of the form; position on pretrained-weight compression, alignment, nesting and joint training, which those lack; do a full related-work search before writing |
| Single-seed noise | Unreliable deltas | Two seeds minimum; treat differences under 0.5 pp as noise |

## 9. What the paper claims if it works

1. Low rank and fixed-transform coding are two ends of one two-axis family, and nesting works on both axes.
2. Function-preserving re-ordering makes fixed transforms useful for pretrained transformer weights; without it they are not.
3. Data-aware spectral selection beats energy-based selection.
4. A better trained Pareto front than FlexRank at small budgets, with lower factor memory.

A negative result on point 2, with the Gaussian-null analysis, is still publishable as a short paper: it would show that the "energy compaction" motivation in this literature does not transfer from updates to pretrained weights.
