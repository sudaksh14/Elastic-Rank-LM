# Brainstorm: beyond the DCT for elastic weight factorization in LLMs

Date: 2026-11-13. Ideas that came out of the frequency-domain reading but do not depend on the DCT. The recommended paper idea is in [paper_plan.md](paper_plan.md). Earlier design notes are in [elastoformer_flexrank_brainstorm.md](../elastoformer_flexrank_brainstorm.md).

**Reading note.** Papers marked (memory) are cited from recall and were not opened in this session. Verify them before citing. LYRA, WaveletFT, Freq-LoRA and the tensor survey have now been read from `related papers/spectral/`. FDA, LFMA and F-Adapter are still abstract-only.

## 1. The lesson from the DCT literature

A nested weight family needs three things:

1. **An ordered axis**, so a submodel is a prefix and needs no index storage.
2. **A cheap basis**, so the budget pays for coefficients, not for the basis.
3. **Dense kernels**, so parameter savings become latency savings.

SVD gives 1 and 3 and pays for the basis. Fixed transforms give 2 and 3 but only help if the weights are compressible in that basis. Each idea below is a different way to get all three.

**Where a real order exists in a transformer.** Neuron indices have none. These axes do:
- Depth (layer index).
- Token position.
- Vocabulary, when sorted by token frequency.
- Singular-value rank, after an SVD.
- Any axis whose order is created by training (ordered dropout).

## 2. Shortlist

| # | Idea | Addresses | Cheapest test | Risk |
|---|---|---|---|---|
| 1 | Train the order in: ordered dropout on a transform axis plus a sparsity penalty | Compaction that does not exist at init | Add an L1 term on transform coefficients to an E6-style run; plot compaction before and after | Medium |
| 2 | Rate–distortion allocation: budget in bits, not parameters | Rank, bandwidth and precision in one DP | Offline: per-coefficient bit allocation vs uniform fp16 at equal bits | Medium |
| 3 | Learned fast transform (butterfly) initialized at the DCT | Permutation-only alignment being too weak | Phase 0 curves with a butterfly fitted to the Gram | Medium-high |
| 4 | Residual-width nesting in a rotated basis | Smallest-budget collapse | Untrained front on Llama-1B with a global PCA rotation and prefix slicing | Low-medium |
| 5 | Cross-layer shared bases along depth | Factor memory | SVD of stacked same-type weights across layers; share of energy in the shared basis | Medium |
| 6 | Frequency-sorted vocabulary axis for embeddings and `lm_head` | The large untouched share of ≤1B models | Rank needed per frequency bucket for fixed logit error | Low-medium |
| 7 | Haar wavelets on seriated neurons | Localized structure; width halving as a scale axis | Phase 0 curves, Haar vs DCT | Medium |
| 8 | Per-token rank or bandwidth routing | Input-dependent compute | Oracle: best fixed profile vs per-token oracle at equal mean FLOPs | High |

## 3. Notes on each idea

**1. Train the order in.**
- FlexRank does not find an ordered rank axis; it trains one, by sampling prefixes. The same can be done for any axis.
- FreqPrune adds an L1 penalty on DCT coefficients and retrains briefly to make them sparse before compressing (its Eq. 4).
- Combine the two: during consolidation, sample a bandwidth prefix per step and penalize high-index coefficients. Compaction then becomes a training outcome, and the kill test in the paper plan stops being fatal.
- Related: nested dropout and FjORD's ordered dropout (memory).

**2. Rate–distortion allocation.**
- Transform coding in JPEG gives each coefficient a precision matched to its variance. The classical result is reverse water-filling: bits go where variance exceeds a threshold, and zero bits elsewhere (truncation is the zero-bit case).
- Applied here: per singular direction or per coefficient, choose a bit width. Rank truncation, bandwidth truncation and quantization become one decision.
- The existing DP already handles a list of (saving, error) options per layer; savings would be counted in bits.
- Open issue from the first brainstorm: factor conditioning after truncation may make INT8 difficult. This framing tests it directly.
- Related: Matryoshka Quantization (memory).

**3. Learned fast transforms.**
- A butterfly factorization has `O(n log n)` parameters and compute, and can represent the DCT and FFT exactly (Dao et al., 2019; memory).
- Initialize at the DCT, then fit to the layer's Gram so the transform approaches the data-optimal one while staying fast.
- This sits between a fixed DCT (free, weak) and a dense rotation (strong, `n²`).
- FourierFT's ablation supports the premise that the choice of basis matters: Fourier beat a random orthogonal basis.
- Related: Monarch matrices (memory).

**4. Residual-width nesting in a rotated basis.**
- The rank axis cannot shrink the residual stream, which is why the 3.6 M level fails on the rank axis (INSIGHTS §2.7).
- For RMSNorm models a single orthogonal rotation of the residual stream is free. Choose it by PCA of pooled activations; a prefix of the rotated coordinates is a narrower model with dense kernels.
- SliceGPT (memory) does this for one budget, with per-block rotations that add shortcut matrices. The new part is nesting, joint training, and combination with per-layer rank.
- This is the KLT end of the transform ladder in the paper plan, and the most likely to work.

**5. Cross-layer shared bases.**
- Depth is a real ordered axis. If the input factors of the same projection type span similar subspaces in neighbouring layers, one basis can serve a group of layers, with small per-layer cores (a Tucker-2 structure).
- The basis cost is then paid once per group, not once per layer.
- A DCT along depth of the stacked per-layer cores is a legitimate use of the DCT, since the axis is ordered.
- Related: Basis Sharing for LLM compression (memory).

**6. Vocabulary axis.**
- `lm_head` is excluded from decomposition by default and is a large share of small models.
- Sorted by token frequency, the vocabulary has a meaningful order. Frequent tokens can keep high rank and rare tokens low rank, with nesting along the frequency order.
- Related: adaptive input and adaptive softmax (memory).

**7. Haar wavelets on seriated neurons.**
- After seriation, neighbouring neurons are similar. A Haar approximation coefficient is then the average of a group of similar neurons, so dropping detail levels amounts to merging neurons.
- Each scale halves the width, which gives a coarse but strictly nested ladder with dense kernels.
- Wavelets also keep localized structure that a global DCT smears.
- Caution from WaveletFT: it trains coefficients at random Haar locations and gets its best result with three high-frequency coefficients per low-frequency one. So it is no evidence that updates are concentrated at coarse scales. Db2 and Sym2 were slightly better than Haar there and are worth including in the diagnostic.

**8. Per-token routing.**
- FourierMoE routes tokens to experts that own different frequency bands, with top-k gating.
- Freq-LoRA shows how small such a controller can be: a 140-parameter gate, driven by DCT statistics of the input, weights four frequency bands and comes within 0.006 mIoU of a variant given ground-truth weather metadata.
- The elastic analogue: a small router picks the chain index (rank and bandwidth) per token or per query.
- This only pays off if switching is free, which holds for prefix-nested layers.
- It is the input-dependent compute direction already listed in the first brainstorm (§2.6).

## 4. Smaller ideas

- **Low rank plus sparse, nested.** SVD prefix plus an importance-ordered sparse residual. Idea C in the paper plan; the control against plain magnitude sparsity decides whether a transform is worth it. Related: OATS, LoSparse (memory).
- **Mode-specific patches beyond the DCT.** Idea B with LoRA or diagonal scaling per level in place of spectral coefficients. Tests whether cross-budget interference is low-dimensional.
- **Nested tensor formats.** The tensor survey lists Tucker, tensor-train and tensor-ring compression of attention and FFN weights, with ranks set by hand or by search. None is nested. A tensor-train core has an ordered rank per bond, so prefix nesting and FlexRank's joint training carry over; the survey's own future-work list asks for adaptive rank selection and hardware-aware formats.
- **Smoothness as a regularizer, not a compressor.** LYRA finds its low-frequency block helps most on the smallest tasks and attributes this to regularization. A low-frequency constraint on the per-level patches of idea B could stabilize the small-budget levels in the same way.
- **Feature-domain compression.** FDA (abstract only) and Freq-LoRA act on activations, not weights. The same idea applied to the KV cache: store cached keys and values in the leading coordinates of the rotated basis at low modes.
- **Curvature-weighted selection.** Replace the Gram weighting with a Fisher or Hessian diagonal estimate, so selection reflects loss sensitivity and not only output error.
- **Head-granular structure.** Keep rank and bandwidth choices constant within an attention head so fused QKV kernels survive; align on RoPE pairs.
- **Theory.** FlexRank's nested-training result relies on Eckart–Young: the SVD prefix is the optimal rank-`r` approximation. Ask for which ordered dictionaries the same argument goes through. The KLT under the Gram norm is a candidate; a fixed DCT is not, unless alignment makes it close to the KLT. A sufficient condition would turn the transform ladder into a theorem.

## 5. Suggested order

1. Run the Phase 0 diagnostics from [implementation_plan.md](implementation_plan.md). They cost a day and inform ideas 3, 4, 5 and 7 at once.
2. Idea 4 next, since it attacks the failure this project has already measured twice.
3. Idea 1 if Phase 0 shows no compaction at initialization.
4. Ideas 2 and 6 as independent extensions that do not depend on the outcome of the above.
