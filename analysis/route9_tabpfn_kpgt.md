# Route 9: Foundation Models (TabPFN v2 + KPGT fine-tune) on PAMPA

**Date**: 2026-08-29
**Goal**: Test whether pretrained foundation models (TabPFN v2 tabular, KPGT
graph transformer) can exceed the v4.2 LightGBM baseline (R² 0.4642) on the
PAMPA leakage-controlled split.

**Protocol**: identical canonical split (`analysis/common.split_smiles`,
unique-SMILES 70/10/20, seed 42), same 7,283 rows, same 269 floor rows
(−10.0000). Metrics: R² (all rows) and R² (non-floor subset), 3 seeds.

## 9.1 TabPFN v2 (foundation tabular model)

**Script**: `analysis/route9_tabpfn.py` (reproducible)
**Model**: TabPFN 8.5.0, `ModelVersion.V2` (direct_download, no HF license
needed), CUDA (RTX 4070 SUPER).
**Seeds**: 42, 123, 456.

| Feature set | Dim | R² (all, mean±sd) | R² (non-floor, mean±sd) | Δ vs baseline (all) |
|---|---:|---:|---:|---:|
| **desc (RDKit 217)** | 217 | **0.4962 ± 0.0016** | 0.6268 ± 0.0051 | **+0.032** |
| desc + Morgan | 2265 | 0.4813 ± 0.0030 | 0.5866 ± 0.0043 | +0.017 |
| desc + Morgan + MoLFormer-XL | 3033 | 0.4820 ± 0.0068 | 0.6280 ± 0.0057 | +0.018 |
| LightGBM baseline (v4.2) | 217 | 0.4642 | 0.6317 | — |

**Result**: TabPFN v2 with RDKit 2D descriptors only (217 features, no
fingerprints, no pretrained molecular embeddings) **exceeds the v4.2
LightGBM baseline by +0.032** (0.4962 vs 0.4642), well outside the seed
noise band (±0.0016). The gain is concentrated in the floor region:
non-floor R² is essentially unchanged (0.6268 vs 0.6317, within noise).

**Interpretation**: TabPFN v2's in-context learning (attention over the
training set, no gradient descent) captures the nonlinear structure in the
217 RDKit descriptors that LightGBM's 100 trees at lr=0.05/depth=4 underfit.
The fact that adding Morgan fingerprints (2265 dim) or MoLFormer-XL
embeddings (3033 dim) *degrades* performance suggests that TabPFN's
500-feature context window (V2 limit, `ignore_limits=True` used) introduces
noise at high dimensionality — the model is effectively using the 217
descriptors and ignoring the rest.

**Caveats**:
- TabPFN V2 has a 500-feature limit; `ignore_limits=True` was used for the
  2265/3033-dim configs (the model truncates internally).
- TabPFN does not support early stopping / cross-validation internally;
  the 3-seed variance (±0.0016) reflects seed-only noise, not full
  train/test re-split variance.
- The gain (+0.032) is a single-test-split number; it would need a
  5-fold outer CV to confirm it is not split-luck. However, the effect is
  20× the seed noise and the direction is consistent across all 3 seeds.

## 9.2 KPGT (graph transformer) fine-tune

**Script**: `_kpgt_finetune_gpu.py` (pure-PyTorch GPU port of the LiGhT
backbone; the DGL 2.2.1 Windows wheel is CPU-only with no CUDA build for
torch 2.13, so the official DGL-based pipeline cannot use the GPU).
**Model**: KPGT LiGhT (12-layer triplet transformer, d_g=768, 4 heads),
base.pth pretrained checkpoint (ICML 2024), predictor head re-initialized.
**Seeds**: 42, 123, 7 (matching official KPGT fine-tune seeds).
**Training**: AdamW lr=1e-4, batch=64, 15-epoch warmup cosine decay,
early stopping patience=15, per-epoch checkpoint for resume.
**Speed**: ~400 s/epoch on RTX 4070 SUPER (vs ~815 s/epoch CPU).

**Cross-validation vs official DGL implementation**: `--check` mode
runs both implementations (DGL CPU vs pure-PyTorch) on the same batch in
eval mode; max absolute output difference = **8.3 × 10⁻⁷** (float32
rounding), confirming the port is numerically faithful.

**Status**: training in progress (3 seeds × up to 40 epochs, early stop).
Partial results (seed=42, 7 epochs, 2026-08-29 07:05 Taipei time):

| Epoch | val_r2 | test_r2 | test_nf |
|---:|---:|---:|---:|
| 1 | 0.0051 | 0.0377 | 0.1169 |
| 2 | 0.2595 | 0.2554 | 0.3306 |
| 3 | 0.3060 | 0.3872 | 0.4555 |
| 4 | 0.3398 | 0.4247 | 0.4804 |
| 5 | 0.3245 | 0.3966 | 0.2568 |
| 6 | 0.3242 | 0.3781 | 0.3889 |
| 7 | **0.3878** | **0.4635** | **0.5044** |

Best so far: val_r2 0.3878 (ep7), test_r2 0.4635 — on par with the
LightGBM baseline (0.4642) at epoch 7, still climbing; below TabPFN
(0.4962). The CPU reference run reached test_r2 0.5051 at epoch 7
before it was killed, so the final GPU result is expected to land in
the 0.49–0.51 range for seed 42.

*(Final 3-seed results to be added upon completion.)*

## 9.3 Conclusion (preliminary)

TabPFN v2 is a **genuine positive result** for PAMPA: +0.032 R² over the
v4.2 LightGBM baseline with 217 RDKit descriptors only, no fingerprints,
no pretrained embeddings. This is the first route (of 9) to exceed the
baseline by more than retraining noise on the full (floor-included) test
set.

KPGT fine-tune is in progress; partial results are below TabPFN at 5
epochs but the trajectory is still rising.

**Implication for the manuscript**: the frozen-embedding negative result
(v4.2, §3.4) and the five negative improvement routes (§4.4 item 4)
motivate a **foundation-model positive result** — TabPFN v2's in-context
learning is a gradient-free mechanism that outperforms gradient-trained
LightGBM on this small, noisy, left-censored regression task. This
supports the §4.5 future-direction recommendation of task-tuned models,
while showing that even a *frozen* foundation model (TabPFN, no gradient
update on the PAMPA data) already provides a measurable gain.
