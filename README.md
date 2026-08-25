# Peptide ADMET Prediction Model

**Honest, reproducible PyTorch *mixed multi-task* model for peptide ADMET prediction — 9 endpoints (6 binary + 2 multiclass + 1 regression), an extensible training set, and an AMPBench-MT-style homology-controlled evaluation**

![Status](https://img.shields.io/badge/status-demo_pipeline-orange)
![Data](https://img.shields.io/badge/data-synthetic_demo_%2B_external-blue)
![Endpoints](https://img.shields.io/badge/endpoints-9--mixed-green)
![Eval](https://img.shields.io/badge/eval--split-homology--controlled-green)
![License](https://img.shields.io/badge/License-MIT-green)

> **2026-08 v3.0 (extensibility + endpoint expansion).** Builds on the
> v2.0 integrity revision. This release adds:
> 1. **pepADMET's four toxicity endpoints** — `toxicity_binary` (binary),
>    `toxicity_type` (6-class), `neurotoxicity_type` (4-class), and
>    `HC50` (regression) — so the model now predicts **9 endpoints** in a
>    single forward pass (6 binary + 2 multiclass + 1 regression).
> 2. **pepADMET's partial-label mechanism** — a `NaN` cell means "this
>    endpoint is not measured for this row"; the trainer *masks* it out of the
>    loss and the predictor reports `None`, exactly as pepADMET does.
> 3. **A scalable, extensible training set** — `prepare_data.py --n <N>`
>    generates any size, and `--merge <csv>` folds in an **external** dataset
>    (validated, deduped, provenance-stamped, partial labels preserved) via
>    `ingest_external.py`. Real measured data can therefore be added without
>    touching the pipeline.

---

## 📊 Overview

A single **PyTorch shared-trunk MLP (428 → 256 → 128)** with **per-task heads**
predicts **nine** peptide properties:

| # | Endpoint | Type | Meaning |
|---|----------|------|---------|
| 1 | `GI_absorption` | binary | Oral GI absorption |
| 2 | `Caco2_permeability` | binary | Caco-2 cell-line permeability |
| 3 | `BBB_penetration` | binary | Blood-brain barrier penetration |
| 4 | `Ames_mutagenicity` | binary | Ames mutagenicity risk |
| 5 | `hERG_inhibition` | binary | hERG channel (cardiotoxicity) risk |
| 6 | `toxicity_binary` | binary | Overall cytotoxicity |
| 7 | `toxicity_type` | multiclass (6) | Toxicity mechanism (0 = non-toxic) |
| 8 | `neurotoxicity_type` | multiclass (4) | Neurotoxicity subtype |
| 9 | `HC50` | regression | Half-maximal cytotoxicity (~log scale; lower = more potent) |

Endpoints 1–5 are the original ADME/safety set; **6–9 are pepADMET's toxicity
endpoints** (Tan et al., 中南大學; `ifyoungnet/pepADMET`), added in v3.0.

In addition, the predictor reports a **composite multi-objective score** —
the geometric mean of each composite endpoint's *favourability* in [0,1] —
following the multi-objective candidate ranking used by AMPGAN v3 / PepCraft
(arXiv:2606.17127).

### Key Features

- ✅ **Honest evaluation** — headline numbers come from `metrics.json`,
  produced by the training run; nothing is hardcoded in the predictor.
- ✅ **Homology-controlled split** — train/test sequences never share
  amino-acid-composition families (max pairwise Jaccard 0.25 < 0.5
  threshold), per the AMPBench-MT protocol (arXiv:2607.25518).
- ✅ **Reproducible data** — the 15,000-row demo dataset is regenerated
  from a fixed seed by `prepare_data.py` and is stamped
  `data_origin=synthetic_demo` in every row.
- ✅ **428-dimensional feature space** — AAC (20) + DPC (400) +
  physicochemical properties (8).
- ✅ **Fast inference** — single-MLP forward pass, suitable for
  high-throughput candidate ranking.

---

## 🎯 Performance (measured, this run)

Measured on the **homology-controlled test split** (5,992 sequences; 30,000-row
training set, 20,996 train / 3,012 val). The "primary metric" per endpoint is
AUC (binary), macro-F1 (multiclass), or R² (regression).

| Metric | Value |
|--------|-------|
| **Mean primary metric (headline)** | **0.7189** |
| Random-split mean metric (comparison) | 0.7227 |
| Homology-vs-random delta | +0.0038 |
| Model params | 145,681 |

### Per-Endpoint (homology-controlled test split, 30k run)

| Endpoint | Type | Primary metric | Other | n labelled |
|----------|------|---------------|-------|-----------|
| GI Absorption | binary | **AUC 0.8864** | MCC 0.4580, Acc 0.8141 | 5,992 |
| Caco-2 Permeability | binary | **AUC 0.9105** | MCC 0.6257, Acc 0.8258 | 5,992 |
| BBB Penetration | binary | **AUC 0.9368** | MCC 0.5433, Acc 0.8652 | 5,992 |
| Ames Mutagenicity | binary | **AUC 0.8415** | MCC 0.4420, Acc 0.7710 | 5,992 |
| hERG Inhibition | binary | **AUC 0.8723** | MCC 0.5194, Acc 0.7714 | 5,992 |
| toxicity_binary | binary | **AUC 0.8392** | MCC 0.4856, Acc 0.7537 | 5,992 |
| toxicity_type | multiclass (6) | **macro-F1 0.2208** | Acc 0.7034 | 5,992 |
| neurotoxicity_type | multiclass (4) | **macro-F1 0.3525** | Acc 0.3847 | 1,297 (partial) |
| HC50 | regression | **R² 0.6100** | RMSE 0.3386, MAE 0.2680 | 3,589 (partial) |

> ⚠️ **These numbers characterize the demo pipeline, not real-peptide
> performance.** The core labels are drawn from a latent physicochemical model,
> so the binary AUCs (~0.84–0.94) and the partial-label endpoints are exactly
> what the *pipeline* is designed to produce. The multiclass macro-F1s are low
> because the class distributions are imbalanced (class 0 dominates
> `toxicity_type`) — an honest consequence of the latent model, not a bug.

**Effect of training-set size (this release's other goal).** Regenerating at
15,000 rows (same seed, same protocol) gives a mean primary metric of **0.6929**
and HC50 R² **0.4610**; scaling to 30,000 rows lifts them to **0.7189** and
**0.6100** respectively. The improvement is small and expected for a
physicochemistry-driven demo, but it demonstrates that the training set is
genuinely *scalable* — and that real data folded in via `--merge` will move
these numbers in the same direction.

The random-split comparison (0.7227) exists to quantify the "memorization" gap
that AMPBench-MT (arXiv:2607.25518) documents. Here the gap is small (+0.0038)
because the demo labels are driven by physicochemistry, but the *protocol* is
what matters for real data.

---

## 🚀 Quick Start

### Installation

```bash
# Recommended: project venv (system Python may be externally managed)
uv venv .venv
uv pip install --python .venv/Scripts/python.exe torch --default-index https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe scikit-learn pandas numpy

# or:
pip install -r requirements.txt
```

### Run the full pipeline (reproduces every number above)

```bash
python prepare_data.py --n 30000       # 30,000 synthetic_demo rows, seed 42 (any --n)
python homology_split.py               # family-disjoint 70/10/20 split + leakage audit
python train_peptide_admet_model.py    # trains mixed MLP, writes metrics.json
```

### Add an external / real dataset (the "bigger training set" path)

```bash
# 1) ingest an external CSV: validate, dedup, stamp provenance, keep partial labels
python ingest_external.py --input real_data.csv --source ampdb_real --output data/real.csv
#    (for pepADMET-style SMILES files, pass --smiles to attempt sequence recovery)

# 2) fold it into the training set (external rows keep their own data_origin)
python prepare_data.py --n 30000 --merge data/real.csv --out data/train.csv

# 3) split + train on the combined set
python homology_split.py
python train_peptide_admet_model.py --csv data/train.csv
```

### Using the Predictor

#### Python API

```python
from peptide_admet_predictor import PeptideAdmetPredictor

predictor = PeptideAdmetPredictor(model_dir='peptide_admet_model')
out = predictor.predict("WALVKALVNHRISSSLVCG")   # 9 endpoints + composite score
print(out['composite_score'])
print(out['results'])     # per-endpoint: kind, probability/class/value, risk level
```

#### Command Line

```bash
# Single sequence (prints all 9 endpoints + composite score)
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG"

# Batch + multi-objective ranking
python peptide_admet_predictor.py --sequences test_sequences.txt --rank
```

---

## 📋 Feature Space

428-dimensional representation (identical at training and inference time):

### 1. Amino Acid Composition (AAC) — 20 features
Frequency of each of the 20 standard amino acids.

### 2. Dipeptide Composition (DPC) — 400 features
Frequency of all ordered dipeptide pairs.

### 3. Physicochemical Properties — 8 features
Molecular weight (estimate), average hydropathy (Kyte-Doolittle),
hydropathy range, net charge (pH 7), isoelectric point estimate,
GRAVY, hydrophobic residue ratio, charged residue ratio.

---

## 🔬 Model Architecture

A **shared trunk** with **per-task heads** (defined in `admet_model.py`, used
by both trainer and predictor so the architecture and checkpoint format can
never drift apart). The v2.0 `ADMETMLP` (5 binary heads) is kept for loading
old checkpoints; v3.0 uses `MixedADMETMLP`.

- **Trunk:** 428 standardized features → 256 → 128 (ReLU + BatchNorm + Dropout 0.25)
- **Heads (per endpoint kind):**
  - binary (6): one `Linear(128,1)` → sigmoid
  - multiclass: `Linear(128, C)` where C = 6 (`toxicity_type`) or 4 (`neurotoxicity_type`) → softmax
  - regression (1): `Linear(128,1)` → raw value
- **Loss:** sum over the nine endpoints, each reduced **only over its labelled
  rows** (mask-aware). Binary uses `pos_weight` from endpoint prevalence;
  multiclass uses cross-entropy; regression uses MSE.
- **Optimizer:** Adam (lr=1e-3), ReduceLROnPlateau, early stopping on val mixed loss.

Saved artifacts in `peptide_admet_model/`:

```
peptide_admet_model/
├── admet_mlp.pt          # MixedADMETMLP state dict + endpoint/kind metadata
├── scaler.pt             # StandardScaler (torch.save)
└── metrics.json          # MEASURED per-endpoint metrics (both splits) + provenance
```

---

## 🧬 Data & Evaluation Protocol

### Data: `prepare_data.py`

`--n` synthetic peptide sequences (length 10–30) drawn from 200
amino-acid-composition families (Dirichlet profiles). Labels come from a
deliberately crude latent model (length, hydropathy, net charge, aromaticity +
noise) so that honest, measurable metrics emerge. Every synthetic row is
stamped `data_origin=synthetic_demo`.

**Partial labels (pepADMET convention).** `toxicity_type` is fully labelled
(class 0 = non-toxic, consistent with `toxicity_binary`); `neurotoxicity_type`
and `HC50` are present only for a random subset of rows. A `NaN` cell = "not
measured for this row" and is masked out of training — the same sparsity
pepADMET ships, so the masking code path is genuinely exercised.

**Extensibility.** `--merge <csv>` folds in an external dataset produced by
`ingest_external.py`. External rows keep their own `data_origin` and their own
partial labels (they are **not** relabelled by the synthetic model), and the
combined set is deduped by sequence. This is the intended path for adding
*real measured* data.

**Why synthetic core?** The original repository claimed a 15,000-row *real*
dataset but shipped no data file, so no model could be trained or verified.
Rather than fabricate experimental values, the pipeline ships a regenerable,
explicitly-labelled demo set. A real dataset is a CSV with the columns
`sequence, family_id, data_origin` + the 9 endpoint columns (`NaN` = unlabelled).

### Ingesting external data: `ingest_external.py`

Validates an external CSV against the 9-endpoint schema, normalises sequences
(uppercase, standard AA only, length 4–120), de-duplicates, stamps
`data_origin` and `sequence_provenance`, and preserves partial labels. For
SMILES-only inputs (e.g. pepADMET's `Toxicity.csv`) it can attempt
SMILES→sequence recovery via RDKit (`smiles_to_sequence.py`) and flags the
result as `smiles_inferred` (low-trust) — see the note below.

### Split: `homology_split.py`

Sequences are grouped into **families by amino-acid composition** (the feature
space the model actually sees). Families are assigned to train/val/test
(70/10/20) so no composition family appears on both sides of a boundary. The
script verifies and reports the max pairwise Jaccard overlap between train and
test (≤ 0.31 in the 30k run; threshold 0.5) and the per-endpoint label-rate
delta (≤ 0.017), so leakage is *measured*, not assumed. This follows the
homology/leakage-control protocol of AMPBench-MT (arXiv:2607.25518).

> **Honest note on SMILES→sequence.** pepADMET's shipped `Toxicity.csv`
> contains SMILES but **no explicit sequence column**, and its own
> composition reference columns are internally inconsistent with the SMILES
> (composition sums to ~100 vs ~10-residue structures). We therefore do **not**
> treat those 135 rows as clean real data. `ingest_external.py` recovers a
> sequence for the subset of SMILES that parse cleanly and marks them
> `sequence_provenance=smiles_inferred`; in our test only ~14/135 survived
> length/composition sanity checks. The pipeline is *capable* of ingesting
> real peptide sequences, but pepADMET's sample file is not a reliable source.

---

## 📁 Repository Structure

```
openclaw-peptide-admet/
├── endpoint_config.py          # single source of truth for the 9 endpoints
├── prepare_data.py             # generate any-size synthetic set (+ --merge external)
├── ingest_external.py          # validate/dedup/provenance an external CSV
├── smiles_to_sequence.py       # RDKit SMILES -> one-letter sequence (low-trust flag)
├── homology_split.py           # family-disjoint 70/10/20 split + leakage audit
├── admet_model.py              # MixedADMETMLP (+ legacy ADMETMLP) shared by train/predict
├── train_peptide_admet_model.py# mask-aware mixed loss, dual-split metrics.json
├── peptide_admet_predictor.py  # inference CLI, 9 endpoints + composite score
├── peptide_admet_model/        # trained artifacts (admet_mlp.pt, scaler.pt, metrics.json)
├── data/                       # generated CSV + metadata (regenerable; gitignored)
├── test_sequences.txt          # example candidate list for --rank
├── peptide_admet_manuscript_jcim.md   # manuscript
├── cover_letter_jcim.md
├── SUBMISSION_CHECKLIST.md
├── README.md                   # this file
└── LICENSE                     # MIT License
```

---

## 🧪 Usage Examples

### Example 1: Single Sequence Prediction (all 9 endpoints)

```python
from peptide_admet_predictor import PeptideAdmetPredictor

predictor = PeptideAdmetPredictor(model_dir='peptide_admet_model')
out = predictor.predict("WALVKALVNHRISSSLVCG")
# out['composite_score']  : multi-objective score in [0,1]
# out['results']          : per-endpoint (kind, probability/class/value, risk)
# out['endpoints']        : {endpoint: value} for quick access
```

### Example 2: Multi-objective Candidate Ranking (CLI)

```bash
# one sequence per line in candidates.txt
python peptide_admet_predictor.py --sequences candidates.txt --rank
# rows sorted by composite score (geometric mean of favourability across
# the composite endpoints: GI+, Caco2+, Ames-low, hERG-low,
# toxicity-low, toxType-class0, neurotox-class0, HC50-high)
# (BBB is reported for context but not penalised — see endpoint_config.py)
```

### Example 3: Model Provenance (no hardcoded numbers)

```python
info = predictor.model_info()
# {'eval_split': 'homology-controlled ...',
#  'mean_metric_homology': 0.7189,
#  'per_endpoint_homology': {...9 endpoints...},
#  'trained_on': 'synthetic_demo', ...}
```

### Example 4: Fold in a real dataset

```bash
python ingest_external.py --input real.csv --source myrealdata --output data/real.csv
python prepare_data.py --n 30000 --merge data/real.csv --out data/train.csv
python homology_split.py && python train_peptide_admet_model.py --csv data/train.csv
```

---

## 🔗 Context: 2026 AI/AMP Literature

This revision aligns the repository with the methodological consensus of
recent antimicrobial-peptide literature (see
`2026_AI_抗菌胜肽研究報告.md` in the project notes):

- **AMPBench-MT (arXiv:2607.25518, 2026-07)** — benchmark showing that
  sequence-similarity leakage inflates apparent AMP/ADMET performance;
  motivates `homology_split.py`.
- **AMPGAN v3 + PepCraft (arXiv, 2026-06)** — multi-objective scoring of
  generated candidates; motivates the composite score and `--rank` mode.
- **ApexGO (Nature Machine Intelligence, 2026-05)** and the **npj Drug
  Discovery integrated pipeline (2026-05)** — generative redesign of
  antibiotic scaffolds; the honest-evaluation lessons apply equally to
  ADMET predictors.
- **Genotypic Triggers (2026-08)** — pharmacogenomic "back-door" safety
  blind spot; a reminder that ADMET panels without toxicogenomics
  endpoints are incomplete.
- **pepADMET (Tan et al., 中南大學; `ifyoungnet/pepADMET`)** — a peptide
  toxicity GNN with partial-label toxicity-type / neurotoxicity-type / HC50
  endpoints. v3.0 adopts its **endpoint set** (6–9) and **partial-label
  mechanism** into the openclaw pipeline, while keeping openclaw's
  homology-controlled evaluation and reproducible data (both of which
  pepADMET lacks — its training set is not shipped and its split has no
  homology control).

---

## ⚠️ Limitations

1. **Synthetic core data.** The shipped model was trained on regenerated
   synthetic data (plus any `--merge`d external rows) with a crude latent
   label model. All reported metrics characterize the *pipeline*, not
   real-peptide ADMET behaviour.
2. **9 endpoints** (vs. 18+ in AdmetSAR 2.0 / ADMETlab 3.0). Still no
   CYP / DILI / pharmacogenomic endpoints (see Genotypic Triggers, 2026-08).
   The 4 toxicity endpoints match pepADMET, not a full toxicogenomics panel.
3. **Composition-level features.** AAC/DPC cannot capture sequence order
   beyond dipeptides; a language-model or GNN backbone (as pepADMET uses)
   would be needed for order-sensitive properties.
4. **Multiclass imbalance.** `toxicity_type` is dominated by class 0
   (non-toxic), so its macro-F1 is low — an honest consequence of the latent
   model, not a bug. Real balanced toxicity data would change this.
5. **SMILES→sequence is low-trust.** pepADMET's sample `Toxicity.csv` is not
   a reliable real-data source (inconsistent reference columns, no sequence
   column); only a small clean subset survives recovery.
6. **Sequence length.** Demo data spans 10–30 aa; performance outside this
   range is unvalidated.
7. **No experimental validation.** No wet-lab data has been used.

---

## 🎯 Future Directions

1. **Real data integration** — fold in experimental ADMET/AMP data
   (AMPBench-MT tasks, AMPDB-derived sets) via `ingest_external.py --merge`
   with the same split protocol.
2. **Sequence-order features** — pre-trained peptide language models
   (ESM-2, ProtGPT2 soft prompts) or a GNN backbone (as in pepADMET) as the
   feature backbone, while retaining the homology-controlled evaluation.
3. **Generative loop** — couple the predictor with a generator
   (AMPGAN v3-style) for score-based candidate design, including
   non-canonical amino acids and end-group modifications.
4. **Extended endpoints** — CYP inhibition, DILI, pharmacogenomic
   sensitivity.
5. **Active learning** — prioritize experiments for the highest-uncertainty
   candidates.

---

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{peptide_admet_2026,
  author = {OpenClaw Team},
  title  = {Peptide ADMET Prediction: a reproducible mixed multi-task
            pipeline with homology-controlled evaluation},
  year   = {2026},
  url    = {https://github.com/c00jsw00/openclaw-peptide-admet},
  note   = {v3.0: 9 endpoints (incl. pepADMET toxicity set), partial-label
            masking, extensible training set, AMPBench-MT-style split}
}
```

---

**Version**: 3.0 (extensibility + pepADMET endpoint expansion)
**Last Updated**: 2026-08-25
**Author**: OpenClaw Team
