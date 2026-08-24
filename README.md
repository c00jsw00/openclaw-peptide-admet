# Peptide ADMET Prediction Model

**Honest, reproducible PyTorch MLP for peptide ADMET property prediction — with an AMPBench-MT-style homology-controlled evaluation**

![Status](https://img.shields.io/badge/status-demo_pipeline-orange)
![Data](https://img.shields.io/badge/data-synthetic_demo-blue)
![Eval](https://img.shields.io/badge/eval--split-homology--controlled-green)
![License](https://img.shields.io/badge/License-MIT-green)

> **2026-08 revision (integrity update).** The earlier version of this
> repository advertised *97.70% accuracy / 0.9987 AUC* measured on a
> "15,000 real peptide" dataset that was not present in the repository, and
> saved a second Random Forest as the "neural network". Those claims are
> removed. The revised repository is a **fully reproducible demonstration
> pipeline** on a clearly-labelled synthetic demo dataset, evaluated on a
> **homology-controlled** test split in the spirit of AMPBench-MT
> (arXiv:2607.25518), which shows that sequence-similarity leakage is the
> main reason naive ADMET/AMP benchmarks overstate generalization.

---

## 📊 Overview

A single **PyTorch MLP (428 → 256 → 128 → 5)** with one sigmoid head per
endpoint predicts five ADMET properties in peptides:

1. **GI Absorption** (Gastrointestinal absorption)
2. **Caco-2 Permeability** (Intestinal cell permeability)
3. **BBB Penetration** (Blood-brain barrier penetration)
4. **Ames Mutagenicity** (Mutagenicity risk)
5. **hERG Inhibition** (Cardiotoxicity risk)

In addition, the predictor reports a **composite multi-objective score** —
the geometric mean of the *favourable* probability of each endpoint —
following the multi-objective candidate ranking used by generative AMP
design frameworks such as AMPGAN v3 / PepCraft (arXiv, 2026-06).

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

Measured on the **homology-controlled test split** (3,020 sequences;
training data is the synthetic demo set):

| Metric | Value |
|--------|-------|
| **Macro AUC-ROC (headline)** | **0.8684** |
| **Mean accuracy** | **0.7836** |
| Random-split macro AUC (comparison) | 0.8688 |
| Homology-vs-random AUC delta | +0.0004 |

### Per-Endpoint (homology-controlled test split)

| Endpoint | AUC | MCC | Accuracy | Positive rate |
|----------|-----|-----|----------|---------------|
| GI Absorption | 0.8810 | 0.4457 | 0.8037 | 0.132 |
| Caco-2 Permeability | 0.8882 | 0.5930 | 0.8094 | 0.319 |
| BBB Penetration | 0.9070 | 0.4575 | 0.8367 | 0.105 |
| Ames Mutagenicity | 0.8011 | 0.3418 | 0.7016 | 0.171 |
| hERG Inhibition | 0.8645 | 0.5261 | 0.7665 | 0.299 |

> ⚠️ **These numbers characterize the demo pipeline, not real-peptide
> performance.** The labels are drawn from a latent physicochemical model,
> so ~0.8–0.9 AUC is exactly what should be expected. On *experimental*
> peptide data, real ADMET predictors (AdmetSAR, SwissADME, ADMETlab)
> typically report per-endpoint AUCs in a comparable or lower range — the
> old 0.9987 figure was a leakage artifact, not a property of the model.

The random-split comparison (macro AUC 0.8688) exists to quantify the
"memorization" gap that AMPBench-MT (arXiv:2607.25518) documents: when
training and test sequences are drawn from the same composition families,
apparent performance inflates. Here the gap is small because the demo
labels are driven by physicochemistry, but the *protocol* is what matters
for real data.

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
python prepare_data.py                # 15,000 synthetic_demo rows, seed 42
python homology_split.py              # family-disjoint 70/10/20 split
python train_peptide_admet_model.py   # trains MLP, writes metrics.json
```

### Using the Predictor

#### Python API

```python
from peptide_admet_predictor import PeptideADMETPredictor

predictor = PeptideADMETPredictor(model_dir='peptide_admet_model')

results = predictor.predict("WALVKALVNHRISSSLVCG")   # single sequence
predictor.print_result(results)

results = predictor.predict(["GAGAGAGAGAGA", "KKKKKKKKKK"])   # batch
ranked = predictor.rank_candidates(["GAGAGAGAGAGA", "KKKKKKKKKK"])  # composite score
```

#### Command Line

```bash
# Single sequence
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

A single shared MLP with one classification head per endpoint
(defined in `admet_model.py`, used by both trainer and predictor so the
architecture and checkpoint format can never drift apart):

- Input: 428 standardized features
- Hidden layers: 256 → 128 (ReLU + BatchNorm + Dropout 0.2)
- Output: 5 sigmoid heads (one per endpoint)
- Loss: mean per-endpoint BCE (class weights from endpoint prevalence)
- Optimizer: Adam (lr=3e-4), ReduceLROnPlateau, early stopping on val BCE

Saved artifacts in `peptide_admet_model/`:

```
peptide_admet_model/
├── admet_mlp.pt          # PyTorch state dict + architecture metadata
├── scaler.pt             # StandardScaler (torch.save)
├── metrics.json          # MEASURED per-endpoint AUC/MCC/Acc, both splits
└── feature_names.txt     # 428 feature names (AAC_*, DPC_*, physchem)
```

---

## 🧬 Data & Evaluation Protocol

### Data: `prepare_data.py`

15,000 synthetic peptide sequences (length 10–30) drawn from 200
amino-acid-composition families (Dirichlet profiles). Labels come from a
deliberately crude latent model (length, hydropathy, net charge,
aromaticity + noise) so that honest, measurable AUCs emerge. Every row is
stamped `data_origin=synthetic_demo`.

**Why synthetic?** The original repository claimed a 15,000-row *real*
dataset but shipped no data file, so no model could be trained or verified.
Rather than fabricate experimental values, the pipeline now ships a
regenerable, explicitly-labelled demo set. Swapping in real data requires
only producing a CSV with the same columns (`sequence, family_id,
data_origin, GI_absorption, Caco2_permeability, BBB_penetration,
Ames_mutagenicity, hERG_inhibition`).

### Split: `homology_split.py`

Sequences are grouped into **families by amino-acid composition** (the
feature space the model actually sees). Families are then assigned to
train/val/test (70/10/20) so that no composition family appears on both
sides of a boundary. The script verifies and reports the maximum pairwise
Jaccard overlap between train and test/test composition profiles
(0.25 in this run; threshold 0.5) and the per-endpoint label-rate delta
between splits (≤ 0.013), so leakage is *measured*, not assumed. This
follows the homology/leakage-control protocol of AMPBench-MT
(arXiv:2607.25518) and is the standard fix for the "sequence-similarity
memorization" failure mode documented across 2026 AMP benchmarks.

---

## 📁 Repository Structure

```
openclaw-peptide-admet/
├── prepare_data.py             # regenerate the synthetic_demo dataset
├── homology_split.py           # family-disjoint 70/10/20 split + leakage audit
├── admet_model.py              # shared MLP definition (trainer + predictor)
├── train_peptide_admet_model.py# training, dual-split evaluation, metrics.json
├── peptide_admet_predictor.py  # inference CLI + composite multi-objective score
├── peptide_admet_model/        # trained artifacts (admet_mlp.pt, scaler.pt, metrics.json)
├── data/                       # generated CSV + metadata (regenerable; gitignored)
├── test_sequences.txt          # example candidate list for --rank
├── peptide_admet_manuscript_jcim.md   # manuscript (updated to measured metrics)
├── cover_letter_jcim.md
├── SUBMISSION_CHECKLIST.md
├── README.md                   # this file
└── LICENSE                     # MIT License
```

---

## 🧪 Usage Examples

### Example 1: Single Sequence Prediction

```python
from peptide_admet_predictor import PeptideADMETPredictor

predictor = PeptideADMETPredictor(model_dir='peptide_admet_model')
results = predictor.predict("WALVKALVNHRISSSLVCG")
predictor.print_result(results)
```

### Example 2: Multi-objective Candidate Ranking

```python
candidates = ["GAGAGAGAGAGA", "MLLLLLLLLL", "KKKKKKKKKK", "ACDE"]
ranked = predictor.rank_candidates(candidates)
# ranked[0] = best composite score (geometric mean of favourable
# endpoint probabilities: GI+, Caco2+, BBB+, Ames-low, hERG-low)
```

### Example 3: Model Provenance (no hardcoded numbers)

```python
info = predictor.model_info()
# {'eval_split': 'homology-controlled ...',
#  'mean_auc_homology_split': 0.8684,
#  'trained_on': 'synthetic_demo', ...}
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

---

## ⚠️ Limitations

1. **Synthetic demo data.** The shipped model was trained on regenerated
   synthetic data with a crude latent label model. All reported metrics
   characterize the *pipeline*, not real-peptide ADMET behaviour.
2. **5 endpoints only** (vs. 18+ in AdmetSAR 2.0 / ADMETlab 3.0). No
   toxicogenomic / pharmacogenomic endpoints (see Genotypic Triggers,
   2026-08).
3. **Composition-level features.** AAC/DPC cannot capture sequence order
   beyond dipeptides; a language-model or GNN backbone would be needed
   for order-sensitive properties.
4. **Sequence length.** Demo data spans 10–30 aa; performance outside
   this range is unvalidated.
5. **No experimental validation.** No wet-lab data has been used.

---

## 🎯 Future Directions

1. **Real data integration** — retrain on experimental ADMET/AMP data
   (AMPBench-MT tasks, AMPDB-derived sets) with the same split protocol.
2. **Sequence-order features** — pre-trained peptide language models
   (ESM-2, ProtGPT2 soft prompts) as the feature backbone.
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
  title  = {Peptide ADMET Prediction: a reproducible demo pipeline with
            homology-controlled evaluation},
  year   = {2026},
  url    = {https://github.com/c00jsw00/openclaw-peptide-admet},
  note   = {Revised 2026-08: synthetic_demo data, AMPBench-MT-style split}
}
```

---

**Version**: 2.0 (integrity revision)
**Last Updated**: 2026-08-24
**Author**: OpenClaw Team
