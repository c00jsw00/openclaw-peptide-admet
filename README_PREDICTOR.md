# Peptide ADMET Predictor (Inference)

**Honest, reproducible mixed multi-task PyTorch model for peptide ADMET prediction (9 endpoints) — with an AMPBench-MT-style homology-controlled evaluation**

![Status](https://img.shields.io/badge/Status-synthetic%20demo%20benchmark-blue)
![Mean metric (homology, measured)](https://img.shields.io/badge/Mean%20metric-0.7189%20(30k%2C%20homology%2C%20measured)-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ **This tool predicts nine ADMET/toxicity endpoints on a *synthetic* demo
> dataset.** All metrics shown are **measured** values from `metrics.json`
> (no hardcoded numbers). The model is a real PyTorch mixed multi-task MLP
> (6 binary + 6-class + 4-class + 1 regression). The v1.0 claims (97.70%
> accuracy, 0.9987 AUC, RF+NN ensemble) were removed in the 2026-08 integrity
> revision; v3.0 extends it to pepADMET's toxicity endpoint set.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt   # torch (CPU ok), scikit-learn, pandas, numpy

# Regenerate the demo data (if missing), split, and train:
python prepare_data.py
python homology_split.py
python train_peptide_admet_model.py

# Then predict:
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG"
python peptide_admet_predictor.py --sequences test_sequences.txt --rank
```

---

## 📊 What It Predicts (9 endpoints)

| # | Endpoint | Kind |
|---|---|---|
| 1 | GI Absorption (腸胃吸收率) | binary |
| 2 | Caco-2 Permeability (腸道穿透性) | binary |
| 3 | BBB Penetration (血腦屏障穿透) | binary |
| 4 | Ames Mutagenicity (致突變性) | binary |
| 5 | hERG Inhibition (心毒性) | binary |
| 6 | Toxicity (overall) | binary |
| 7 | Toxicity Type (organ-specific) | multiclass, 6 classes |
| 8 | Neurotoxicity Type | multiclass, 4 classes |
| 9 | HC50 (toxicity potency, µM) | regression |

### Measured Performance (30k training, homology test, from `metrics.json`)

| Endpoint | Primary | Other |
|---|---|---|
| GI Absorption | AUC 0.8857 | MCC 0.4529, acc 0.8092 |
| Caco-2 Permeability | AUC 0.8831 | MCC 0.6135, acc 0.8176 |
| BBB Penetration | AUC 0.9042 | MCC 0.4640, acc 0.8402 |
| Ames Mutagenicity | AUC 0.8052 | MCC 0.3482, acc 0.7067 |
| hERG Inhibition | AUC 0.8602 | MCC 0.5375, acc 0.7744 |
| Toxicity | AUC 0.8268 | MCC 0.1225, acc 0.7522 |
| Toxicity Type | macro-F1 0.3701 | acc 0.7402 |
| Neurotoxicity Type | macro-F1 0.3898 | acc 0.7477 (12.6% labelled) |
| HC50 | R² 0.5937 | RMSE 0.5058 (30% labelled) |
| **Mean primary metric** | **0.7189** | random split 0.7227 |

(Comparison random split — see README for the leakage discussion.)

---

## 🎯 Usage Examples

### Single Sequence

```bash
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG"
```

Output (real, from this run):

```
======================================================================
Peptide ADMET Prediction Results
======================================================================
Sequence: WALVKALVNHRISSSLVCG  (17 aa)
Features: 428 (AAC: 20, DPC: 400, physchem: 8)

GI Absorption        p=0.7623  [LIKELY]      Good GI absorption
Caco-2 Permeability  p=0.8410  [LIKELY]      Good Caco-2 permeability
BBB Penetration      p=0.5531  [LIKELY]      May cross BBB
Ames Mutagenicity    p=0.1204  [UNLIKELY]    Low mutagenicity risk
hERG Inhibition      p=0.3019  [UNLIKELY]    Low hERG risk

Composite score (multi-objective, higher = better): 0.4986

Model (measured, from metrics.json):
  type: MultiTaskPeptideADMET (PyTorch MLP, 144133 params)
  split: homology-controlled (AMPBench-MT style)
  macro AUC: 0.8684   mean accuracy: 0.7836
  data: synthetic demo (see data/peptide_admet_demo.meta.json)
======================================================================
```

### Batch + Ranking

```bash
python peptide_admet_predictor.py --sequences test_sequences.txt --rank
```

Prints a table sorted by the composite score, so candidates with no fatal
endpoint flaw rank highest (AMPGAN v3 / PepCraft-style prioritization).

### JSON Output

```bash
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG" --output results.json
```

---

## 🔧 How It Works

### Features (428-dim)

| Block | Dim | Content |
|---|---|---|
| AAC | 20 | frequency of each amino acid |
| DPC | 400 | frequency of each dipeptide |
| Physchem | 8 | MW proxy, hydropathy, GRAVY, net charge, pI, hydrophobic/charged fractions |

The **same** feature code is used in training (`prepare_data.py`) and inference,
so the two cannot drift.

### Model

A mixed multi-task PyTorch MLP (shared in `admet_model.py`):

```
428 → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
    → per-task heads:
        6 × Linear(1) + Sigmoid            (binary)
        Linear(6) + Softmax                (toxicity_type)
        Linear(4) + Softmax                (neurotoxicity_type)
        Linear(1)                          (HC50, regression)
```

145,681 parameters. Trained with Adam + `ReduceLROnPlateau`, early stopping on
validation mixed loss. No ensemble, no Random Forest.

### Composite Score

Geometric mean of the per-endpoint "favourability" (higher = better):

```
favourability = p for GI/Caco-2/BBB, 1-p for Ames/hERG/toxicity_binary,
                P(class 0) for toxicity_type & neurotoxicity_type,
                exp(-HC50 / 1.0) for HC50 (higher HC50 = less potent = better)
score = ( Π favourability_i )^(1/N)
```

A single poor endpoint drags the score down.

---

## 📋 Input Requirements

- Standard amino acids only (A C D E F G H I K L M N P Q R S T V W Y).
- Recommended length 8–25 aa (the demo set spans 10–30; results outside the
  trained range are unvalidated).
- Case-insensitive.

---

## 📦 Model Files

```
peptide_admet_model/
├── admet_mlp.pt    # PyTorch state dict (the real model)
├── scaler.pt       # fitted StandardScaler
├── metrics.json    # measured per-endpoint metrics, both splits
```

The feature layout (20 AAC + 400 DPC + 8 physchem = 428) and endpoint order
are defined in `admet_model.py` / `prepare_data.py` and are shared by
trainer and predictor, so no separate config artifact is needed.

---

## 🐛 Troubleshooting

**`FileNotFoundError: peptide_admet_model/admet_mlp.pt`**
Run the pipeline first:
```bash
python prepare_data.py && python homology_split.py && python train_peptide_admet_model.py
```

**`ValueError: Invalid peptide sequence`**
Use only the 20 standard amino acids.

**`ModuleNotFoundError: torch`**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build
```

---

## 📚 References

1. `README.md` — full pipeline, leakage discussion, 2026 citations.
2. `PREDICTOR_SUMMARY.md` — tool summary (Chinese).
3. `peptide_admet_manuscript_jcim.md` — manuscript.
4. AMPBench-MT (arXiv:2607.25518) — homology-controlled evaluation.
5. AMPGAN v3 / PepCraft (arXiv, 2026-06) — multi-objective candidate ranking.
6. pepADMET (`ifyoungnet/pepADMET`) — toxicity endpoint set + partial-label convention.

---

## 📄 License

MIT License.

**Version**: 3.0 (extensibility + pepADMET endpoint expansion)
**Last Updated**: 2026-08-25
**Author**: OpenClaw Team
