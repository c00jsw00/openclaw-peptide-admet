# Peptide ADMET Predictor (Inference)

**Honest, reproducible multi-task PyTorch MLP for peptide ADMET prediction — with an AMPBench-MT-style homology-controlled evaluation**

![Status](https://img.shields.io/badge/Status-synthetic%20demo%20benchmark-blue)
![Macro AUC (homology, measured)](https://img.shields.io/badge/Macro%20AUC-0.8684%20(homology,%20measured)-green)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ **This tool predicts five ADMET endpoints on a *synthetic* demo dataset.**
> All metrics shown are **measured** values from `metrics.json` (no hardcoded
> numbers). The model is a real PyTorch multi-task MLP. There is no Random
> Forest component and no "ensemble" — the v1.0 claims (97.70% accuracy,
> 0.9987 AUC, RF+NN ensemble) were removed in the 2026-08 integrity revision.

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

## 📊 What It Predicts

1. **GI Absorption** (腸胃吸收率)
2. **Caco-2 Permeability** (腸道穿透性)
3. **BBB Penetration** (血腦屏障穿透)
4. **Ames Mutagenicity** (致突變性)
5. **hERG Inhibition** (心毒性)

### Measured Performance (from `metrics.json`)

| Endpoint | AUC | MCC | Accuracy (homology test) |
|---|---|---|---|
| GI Absorption | 0.8810 | 0.4457 | 0.8037 |
| Caco-2 Permeability | 0.8882 | 0.5930 | 0.8094 |
| BBB Penetration | 0.9070 | 0.4575 | 0.8367 |
| Ames Mutagenicity | 0.8011 | 0.3418 | 0.7016 |
| hERG Inhibition | 0.8645 | 0.5261 | 0.7665 |
| **Macro AUC** | **0.8684** | — | mean 0.7836 |

(Comparison random split: macro AUC 0.8688 — see README for the leakage discussion.)

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

A single multi-task PyTorch MLP (shared in `admet_model.py`):

```
428 → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.2)
    → 5 × Linear(1) + Sigmoid
```

144,133 parameters. Trained with Adam + `ReduceLROnPlateau`, early stopping on
validation BCE. No ensemble, no Random Forest.

### Composite Score

```
score = ( p(GI) · p(Caco-2) · p(BBB) · (1 − p(Ames)) · (1 − p(hERG)) )^(1/5)
```

Geometric mean: a single poor endpoint drags the score down.

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
3. `peptide_admet_manuscript_jcim.md` — v2.0 manuscript (integrity revision).
4. AMPBench-MT (arXiv:2607.25518) — homology-controlled evaluation.
5. AMPGAN v3 / PepCraft (arXiv, 2026-06) — multi-objective candidate ranking.

---

## 📄 License

MIT License.

**Version**: 2.0 (integrity revision)
**Last Updated**: 2026-08-24
**Author**: Pinwan (OpenClaw Team)
