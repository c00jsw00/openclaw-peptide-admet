# Usage Guide for Peptide ADMET Prediction Model

> **2026-08 v2.0**: this guide reflects the revised pipeline. The old
> `peptide_admet_inference.py` / `PeptideADMETPredictor` / `rf_model.pkl` /
> `nn_model.pkl` artifacts no longer exist — see `README.md` for what
> replaced them.

## Installation

```bash
pip install torch scikit-learn pandas numpy
# CPU-only torch (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Pipeline (train from scratch)

```bash
python prepare_data.py          # regenerate the 15,000-row synthetic demo CSV (seed 42)
python homology_split.py        # AMPBench-MT-style family-controlled 70/10/20 split + leakage audit
python train_peptide_admet_model.py   # trains the MLP, writes peptide_admet_model/ + metrics.json
```

## Quick Start

```python
from peptide_admet_predictor import PeptideADMETPredictor

predictor = PeptideADMETPredictor(model_dir='peptide_admet_model')

results = predictor.predict("WALVKALVNHRISSSLVCG")   # dict
for ep in results['endpoints']:
    print(ep['endpoint'], round(ep['probability'], 4))
print('composite', round(results['composite_score'], 4))
```

## Command Line

```bash
python peptide_admet_predictor.py --sequence "WALVKALVNHRISSSLVCG"
python peptide_admet_predictor.py --sequences test_sequences.txt --rank   # ranked by composite score
python peptide_admet_predictor.py --sequence "ACDEFGHIK" --output out.json
```

## Model Files Required

```
peptide_admet_model/
├── admet_mlp.pt   # PyTorch state dict
├── scaler.pt      # fitted StandardScaler
└── metrics.json   # measured per-endpoint metrics (printed by the CLI)
```

## API

### `PeptideADMETPredictor(model_dir='peptide_admet_model')`

Loads `admet_mlp.pt` + `scaler.pt`; reads `metrics.json` if present.

### `predict(sequences, endpoints=None) -> dict`

- `sequences`: str or list[str]
- returns: `{'sequence', 'length', 'endpoints': [{endpoint, probability, prediction, interpretation, risk_level}], 'composite_score', 'model_info'}`

### `model_info() -> dict`

Measured-only model summary (type, param count, split, macro AUC, mean
accuracy, per-endpoint AUC, data origin). Never fabricates values: if
`metrics.json` is missing, fields are `None`/`"not measured"`.

### `rank_candidates(sequences) -> list`

Sorts by the composite score (geometric mean over favorable endpoint
probabilities) — the multi-objective prioritization from AMPGAN v3 / PepCraft.

## Feature Engineering (428-dim)

| Block | Dim | Content |
|---|---|---|
| AAC | 20 | amino acid frequencies |
| DPC | 400 | dipeptide frequencies |
| Physchem | 8 | MW proxy (length × 110 Da), avg hydropathy, hydropathy range, net charge @ pH 7, pI estimate, GRAVY, hydrophobic fraction, charged fraction |

Identical code in training and inference.

## Interpretation Guide

| Endpoint | High probability means | Action |
|---|---|---|
| GI Absorption | good oral bioavailability | favorable for oral delivery |
| Caco-2 Permeability | good intestinal permeability | favorable |
| BBB Penetration | may cross blood-brain barrier | useful for CNS targets; check off-target CNS effects |
| Ames Mutagenicity | mutagenicity risk | optimize / re-screen |
| hERG Inhibition | cardiotoxicity risk | critical — reduce cationic/hydrophobic content |

These are **synthetic-demo** probabilities — validate any real candidate
experimentally before acting on them.

## Best Practices

- Batch prediction is faster than many single calls.
- Use `--rank` for candidate prioritization; the composite score penalizes any
  single poor endpoint.
- Do not extrapolate far outside the 10–30 aa trained range.
- Keep the split audit (`data/split/leakage_audit.json`) in mind: the headline
  numbers come from the homology-controlled split.

## Troubleshooting

**Model files missing** — run the three pipeline commands above.
**Invalid sequence** — standard 20 amino acids only.
**`ModuleNotFoundError: torch`** — install torch (CPU index URL above).

## References

1. `README.md` — pipeline + leakage discussion + 2026 citations.
2. `peptide_admet_manuscript_jcim.md` — v2.0 manuscript.
3. AMPBench-MT (arXiv:2607.25518).

**Version**: 2.0
**Last Updated**: 2026-08-24
**Author**: Pinwan (OpenClaw Team)
