# Usage Guide for Peptide ADMET Prediction Model

> **2026-08 v3.0**: this guide reflects the extended pipeline — 9 endpoints
> (6 binary + 6-class + 4-class + 1 regression), partial-label masking, and
> an extensible training set. The old `peptide_admet_inference.py` /
> `PeptideADMETPredictor` / `rf_model.pkl` / `nn_model.pkl` artifacts no
> longer exist — see `README.md` for what replaced them.

## Installation

```bash
pip install torch scikit-learn pandas numpy rdkit
# CPU-only torch (smaller download):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`rdkit` is only needed for the optional SMILES→sequence ingestion path.

## Pipeline (train from scratch)

```bash
python prepare_data.py --n 30000        # any size; seed 42 (default)
python homology_split.py                # AMPBench-MT-style family-controlled 70/10/20 split + leakage audit
python train_peptide_admet_model.py     # mixed multi-task training, writes peptide_admet_model/ + metrics.json
```

## Extending the training set

```bash
# Option A: just make it bigger (more synthetic rows)
python prepare_data.py --n 100000

# Option B: fold in a real / external dataset (one sequence per row + label columns)
python ingest_external.py --input real.csv --source my_real_data --output data/real.csv
python prepare_data.py --n 30000 --merge data/real.csv
python homology_split.py
python train_peptide_admet_model.py
```

External CSV label columns use the endpoint names (`GI_absorption`,
`Caco2_permeability`, `BBB_penetration`, `Ames_mutagenicity`, `hERG_inhibition`,
`toxicity_binary`, `toxicity_type`, `neurotoxicity_type`, `HC50`); empty cell
= "not measured" (masked out for that endpoint). `ingest_external.py`
validates sequences, dedups, and stamps `data_origin` +
`sequence_provenance` on every row.

## Quick Start

```python
from peptide_admet_predictor import PeptideAdmetPredictor

predictor = PeptideAdmetPredictor(model_dir='peptide_admet_model')

out = predictor.predict("WALVKALVNHRISSSLVCG")
for ep in out['results']:
    print(ep['endpoint'], ep['kind'], ep['value'])
print('composite', round(out['composite_score'], 4))
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
├── admet_mlp.pt   # PyTorch state dict (+ architecture metadata, model_version: v3_mixed)
├── scaler.pt      # fitted StandardScaler
└── metrics.json   # measured per-endpoint metrics (printed by the CLI)
```

The predictor auto-detects v2 (5-head binary) vs v3 (mixed 9-endpoint)
checkpoints.

## API

### `PeptideAdmetPredictor(model_dir='peptide_admet_model')`

Loads `admet_mlp.pt` + `scaler.pt`; reads `metrics.json` if present.

### `predict(sequences, endpoints=None) -> dict`

- `sequences`: str or list[str]
- returns: `{'sequence', 'length', 'results': [{endpoint, kind, value, risk}], 'composite_score', 'endpoints': {name: value}, 'model_info'}`
  - `kind` is `binary` (value = probability), `multiclass` (value = class
    index), or `regression` (value = predicted quantity).

### `model_info() -> dict`

Measured-only model summary (type, param count, split, mean primary metric,
per-endpoint metrics, data origin). Never fabricates values: if
`metrics.json` is missing, fields are `None`/`"not measured"`.

## Feature Engineering (428-dim)

| Block | Dim | Content |
|---|---|---|
| AAC | 20 | amino acid frequencies |
| DPC | 400 | dipeptide frequencies |
| Physchem | 8 | MW proxy (length × 110 Da), avg hydropathy, hydropathy range, net charge @ pH 7, pI estimate, GRAVY, hydrophobic fraction, charged fraction |

Identical code in training and inference.

## Interpretation Guide

| Endpoint | Kind | High value means | Action |
|---|---|---|---|
| GI Absorption | binary | good oral bioavailability | favorable for oral delivery |
| Caco-2 Permeability | binary | good intestinal permeability | favorable |
| BBB Penetration | binary | may cross blood-brain barrier | useful for CNS targets; check off-target CNS effects |
| Ames Mutagenicity | binary | mutagenicity risk | optimize / re-screen |
| hERG Inhibition | binary | cardiotoxicity risk | critical — reduce cationic/hydrophobic content |
| Toxicity (binary) | binary | overall toxicity risk | screen before progression |
| Toxicity Type | 6-class | class 0 = non-toxic; 1–5 = organ-specific | inspect argmax class |
| Neurotoxicity Type | 4-class | class 0 = non-neurotoxic; 1–3 = neurotoxic subtypes | inspect argmax class |
| HC50 | regression | lower = more potent (µM) | prefer higher values |

These are **synthetic-demo** predictions — validate any real candidate
experimentally before acting on them.

## Best Practices

- Batch prediction is faster than many single calls.
- Use `--rank` for candidate prioritization; the composite score penalizes any
  single poor endpoint.
- Do not extrapolate far outside the 10–30 aa trained range.
- Keep the split audit (`data/split/leakage_audit.json`) in mind: the headline
  numbers come from the homology-controlled split.
- For multiclass endpoints on imbalanced data, prefer the probability vector
  (`out['results'][i]['value']` is the argmax class; the model exposes raw
  logits if you subclass) over a hard threshold.

## Troubleshooting

**Model files missing** — run the pipeline commands above.
**Invalid sequence** — standard 20 amino acids only.
**`ModuleNotFoundError: torch`** — install torch (CPU index URL above).
**`ModuleNotFoundError: rdkit`** — only needed for `ingest_external.py` on
SMILES-only files; `pip install rdkit`.

## References

1. `README.md` — pipeline + leakage discussion + 2026 citations.
2. `peptide_admet_manuscript_jcim.md` — manuscript.
3. AMPBench-MT (arXiv:2607.25518).
4. pepADMET (`ifyoungnet/pepADMET`) — source of the toxicity endpoint set and
   the partial-label convention.

**Version**: 3.0
**Last Updated**: 2026-08-25
**Author**: OpenClaw Team
