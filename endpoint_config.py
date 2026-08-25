#!/usr/bin/env python3
"""
endpoint_config.py
==================

Single source of truth for the endpoint set of the openclaw peptide ADMET
pipeline (v4.0 — real-data edition).

v4.0 replaces the synthetic 9-endpoint set with the four endpoints requested
for the **Chemit797/PepADMET-Dataset** release (cleaned ``整理/`` tables):

  * Hemolysis      — binary     (hemolytic 1 / non-hemolytic 0)   [SEQUENCE modality]
  * Half-life      — regression (plasma half-life, seconds)        [SEQUENCE modality]
  * Caco-2         — regression (apparent permeability, logPapp)   [MOLECULAR modality]
  * PAMPA/MDCK     — regression (PAMPA apparent permeability)      [MOLECULAR modality]

Two modalities
--------------
The four datasets are **disjoint molecules** that fall into two feature spaces:

  * ``sequence``   — hemolysis & half-life ship clean one-letter 20-AA peptide
    sequences -> the 428-dim sequence feature vector (AAC+DPC+phys-chem).
  * ``molecular``  — caco-2 & pampa ship only usable SMILES (their ``Sequence``
    column is a non-standard CycPeptMPDB residue-name list, not a 20-AA string)
    -> the ~2265-dim RDKit molecular feature vector (2D descriptors + Morgan).

Each ``Endpoint`` therefore carries a ``modality`` so the trainer / predictor
can route a row to the right feature extractor and load the matching model.
The four datasets do not overlap, so the pipeline trains **four focused
single-task models**, one per endpoint (a single shared trunk would just learn
two disconnected zero-padded subspaces).

Data provenance & honesty
-------------------------
  * Hemolysis ``label`` and Half-life ``half_life_seconds`` are the dataset's
    curated measured values (see the source repo's ``dataset_stats`` /
    ``Final_Datasets_Overview``).  We do NOT fabricate or impute labels.
  * Caco-2 ``Permeability`` and PAMPA ``PAMPA`` are already log-scale apparent
    permeability (logPapp) values; we train on them directly.
  * Rows with a null/unparseable input for their modality are dropped at
    ingestion (never zero-filled into training) so no fake features enter the
    model.  Nulls are counted and reported in the data meta JSON.

Column naming
-------------
``column`` is the canonical internal name (also the CSV column the prepared
per-endpoint table writes).  ``source_column`` is the raw column in the
Chemit797 release we read from; ``seq_column`` / ``smiles_column`` name the
raw input feature column for the endpoint's modality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --------------------------------------------------------------------------- #
# Endpoint kinds
# --------------------------------------------------------------------------- #
KIND_BINARY = 'binary'
KIND_MULTICLASS = 'multiclass'
KIND_REGRESSION = 'regression'

# --------------------------------------------------------------------------- #
# Feature modalities
# --------------------------------------------------------------------------- #
MODALITY_SEQUENCE = 'sequence'    # clean 20-AA one-letter peptide string
MODALITY_MOLECULAR = 'molecular'  # SMILES -> RDKit 2D descriptors + Morgan

# Acceptable one-letter peptide sequence length window (sequence modality).
SEQUENCE_MIN_LEN = 4
SEQUENCE_MAX_LEN = 120
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Endpoint:
    """One prediction target."""
    name: str                 # canonical name (also the CSV column)
    kind: str                 # 'binary' | 'multiclass' | 'regression'
    modality: str = MODALITY_SEQUENCE   # 'sequence' | 'molecular'
    num_classes: int = 1      # for multiclass
    # raw-column mapping into the Chemit797 release
    source_file: str = ""     # relative path under the release root
    source_column: str = ""   # raw label column
    seq_column: Optional[str] = None     # raw sequence column (sequence modality)
    smiles_column: Optional[str] = None  # raw SMILES column (molecular modality)
    # label transform (regression only)
    target_transform: str = 'identity'   # 'identity' | 'log10' | 'log10plus'
    raw_units: str = ''                  # human units of the RAW source column
    description: str = ""
    # composite-score role
    in_composite: bool = True      # does it enter the overall "risk" score?
    higher_is_worse: bool = True   # for composite aggregation direction
    # direction hint used to fold a probability/estimate into [0,1] risk
    risk_direction: str = 'prob'   # 'prob' (P(positive)) | 'class' | 'value'


# --------------------------------------------------------------------------- #
# The 4 endpoints (v4.0 — Chemit797/PepADMET-Dataset)
# --------------------------------------------------------------------------- #
ENDPOINTS: List[Endpoint] = [
    Endpoint('Hemolysis', KIND_BINARY, MODALITY_SEQUENCE, 1,
             source_file='整理/hemolysis_unified/hemolysis_unified.csv',
             source_column='label',
             seq_column='sequence_std',
             description='Hemolytic activity of the peptide (1 = hemolytic, 0 = non-hemolytic)',
             in_composite=True, higher_is_worse=True, risk_direction='prob'),

    Endpoint('Half_life', KIND_REGRESSION, MODALITY_SEQUENCE, 1,
             source_file='整理/half_life_unified/half_life_final_minimal.csv',
             source_column='half_life_seconds',
             seq_column='sequence',
             target_transform='log10',
             raw_units='seconds',
             description='Plasma half-life (seconds); modelled in log10 space',
             in_composite=True, higher_is_worse=False, risk_direction='value'),

    Endpoint('Caco2', KIND_REGRESSION, MODALITY_MOLECULAR, 1,
             source_file='整理/caco2_out/caco2_unified.csv',
             source_column='Permeability',
             smiles_column='SMILES',
             target_transform='identity',
             raw_units='logPapp',
             description='Caco-2 apparent permeability (logPapp); molecular features',
             in_composite=True, higher_is_worse=False, risk_direction='value'),

    Endpoint('PAMPA_MDCK', KIND_REGRESSION, MODALITY_MOLECULAR, 1,
             source_file='整理/permeability_out/permeability_unified.csv',
             source_column='PAMPA',
             smiles_column='SMILES',
             target_transform='identity',
             raw_units='logPapp',
             description='PAMPA apparent permeability (logPapp); molecular features',
             in_composite=True, higher_is_worse=False, risk_direction='value'),
]

ENDPOINT_NAMES: List[str] = [e.name for e in ENDPOINTS]
ENDPOINT_BY_NAME: Dict[str, Endpoint] = {e.name: e for e in ENDPOINTS}
N_ENDPOINTS = len(ENDPOINTS)

# Convenience splits
BINARY_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_BINARY]
MULTICLASS_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_MULTICLASS]
REGRESSION_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_REGRESSION]
SEQUENCE_NAMES = [e.name for e in ENDPOINTS if e.modality == MODALITY_SEQUENCE]
MOLECULAR_NAMES = [e.name for e in ENDPOINTS if e.modality == MODALITY_MOLECULAR]

# Composite-score inputs (the endpoints folded into the overall risk number)
COMPOSITE_NAMES = [e.name for e in ENDPOINTS if e.in_composite]


def endpoint(name: str) -> Endpoint:
    return ENDPOINT_BY_NAME[name]


def n_multiclass() -> int:
    return sum(1 for e in ENDPOINTS if e.kind == KIND_MULTICLASS)


if __name__ == '__main__':
    print(f'{N_ENDPOINTS} endpoints (v4.0 real-data):')
    for e in ENDPOINTS:
        extra = f' (num_classes={e.num_classes})' if e.kind == KIND_MULTICLASS else ''
        comp = '' if e.in_composite else '  [not in composite]'
        print(f'  {e.name:12s} {e.kind:11s} {e.modality:10s}{extra}{comp}')
