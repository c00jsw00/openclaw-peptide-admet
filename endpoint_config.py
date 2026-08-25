#!/usr/bin/env python3
"""
endpoint_config.py
==================

Single source of truth for the endpoint set of the openclaw peptide ADMET
pipeline (v3.0).

v3.0 extends the original 5 binary endpoints with the four **pepADMET**
toxicity endpoints (Tan et al., 中南大學; repo ifyoungnet/pepADMET), giving a
mixed multi-task set:

  * 5  binary     — ADME + 2 safety (GI absorption, Caco-2, BBB, Ames, hERG)
  * 1  multiclass — toxicity type (6 classes, 0..5)
  * 1  multiclass — neurotoxicity type (4 classes, 0..3)
  * 1  regression — HC50 (half-effective concentration, ~log scale 0..3)

Every endpoint carries a ``kind`` and a ``num_classes`` so the model, trainer
and predictor all agree on how to read/write that column.

Partial labels (pepADMET mechanism)
-----------------------------------
pepADMET labels rows sparsely: a given row may carry a toxicity-type label but
no neurotoxicity-type label, etc.  We follow the same convention here:

  * **NaN in an endpoint column = "not labelled for this row"** = *mask off*
    for that (row, endpoint) pair.  The trainer never penalises a masked cell,
    and the predictor reports ``None`` for it.
  * ``prepare_data.py`` (synthetic) labels every endpoint by default so the
    pipeline always has something to train; ``ingest_external.py`` preserves
    the external file's NaNs as masks.  Pass ``--partial`` to prepare_data to
    synthesise realistic sparsity for benchmarking.

Column naming
-------------
Internal canonical column names (used everywhere in the repo) are the
``column`` fields below.  pepADMET's native column names are recorded in
``pep_column`` so ``ingest_external.py`` can map an external pepADMET CSV onto
the canonical schema without ambiguity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --------------------------------------------------------------------------- #
# Endpoint kinds
# --------------------------------------------------------------------------- #
KIND_BINARY = 'binary'
KIND_MULTICLASS = 'multiclass'
KIND_REGRESSION = 'regression'

# Acceptable one-letter peptide sequence length window. Rows outside this are
# dropped at ingestion (too short to be a peptide / too long for the feature
# pipeline's fixed-length assumption).
SEQUENCE_MIN_LEN = 4
SEQUENCE_MAX_LEN = 120
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Endpoint:
    """One prediction target."""
    name: str                 # canonical name (also the CSV column)
    kind: str                 # 'binary' | 'multiclass' | 'regression'
    num_classes: int = 1      # for multiclass
    pep_column: Optional[str] = None   # pepADMET native column (None = openclaw-native)
    pep_class_labels: Optional[Dict[int, str]] = None  # class id -> human label
    description: str = ""
    # composite-score role
    in_composite: bool = True      # does it enter the overall "risk" score?
    higher_is_worse: bool = True   # for composite aggregation direction
    # direction hint used to fold a probability/estimate into [0,1] risk
    risk_direction: str = 'prob'   # 'prob' (P(positive)) | 'class' | 'value'


# --------------------------------------------------------------------------- #
# The 9 endpoints (v3.0)
# --------------------------------------------------------------------------- #
ENDPOINTS: List[Endpoint] = [
    # ---- original 5 binary (openclaw-native) ----------------------------- #
    # NOTE direction: GI absorption & Caco-2 permeability are FAVORABLE when
    # high (good oral delivery) -> higher_is_worse=False. BBB penetration is
    # desirable for CNS drugs (neutral otherwise) -> higher_is_worse=False;
    # it is reported for context but NOT penalised in the composite, because
    # crossing the BBB is a property, not a defect, for non-CNS drugs.
    Endpoint('GI_absorption', KIND_BINARY, 1, None, None,
             'Oral GI absorption (fraction absorbed > threshold)',
             in_composite=True, higher_is_worse=False, risk_direction='prob'),
    Endpoint('Caco2_permeability', KIND_BINARY, 1, None, None,
             'Caco-2 cell-line permeability (high vs low)',
             in_composite=True, higher_is_worse=False, risk_direction='prob'),
    Endpoint('BBB_penetration', KIND_BINARY, 1, None, None,
             'Blood-brain barrier penetration (high vs low)',
             in_composite=False, higher_is_worse=False, risk_direction='prob'),
    Endpoint('Ames_mutagenicity', KIND_BINARY, 1, None, None,
             'Ames mutagenicity (mutagenic vs not)',
             in_composite=True, higher_is_worse=True, risk_direction='prob'),
    Endpoint('hERG_inhibition', KIND_BINARY, 1, None, None,
             'hERG channel inhibition (positive vs negative)',
             in_composite=True, higher_is_worse=True, risk_direction='prob'),

    # ---- pepADMET toxicity endpoints -------------------------------------- #
    Endpoint('toxicity_binary', KIND_BINARY, 1, 'toxicity_nontoxicity', None,
             'Overall cytotoxicity (toxic vs non-toxic)',
             in_composite=True, higher_is_worse=True, risk_direction='prob'),
    Endpoint('toxicity_type', KIND_MULTICLASS, 6, 'toxicity_type_class',
             {0: 'non-toxic', 1: 'hepatotoxic', 2: 'cardiotoxic',
              3: 'neurotoxic', 4: 'nephrotoxic', 5: 'hematotoxic'},
             'Toxicity mechanism/type (6 classes)',
             in_composite=True, higher_is_worse=True, risk_direction='class'),
    Endpoint('neurotoxicity_type', KIND_MULTICLASS, 4, 'neurotoxicity_type_class',
             {0: 'non-neurotoxic', 1: 'neurotoxic_A',
              2: 'neurotoxic_B', 3: 'neurotoxic_C'},
             'Neurotoxicity subtype (4 classes)',
             in_composite=True, higher_is_worse=True, risk_direction='class'),
    Endpoint('HC50', KIND_REGRESSION, 1, 'HC50', None,
             'Half-maximal cytotoxicity concentration (~log scale); lower = more potent',
             in_composite=True, higher_is_worse=False, risk_direction='value'),
]

ENDPOINT_NAMES: List[str] = [e.name for e in ENDPOINTS]
ENDPOINT_BY_NAME: Dict[str, Endpoint] = {e.name: e for e in ENDPOINTS}
N_ENDPOINTS = len(ENDPOINTS)

# Convenience splits
BINARY_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_BINARY]
MULTICLASS_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_MULTICLASS]
REGRESSION_NAMES = [e.name for e in ENDPOINTS if e.kind == KIND_REGRESSION]
PEPADMET_NAMES = [e.name for e in ENDPOINTS if e.pep_column is not None]

# Composite-score inputs (the endpoints folded into the overall risk number)
COMPOSITE_NAMES = [e.name for e in ENDPOINTS if e.in_composite]


def endpoint(name: str) -> Endpoint:
    return ENDPOINT_BY_NAME[name]


def n_multiclass() -> int:
    return sum(1 for e in ENDPOINTS if e.kind == KIND_MULTICLASS)


if __name__ == '__main__':
    print(f'{N_ENDPOINTS} endpoints:')
    for e in ENDPOINTS:
        extra = f' (num_classes={e.num_classes})' if e.kind == KIND_MULTICLASS else ''
        pep = f'  pepADMET[{e.pep_column}]' if e.pep_column else ''
        comp = '' if e.in_composite else '  [not in composite]'
        print(f'  {e.name:22s} {e.kind:11s}{extra}{pep}{comp}')
