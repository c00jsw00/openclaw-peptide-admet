#!/usr/bin/env python3
"""
peptide_admet_predictor.py
==========================

v4.0 real-data predictor for the four-endpoint **Chemit797/PepADMET-Dataset**
pipeline.  Each endpoint is an independent single-task ``MixedADMETMLP`` loaded
from ``models_v4/<slug>/admet_mlp.pt`` (+ ``scaler.pt``), matched to the
modality its data has:

  ====================  ========  ============  ==============  ====================
  endpoint              kind      modality      input           output
  ====================  ========  ============  ==============  ====================
  Hemolysis             binary    sequence      one-letter seq  P(hemolytic)
  Half_life             reg       sequence      one-letter seq  half-life [s]
  Caco2                 reg       molecular     SMILES          logPapp [-]
  PAMPA_MDCK            reg       molecular     SMILES          logPapp [-]
  ====================  ========  ============  ==============  ====================

The four models are *not* combined into a composite score — the four datasets
are disjoint molecules in two different feature spaces, so there is no
meaningful cross-endpoint aggregate (this is stated in the metrics file too).

Usage
-----
    # one sequence -> the two sequence endpoints
    python peptide_admet_predictor.py --sequence ACDEFGHIKLMNPQRSTVWY

    # one SMILES -> the two molecular endpoints
    python peptide_admet_predictor.py --smiles "CC(=O)NC..."

    # both at once
    python peptide_admet_predictor.py --sequence ACDEF... --smiles "CC(=O)NC..."

    # a batch CSV with optional 'sequence' and/or 'smiles' columns
    python peptide_admet_predictor.py --csv data/test_candidates.csv

    # restrict which endpoints to run
    python peptide_admet_predictor.py --sequence ACDEF... --endpoints Hemolysis

    # --out writes a JSON results file
    python peptide_admet_predictor.py --csv data/test_candidates.csv --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from admet_model import MixedADMETMLP
from endpoint_config import ENDPOINTS, ENDPOINT_BY_NAME
from feature_extractor import molecule_features, sequence_features

SLUGS = {ep.name: ep.name.lower().replace(' ', '_') for ep in ENDPOINTS}
SEQ_ENDPOINTS = [e.name for e in ENDPOINTS if e.modality == 'sequence']
MOL_ENDPOINTS = [e.name for e in ENDPOINTS if e.modality == 'molecular']


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
class EndpointModel:
    """One single-task model + its scaler, ready for inference."""

    def __init__(self, name, model_root):
        self.name = name
        self.ep = ENDPOINT_BY_NAME[name]
        d = Path(model_root) / SLUGS[name]
        pt = d / 'admet_mlp.pt'
        sc = d / 'scaler.pt'
        if not pt.exists():
            raise FileNotFoundError(
                f'model for {name} not found at {pt}; run train_pepadmet_model.py')
        blob = torch.load(pt, map_location='cpu', weights_only=False)
        self.model = MixedADMETMLP(input_dim=blob['input_dim'],
                                   endpoints=blob['endpoints'])
        self.model.load_state_dict(blob['state_dict'])
        self.model.eval()
        self.scaler = StandardScaler()
        if sc.exists():
            self.scaler = torch.load(sc, map_location='cpu', weights_only=False)
        self.device = torch.device('cpu')

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (N, D) raw features -> model-space predictions (N,)."""
        Xs = np.asarray(self.scaler.transform(np.asarray(X)), dtype=np.float32)
        out = []
        for i in range(0, len(Xs), 2048):
            t = torch.from_numpy(Xs[i:i + 2048])
            o = self.model(t)[self.name]
            if self.ep.kind == 'binary':
                o = torch.sigmoid(o).squeeze(1)
            else:
                o = o.squeeze(1)
            out.append(o.numpy())
        return np.concatenate(out, axis=0)

    def to_readable(self, preds: np.ndarray) -> np.ndarray:
        """Convert model-space predictions back to human units."""
        if self.ep.kind == 'binary':
            return preds  # probability of the positive class
        if self.ep.target_transform == 'log10':
            return np.power(10.0, preds)  # back to raw units (e.g. seconds)
        return preds  # identity (already in raw units, e.g. logPapp)


def load_endpoint_models(names, model_root):
    return {n: EndpointModel(n, model_root) for n in names}


# --------------------------------------------------------------------------- #
# Feature construction per modality
# --------------------------------------------------------------------------- #
def build_features(ep_name, sequences, smiles):
    """Return (X, n_valid, valid_mask) for the endpoint's modality.

    A row is valid only if its required input (sequence or SMILES) is present.
    X has one row per VALID input (compact), and ``valid_mask`` is boolean over
    the original rows.  Returns (None, 0, valid_mask) when no row is valid.
    """
    ep = ENDPOINT_BY_NAME[ep_name]
    if ep.modality == 'sequence':
        valid = np.array([s is not None and len(str(s)) >= 3
                          and set(str(s).upper()) <= set('ACDEFGHIKLMNPQRSTVWY')
                          for s in sequences], dtype=bool)
        clean = [str(sequences[i]).upper()
                 for i in range(len(sequences)) if valid[i]]
        if not clean:
            return None, 0, valid
        X = sequence_features(clean).astype(np.float32)
        return X, len(X), valid
    else:  # molecular
        valid = np.array([s is not None and str(s).strip() not in ('', 'nan', 'None')
                          for s in smiles], dtype=bool)
        clean = [str(smiles[i]).strip()
                 for i in range(len(smiles)) if valid[i]]
        if not clean:
            return None, 0, valid
        X = molecule_features(clean).astype(np.float32)
        return X, len(X), valid


# --------------------------------------------------------------------------- #
# Prediction driver
# --------------------------------------------------------------------------- #
def predict_rows(rows, endpoints, model_root):
    """rows: list of dicts each with optional 'sequence' / 'smiles'.

    Returns a list of result dicts, one per row, each mapping endpoint ->
    {'value': float|None, 'unit': str, 'ok': bool, ...}.
    """
    seqs = [r.get('sequence') for r in rows]
    smis = [r.get('smiles') for r in rows]
    models = load_endpoint_models(endpoints, model_root)

    # For each endpoint: build compact feature matrix, run the model, and keep
    # the boolean valid-mask (over original rows) + the compact predictions.
    cache = {}
    for name in endpoints:
        X, n_valid, valid = build_features(name, seqs, smis)
        if X is None:
            cache[name] = (None, valid)
            continue
        preds = models[name].to_readable(models[name].predict(X))
        cache[name] = (preds, valid)

    # compact index for each ORIGINAL row (or -1 when invalid)
    compact_idx = {}
    for name in endpoints:
        preds, valid = cache[name]
        running = -1
        idxs = []
        for i, v in enumerate(valid):
            if v:
                running += 1
            idxs.append(running)
        compact_idx[name] = idxs

    results = []
    for i, r in enumerate(rows):
        rowres = {}
        for name in endpoints:
            preds, valid = cache[name]
            if preds is None or not valid[i]:
                rowres[name] = {'value': None, 'unit': _unit(name),
                                'ok': False, 'reason': 'no input'}
                continue
            j = compact_idx[name][i]
            rowres[name] = {'value': float(preds[j]), 'unit': _unit(name),
                            'ok': True}
        results.append(rowres)
    return results


def _unit(name):
    ep = ENDPOINT_BY_NAME[name]
    if ep.kind == 'binary':
        return 'P(positive)'
    if ep.target_transform == 'log10':
        return ep.raw_units  # 'seconds'
    return ep.raw_units  # 'logPapp'


# --------------------------------------------------------------------------- #
# Input collection
# --------------------------------------------------------------------------- #
def collect_rows(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        seq_col = 'sequence' if 'sequence' in df.columns else None
        smi_col = 'smiles' if 'smiles' in df.columns else None
        rows = []
        for _, r in df.iterrows():
            rows.append({
                'sequence': (r[seq_col] if seq_col and pd.notna(r[seq_col]) else None),
                'smiles': (r[smi_col] if smi_col and pd.notna(r[smi_col]) else None),
            })
        return rows
    rows = [{'sequence': args.sequence, 'smiles': args.smiles}]
    if not args.sequence and not args.smiles:
        raise SystemExit('provide --sequence, --smiles, or --csv')
    return rows


def resolve_endpoints(args):
    """Pick the endpoints the given inputs can actually support."""
    if args.endpoints:
        for e in args.endpoints:
            if e not in ENDPOINT_BY_NAME:
                raise SystemExit(f'unknown endpoint {e}; valid: {list(ENDPOINT_BY_NAME)}')
        return args.endpoints
    # auto: sequence -> seq endpoints, smiles -> mol endpoints
    sel = []
    if args.sequence:
        sel += SEQ_ENDPOINTS
    if args.smiles:
        sel += MOL_ENDPOINTS
    if args.csv:
        sel = [e.name for e in ENDPOINTS]
    if not sel:
        sel = [e.name for e in ENDPOINTS]
    return sel


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description='v4.0 4-endpoint real-data predictor')
    ap.add_argument('--sequence', type=str, default=None,
                    help='one-letter amino-acid sequence (for sequence endpoints)')
    ap.add_argument('--smiles', type=str, default=None,
                    help='SMILES string (for molecular endpoints)')
    ap.add_argument('--csv', type=str, default=None,
                    help='batch CSV with optional sequence/smiles columns')
    ap.add_argument('--endpoints', nargs='+', default=None,
                    help='subset of: Hemolysis Half_life Caco2 PAMPA_MDCK')
    ap.add_argument('--model-root', type=str, default='models_v4')
    ap.add_argument('--out', type=str, default=None, help='write results to JSON')
    args = ap.parse_args()

    endpoints = resolve_endpoints(args)
    rows = collect_rows(args)
    print(f'Predicting {len(rows)} row(s) over endpoints: {endpoints}\n')

    results = predict_rows(rows, endpoints, args.model_root)

    # print
    for i, res in enumerate(results):
        print(f'--- row {i} ---')
        for name in endpoints:
            rec = res[name]
            if rec['ok']:
                v = rec['value']
                if ENDPOINT_BY_NAME[name].kind == 'binary':
                    print(f'  {name:12s} {rec["unit"]:<12s} = {v:.4f}')
                else:
                    print(f'  {name:12s} {rec["unit"]:<12s} = {v:.4f}')
            else:
                print(f'  {name:12s} (skipped: {rec["reason"]})')
        print()

    if args.out:
        payload = {
            'endpoints': endpoints,
            'n_rows': len(rows),
            'models': {n: {'input_dim': ENDPOINT_BY_NAME[n].kind,
                           'modality': ENDPOINT_BY_NAME[n].modality}
                       for n in endpoints},
            'results': results,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
