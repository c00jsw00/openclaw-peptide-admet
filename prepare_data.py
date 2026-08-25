#!/usr/bin/env python3
"""
prepare_data.py
===============

Generate a **reproducible, clearly-labelled DEMO dataset** for the peptide
ADMET predictor (v3.0).

Why this file exists
--------------------
The original repository claimed "15,000 real peptides" but shipped no data at
all, so the pipeline could not be run or reproduced.  This script generates
``--n`` synthetic peptide sequences whose labels are drawn from a simple
*latent physicochemical model* (length, hydropathy, net charge, aromaticity,
charged fraction).  The data is therefore:

  * fully reproducible (fixed seed) at any scale (``--n``),
  * guaranteed to carry learnable signal (labels depend on the 428-dim
    feature vector the model sees),
  * explicitly labelled ``data_origin=synthetic_demo`` in every row and in
    the accompanying metadata, so it can never be mistaken for experimental
    data,
  * **extensible**: ``--merge`` folds in an external (validated) CSV produced
    by ``ingest_external.py`` — real or externally-sourced labels keep their
    own ``data_origin`` and are NOT relabelled by the synthetic model.

v3.0 endpoint set (see ``endpoint_config.py``)
----------------------------------------------
The original 5 binary ADME/safety endpoints are kept, and the four
pepADMET-style endpoints are added with pepADMET's **partial-label**
convention — a label cell is NaN when that endpoint is not measured for the
row, so the trainer masks it out:

  * 6 binary:   GI_absorption, Caco2_permeability, BBB_penetration,
                Ames_mutagenicity, hERG_inhibition, toxicity_binary
  * 2 multiclass: toxicity_type (6 classes, 0 = non-toxic),
                neurotoxicity_type (4 classes)
  * 1 regression: HC50 (half-maximal cytotoxicity, log-ish scale)

In the synthetic generator the 6 binary endpoints and toxicity_type carry a
label on every row, while neurotoxicity_type and HC50 are present only for a
random subset — exactly the partial-label structure pepADMET ships, so the
masking code path is genuinely exercised.

Usage
-----
    python prepare_data.py                                  # 15,000 rows, 9 endpoints
    python prepare_data.py --n 50000 --seed 7               # bigger training set
    python prepare_data.py --merge data/external.csv        # fold in external data
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from endpoint_config import (
    ENDPOINTS,
    ENDPOINT_NAMES as ENDPOINT_ORDER,
    ENDPOINT_BY_NAME,
    BINARY_NAMES as BINARY_ENDPOINTS,
    MULTICLASS_NAMES as MULTICLASS_ENDPOINTS,
    REGRESSION_NAMES as REGRESSION_ENDPOINTS,
    SEQUENCE_MIN_LEN,
)

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
N_AA = len(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Physicochemical feature offsets inside the 428-dim vector
# (20 AAC + 400 DPC + 8 physchem)
I_MW, I_HYDRO, I_HYDRO_R, I_NETQ, I_PI, I_GRAVY, I_HR, I_CR = 420, 421, 422, 423, 424, 425, 426, 427

# --- physicochemical constants (identical to PeptideFeatureExtractor) -------
HYDROPATHY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
    'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
    'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
    'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}
CHARGE = {'R': 1.0, 'K': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}

_AA_HYDRO = np.array([HYDROPATHY[aa] for aa in AMINO_ACIDS])
_AA_CHARGE = np.array([CHARGE.get(aa, 0.0) for aa in AMINO_ACIDS])


def vectorized_features(sequences):
    """
    Vectorized 428-dim feature extraction.

    Produces EXACTLY the same vector layout as
    ``PeptideFeatureExtractor.extract_all_features``:
        [AAC (20, in AMINO_ACIDS order)
         DPC (400, row-major over (a, b) in AMINO_ACIDS order)
         physchem (8): mw, avg_hydropathy, hydropathy_range,
                        net_charge, pi_estimate, gravy,
                        hydrophobic_ratio, charged_ratio]
    """
    n = len(sequences)
    X = np.zeros((n, 428), dtype=np.float64)

    codes = np.empty((n,), dtype=object)
    for i, seq in enumerate(sequences):
        codes[i] = np.fromiter((AA_TO_IDX[c] for c in seq), dtype=np.int64)

    for i in range(n):
        c = codes[i]
        L = len(c)
        # AAC
        X[i, 0:20] = np.bincount(c, minlength=20) / L
        # DPC
        if L >= 2:
            di = c[:-1] * 20 + c[1:]
            X[i, 20:420] = np.bincount(di, minlength=400) / (L - 1)
        # physchem
        hydro = _AA_HYDRO[c]
        charge = _AA_CHARGE[c]
        mw = L * 110
        avg_hydro = hydro.mean()
        hydro_range = hydro.max() - hydro.min()
        net_charge = charge.sum()
        basic = int((c == AA_TO_IDX['R']).sum() + (c == AA_TO_IDX['K']).sum())
        acidic = int((c == AA_TO_IDX['D']).sum() + (c == AA_TO_IDX['E']).sum())
        pi_est = 7.0 + (basic - acidic) / L * 2
        gravy = hydro.sum() / L
        hydro_ratio = (hydro > 0).mean()
        charged_ratio = np.isin(c, list(AA_TO_IDX[a] for a in CHARGE)).mean()
        X[i, 420:428] = [mw, avg_hydro, hydro_range,
                         net_charge, pi_est, gravy,
                         hydro_ratio, charged_ratio]
    return X


def family_profiles(rng, n_families):
    """Per-family amino-acid composition profiles (Dirichlet)."""
    alpha = rng.uniform(0.8, 6.0, size=(n_families, N_AA))
    # explicit loop: some numpy versions reject 2-D alpha for dirichlet
    return np.array([rng.dirichlet(alpha[i]) for i in range(n_families)])


def generate_sequences(rng, n, profiles, min_len=10, max_len=30):
    """Random sequences sampled from family composition profiles."""
    fam = rng.integers(0, len(profiles), size=n)
    lengths = rng.integers(min_len, max_len + 1, size=n)
    aa_arr = np.array(list(AMINO_ACIDS))
    seqs = []
    for i in range(n):
        idx = rng.choice(N_AA, size=int(lengths[i]), p=profiles[fam[i]])
        seqs.append(''.join(aa_arr[idx]))
    return seqs, fam


# ---------------------------------------------------------------------------
# Latent ADMET label model (v3.0: 9 endpoints, partial labels)
# ---------------------------------------------------------------------------
def _latent_labels(X, rng):
    """
    Crude-but-plausible latent labels for all 9 endpoints, driven by the
    physchem block plus a small random per-endpoint AAC tilt.  Returns a
    (n, 9) float array in ``ENDPOINT_ORDER`` where a NaN cell means
    "this endpoint is not labelled for this row" (pepADMET partial labels).

    The 5 original ADME endpoints + toxicity_binary are fully labelled;
    toxicity_type is fully labelled (0 = non-toxic, mirroring pepADMET);
    neurotoxicity_type and HC50 are present only for a random subset so the
    partial-label mask is genuinely exercised.
    """
    n = X.shape[0]
    idx = {ep: i for i, ep in enumerate(ENDPOINT_ORDER)}

    length_norm = X[:, I_MW] / 110.0 / 25.0          # ~0.4 (10 aa) .. 1.2 (30 aa)
    hydro = X[:, I_HYDRO]
    netq = X[:, I_NETQ]
    gravy = X[:, I_GRAVY]
    hydro_ratio = X[:, I_HR]
    charged_ratio = X[:, I_CR]
    arom = X[:, AA_TO_IDX['F']] + X[:, AA_TO_IDX['W']] + X[:, AA_TO_IDX['Y']]

    # Small, fixed random per-endpoint AAC tilt (adds endpoint-specific
    # sequence dependence beyond the 8 physchem features)
    n_bin = len(BINARY_ENDPOINTS)
    aac_tilt = rng.normal(0.0, 0.15, size=(n_bin, 20))
    aac_term = X[:, :20] @ aac_tilt.T
    eps = lambda s: rng.normal(0.0, s, size=n)

    Y = np.full((n, len(ENDPOINT_ORDER)), np.nan, dtype=np.float64)

    # ---- 5 original ADME binary endpoints (kept from v2.0) ---------------
    gi = (-1.2 * (length_norm - 0.8) + 0.8 * (hydro - 0.2)
          - 0.5 * netq ** 2 - 0.4 * charged_ratio + aac_term[:, 0] + eps(1.0))
    caco2 = (-0.8 * (length_norm - 1.0) + 1.4 * hydro + 0.5 * gravy
             + aac_term[:, 1] + eps(1.0))
    bbb = (-1.5 * (length_norm - 0.7) + 1.2 * hydro
           - 1.0 * netq ** 2 - 0.5 * charged_ratio + aac_term[:, 2] + eps(1.0))
    ames = (-1.8 + 0.4 * np.abs(netq) + 0.3 * arom + aac_term[:, 3] + eps(0.8))
    herg = (1.4 * hydro + 0.7 * arom + 0.5 * np.maximum(netq, 0.0)
            - 0.6 * length_norm - 0.3 + aac_term[:, 4] + eps(1.0))
    Y[:, idx['GI_absorption']] = (gi > 0).astype(float)
    Y[:, idx['Caco2_permeability']] = (caco2 > 0).astype(float)
    Y[:, idx['BBB_penetration']] = (bbb > 0).astype(float)
    Y[:, idx['Ames_mutagenicity']] = (ames > 0).astype(float)
    Y[:, idx['hERG_inhibition']] = (herg > 0).astype(float)

    # ---- toxicity_binary (binary, 6th binary head) ------------------------
    s_tox = (1.0 * hydro + 0.6 * arom + 0.4 * np.maximum(netq, 0.0)
             - 0.6 * (length_norm - 0.8) - 0.5 + aac_term[:, 5] + eps(0.9))
    toxic = (s_tox > 0)
    Y[:, idx['toxicity_binary']] = toxic.astype(float)

    # ---- toxicity_type (6-class: 0 = non-toxic, 1..5 = toxicity subtypes) --
    # Coherent with toxicity_binary: non-toxic rows are class 0; toxic rows
    # pick a subtype via argmax over 5 class latent scores.
    t = np.stack([
        1.0 * arom + 0.3 * hydro + eps(1.0),
        1.0 * charged_ratio - 0.2 * hydro + eps(1.0),
        0.8 * gravy + 0.3 * netq + eps(1.0),
        0.9 * (length_norm - 0.8) + eps(1.0),
        0.7 * hydro + 0.5 * arom - 0.4 * charged_ratio + eps(1.0),
    ], axis=1)
    subtype = 1 + t.argmax(axis=1)
    Y[:, idx['toxicity_type']] = np.where(toxic, subtype, 0)

    # ---- neurotoxicity_type (4-class, partial) ----------------------------
    # Present only for toxic rows (a random 70% of them) — a genuinely
    # partial endpoint, as in pepADMET.
    neuro_present = toxic & (rng.random(n) < 0.70)
    nt = np.stack([
        1.0 * arom + eps(1.0),
        1.0 * charged_ratio + eps(1.0),
        0.8 * gravy + eps(1.0),
        0.7 * hydro + 0.3 * netq + eps(1.0),
    ], axis=1)
    nt_class = nt.argmax(axis=1)
    Y[neuro_present, idx['neurotoxicity_type']] = nt_class[neuro_present]

    # ---- HC50 (regression, partial) ---------------------------------------
    # Half-maximal cytotoxicity on a log-ish scale (pepADMET's sample values
    # span ~0.8..2.6).  More potent (hydrophobic/aromatic) -> lower HC50.
    # Present for a random 60% of rows.
    hc_present = rng.random(n) < 0.60
    hc_latent = 1.6 - 0.6 * hydro - 0.4 * arom + eps(0.25)
    Y[hc_present, idx['HC50']] = np.clip(hc_latent[hc_present], 0.5, 3.0)

    return Y


def _csv_label(v):
    """Write NaN as an empty cell so it round-trips as NaN on read."""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ''
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return v
    except AttributeError:
        return v


def main():
    ap = argparse.ArgumentParser(description='Generate the synthetic demo dataset (9 endpoints)')
    ap.add_argument('--n', type=int, default=15000, help='number of synthetic peptides (default 15000)')
    ap.add_argument('--seed', type=int, default=42, help='random seed (default 42)')
    ap.add_argument('--n-families', type=int, default=200,
                    help='number of sequence composition families (default 200)')
    ap.add_argument('--out', type=str, default='data/peptide_admet_demo.csv',
                    help='output CSV path')
    ap.add_argument('--merge', type=str, default=None,
                    help='optional external CSV (from ingest_external.py) to fold in; '
                         'its labels are kept as-is and NOT relabelled')
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print(f'Generating {args.n} synthetic peptides (seed={args.seed}) ...')

    profiles = family_profiles(rng, args.n_families)
    sequences, fam = generate_sequences(rng, args.n, profiles)

    print('Extracting 428-dim features (vectorized) ...')
    X = vectorized_features(sequences)

    print('Drawing labels from the latent ADMET model (9 endpoints) ...')
    Y = _latent_labels(X, rng)

    # ---- fold in external data (optional) --------------------------------
    merged_external = []
    if args.merge:
        import pandas as pd
        ext = pd.read_csv(args.merge)
        need = {'sequence', 'data_origin'} | set(ENDPOINT_ORDER)
        missing = need - set(ext.columns)
        if missing:
            raise SystemExit(f'{args.merge} missing columns: {sorted(missing)}')
        ext = ext.dropna(subset=['sequence'])
        ext['sequence'] = ext['sequence'].astype(str).str.strip().str.upper()
        # Guard: feature extraction (prepare_data.vectorized_features and the
        # training-time encoder) only knows the 20 standard amino acids.  A
        # non-alphabetic or X-containing sequence would crash training with a
        # KeyError, so drop such rows here with an explicit warning instead of
        # letting them surface deep inside the training loop.
        valid_mask = ext['sequence'].apply(
            lambda s: len(s) >= 3 and set(s) <= set(AMINO_ACIDS))
        n_dropped = int((~valid_mask).sum())
        if n_dropped:
            print(f'WARNING: dropping {n_dropped} external row(s) with '
                  f'sequences outside the 20 standard amino acids '
                  f'(e.g. SMILES-inferred X residues); they cannot be '
                  f'encoded by the 428-dim feature extractor.')
        ext = ext[valid_mask]
        merged_external = ext.to_dict('records')
        print(f'Folding in {len(ext)} external rows from {args.merge} '
              f'(data_origin preserved, labels NOT relabelled)')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # dedup by sequence across synthetic + external (synthetic wins ties)
    seen = set()
    rows_written = 0
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sequence', 'family_id', 'data_origin', *ENDPOINT_ORDER])
        for i, seq in enumerate(sequences):
            if seq in seen:
                continue
            seen.add(seq)
            w.writerow([seq, int(fam[i]), 'synthetic_demo',
                        *[_csv_label(v) for v in Y[i].tolist()]])
            rows_written += 1
        for erow in merged_external:
            seq = str(erow['sequence']).strip().upper()
            if seq in seen or len(seq) < 3:
                continue
            seen.add(seq)
            fam_id = int(erow.get('family_id', -1)) if 'family_id' in erow else -1
            labels = [_csv_label(erow[ep]) for ep in ENDPOINT_ORDER]
            w.writerow([seq, fam_id, str(erow['data_origin']), *labels])
            rows_written += 1

    # ---- metadata ---------------------------------------------------------
    meta = {
        'n_samples_total': rows_written,
        'n_synthetic': int(args.n),
        'n_external': int(len(merged_external)),
        'seed': args.seed,
        'n_families': args.n_families,
        'feature_dim': 428,
        'feature_layout': '20 AAC + 400 DPC + 8 physchem (same as PeptideFeatureExtractor)',
        'endpoints': ENDPOINT_ORDER,
        'endpoint_kinds': {ep: ENDPOINT_BY_NAME[ep].kind for ep in ENDPOINT_ORDER},
        'label_model': ('latent physicochemical scores; binary = score>0; '
                        'multiclass = argmax of class latent scores '
                        '(toxicity_type: 0 = non-toxic); regression = latent HC50. '
                        'neurotoxicity_type & HC50 are PARTIAL (NaN = unlabelled).'),
        'data_origin': 'synthetic_demo (NOT experimental data); external rows keep their own origin',
        'partial_labels': 'NaN cell = endpoint not measured for that row (pepADMET convention)',
    }
    meta_path = out.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # ---- summary ----------------------------------------------------------
    print(f'\nWrote {out} ({rows_written} rows) in {time.time() - t0:.1f}s')
    print(f'Wrote {meta_path}')
    import pandas as pd
    chk = pd.read_csv(out)
    print('\nPer-endpoint label coverage (non-NaN / total):')
    for ep in ENDPOINT_ORDER:
        kind = ENDPOINT_BY_NAME[ep].kind
        cov = int(chk[ep].notna().sum())
        extra = ''
        if kind == 'binary':
            extra = f'  pos_rate={chk[ep].mean():.3f}'
        elif kind == 'multiclass':
            extra = f'  classes={sorted(chk[ep].dropna().unique().astype(int).tolist())}'
        print(f'  {ep:22} {cov:>7}/{len(chk)}  [{kind}]{extra}')


if __name__ == '__main__':
    main()
