#!/usr/bin/env python3
"""
prepare_data.py
===============

Generate a **reproducible, clearly-labelled DEMO dataset** for the peptide
ADMET predictor.

Why this file exists
--------------------
The original repository claimed "15,000 real peptides" but shipped no data at
all, so the pipeline could not be run or reproduced.  This script generates
15,000 synthetic peptide sequences whose ADMET labels are drawn from a simple
*latent physicochemical model* (length, hydropathy, net charge, aromaticity).
The data is therefore:

  * fully reproducible (fixed seed),
  * guaranteed to carry learnable signal (labels depend on the 428-dim
    feature vector the model sees),
  * explicitly labelled ``data_origin=synthetic_demo`` in every row and in
    the accompanying metadata, so it can never be mistaken for experimental
    data.

The latent label model is deliberately crude: it mimics the well-known coarse
trends (small + neutral + moderately hydrophobic -> better permeability;
cationic amphiphilic -> hERG risk; etc.) so that honest, *measurable* AUCs
(~0.7-0.9) emerge instead of the old hardcoded 0.9987.

Usage
-----
    python prepare_data.py                     # 15,000 rows, seed 42
    python prepare_data.py --n 5000 --seed 7 --out data/peptide_admet_demo.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
N_AA = len(AMINO_ACIDS)
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

ENDPOINTS = [
    'GI_absorption',
    'Caco2_permeability',
    'BBB_penetration',
    'Ames_mutagenicity',
    'hERG_inhibition',
]

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
# Latent ADMET label model
# ---------------------------------------------------------------------------
def _latent_scores(X, rng):
    """
    Crude but plausible latent scores per endpoint, driven by the physchem
    block and a small random AAC contribution.  Labels = (score > 0).
    """
    n = X.shape[0]
    length_norm = X[:, I_MW] / 110.0 / 25.0          # ~0.4 (10 aa) .. 1.2 (30 aa)
    hydro = X[:, I_HYDRO]
    netq = X[:, I_NETQ]
    gravy = X[:, I_GRAVY]
    hydro_ratio = X[:, I_HR]
    charged_ratio = X[:, I_CR]

    # Aromatic fraction (F, W, Y) from the AAC block
    arom = X[:, AA_TO_IDX['F']] + X[:, AA_TO_IDX['W']] + X[:, AA_TO_IDX['Y']]

    # Small, fixed random per-endpoint AAC tilt (adds endpoint-specific
    # sequence dependence beyond the 8 physchem features)
    aac_tilt = rng.normal(0.0, 0.15, size=(5, 20))
    aac_term = X[:, :20] @ aac_tilt.T

    eps = lambda s: rng.normal(0.0, s, size=n)

    gi = (-1.2 * (length_norm - 0.8) + 0.8 * (hydro - 0.2)
          - 0.5 * netq ** 2 - 0.4 * charged_ratio + aac_term[:, 0] + eps(1.0))
    caco2 = (-0.8 * (length_norm - 1.0) + 1.4 * hydro + 0.5 * gravy
             + aac_term[:, 1] + eps(1.0))
    bbb = (-1.5 * (length_norm - 0.7) + 1.2 * hydro
           - 1.0 * netq ** 2 - 0.5 * charged_ratio + aac_term[:, 2] + eps(1.0))
    ames = (-1.8 + 0.4 * np.abs(netq) + 0.3 * arom + aac_term[:, 3] + eps(0.8))
    herg = (1.4 * hydro + 0.7 * arom + 0.5 * np.maximum(netq, 0.0)
            - 0.6 * length_norm - 0.3 + aac_term[:, 4] + eps(1.0))

    return np.stack([gi, caco2, bbb, ames, herg], axis=1)


def main():
    ap = argparse.ArgumentParser(description='Generate the synthetic demo dataset')
    ap.add_argument('--n', type=int, default=15000, help='number of peptides (default 15000)')
    ap.add_argument('--seed', type=int, default=42, help='random seed (default 42)')
    ap.add_argument('--n-families', type=int, default=200,
                    help='number of sequence composition families (default 200)')
    ap.add_argument('--out', type=str, default='data/peptide_admet_demo.csv',
                    help='output CSV path')
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print(f'Generating {args.n} synthetic peptides (seed={args.seed}) ...')

    profiles = family_profiles(rng, args.n_families)
    sequences, fam = generate_sequences(rng, args.n, profiles)

    print('Extracting 428-dim features (vectorized) ...')
    X = vectorized_features(sequences)

    print('Drawing labels from the latent ADMET model ...')
    scores = _latent_scores(X, rng)
    Y = (scores > 0).astype(int)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sequence', 'family_id', 'data_origin', *ENDPOINTS])
        for i, seq in enumerate(sequences):
            w.writerow([seq, int(fam[i]), 'synthetic_demo', *Y[i].tolist()])

    meta = {
        'n_samples': args.n,
        'seed': args.seed,
        'n_families': args.n_families,
        'feature_dim': 428,
        'feature_layout': '20 AAC + 400 DPC + 8 physchem (same as PeptideFeatureExtractor)',
        'endpoints': ENDPOINTS,
        'label_model': ('latent physicochemical linear scores + Gaussian noise; '
                        'labels = score > 0'),
        'data_origin': 'synthetic_demo (NOT experimental data)',
    }
    meta_path = out.with_suffix('.meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'\nWrote {out} ({args.n} rows) in {time.time() - t0:.1f}s')
    print(f'Wrote {meta_path}')
    print('\nEndpoint label rates:')
    for j, ep in enumerate(ENDPOINTS):
        print(f'  {ep:20s} positive rate = {Y[:, j].mean():.3f}')


if __name__ == '__main__':
    main()
