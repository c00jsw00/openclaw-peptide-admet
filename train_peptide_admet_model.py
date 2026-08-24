#!/usr/bin/env python3
"""
train_peptide_admet_model.py
============================

Train the peptide ADMET predictor (real PyTorch MLP, per-endpoint heads)
and produce *measured*, reproducible metrics under two splits:

  1. **homology** — the AMPBench-MT-style homology-controlled split
     (from homology_split.py; sequences similar by 3-mer Jaccard never
     cross the boundary), and
  2. **random**   — a plain stratified random 70/10/20 split, reported
     alongside so the reader can see how much the homology control
     lowers the (inflated) numbers.

Every number written to ``peptide_admet_model/metrics.json`` is computed
here at train time.  Nothing is hardcoded.  The JSON also records the
data provenance (synthetic demo data) so downstream reports cannot
silently present it as experimental evidence.

Usage
-----
    python prepare_data.py          # 1) build data/peptide_admet_demo.csv
    python homology_split.py        # 2) build data/split*.{npy,json}
    python train_peptide_admet_model.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from admet_model import ADMETMLP, ENDPOINTS, predict_proba, save_admet_model
from prepare_data import vectorized_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sequences(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {'sequence', *ENDPOINTS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'{csv_path} is missing columns: {sorted(missing)}')
    seqs = df['sequence'].astype(str).tolist()
    y = df[ENDPOINTS].to_numpy(dtype=np.float32)
    origin = (df['data_origin'].iloc[0]
              if 'data_origin' in df.columns else 'unknown')
    print(f'Loaded {len(seqs)} sequences from {csv_path} '
          f'(data_origin={origin})')
    return seqs, y, origin


def random_split_indices(n, train_frac, val_frac, seed):
    """Plain stratified random split (reported for comparison only)."""
    idx = np.arange(n)
    a, b = train_test_split(idx, test_size=(1 - train_frac),
                            random_state=seed, shuffle=True)
    b, c = train_test_split(b, test_size=val_frac / (val_frac + (1 - train_frac)),
                            random_state=seed, shuffle=True)
    return a, b, c


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mlp(X_train, y_train, X_val, y_val,
              lr=1e-3, epochs=60, patience=8, batch_size=128, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining ADMETMLP on {device} ...')

    model = ADMETMLP(input_dim=X_train.shape[1]).to(device)

    pos = torch.tensor(y_train.mean(axis=0), dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    pos_weights = (1 - pos) / pos
    pos_weights = pos_weights.clamp(max=20.0).to(device)
    print(f'per-endpoint positive rates: {pos.numpy().round(3).tolist()}')

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3)

    Xt = torch.from_numpy(X_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv), yv).item()
        sched.step(vloss)
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in
                          model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f'  early stop at epoch {epoch} (val bce {best_val:.4f})')
                break
        if epoch % 5 == 0 or epoch == 1:
            print(f'  epoch {epoch:3d}  train bce {total / len(Xt):.4f}  '
                  f'val bce {vloss:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, pos_weights.cpu(), device


def evaluate(model, X, y, device, tag):
    """Per-endpoint AUC / MCC / accuracy on a split."""
    proba = predict_proba(model, X)
    pred = (proba >= 0.5).astype(int)
    rows = {}
    for j, ep in enumerate(ENDPOINTS):
        try:
            auc = float(roc_auc_score(y[:, j], proba[:, j]))
        except ValueError:
            auc = float('nan')
        rows[ep] = {
            'auc': round(auc, 4),
            'mcc': round(float(matthews_corrcoef(y[:, j], pred[:, j])), 4),
            'accuracy': round(float(accuracy_score(y[:, j], pred[:, j])), 4),
            'n': int(len(y)),
            'pos_rate': round(float(y[:, j].mean()), 4),
        }
    macro_auc = float(np.nanmean([r['auc'] for r in rows.values()]))
    print(f'\n{tag} (macro AUC {macro_auc:.4f}):')
    for ep, r in rows.items():
        print(f'  {ep:20s} AUC={r["auc"]:.4f}  MCC={r["mcc"]:.4f}  '
              f'Acc={r["accuracy"]:.4f}  pos={r["pos_rate"]:.3f}')
    rows['MACRO'] = {'auc': round(macro_auc, 4)}
    return rows, proba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Train the peptide ADMET MLP')
    ap.add_argument('--csv', type=str, default='data/peptide_admet_demo.csv')
    ap.add_argument('--split-dir', type=str, default='data')
    ap.add_argument('--model-dir', type=str, default='peptide_admet_model')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    seqs, y, origin = load_sequences(args.csv)
    print('Extracting 428-dim features ...')
    X = vectorized_features(seqs).astype(np.float32)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- 1) homology-controlled split (primary) --------------
    split = Path(args.split_dir, 'split')
    fam = np.load(Path(args.split_dir, 'split_families.npy'))
    mask = np.load(Path(args.split_dir, 'split_mask.npy'))
    audit_path = Path(args.split_dir, 'split_audit.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}

    tr, va, te = (np.where(mask == m)[0] for m in (0, 1, 2))
    print(f'\n[homology split] train={len(tr)} val={len(va)} test={len(te)}')

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)

    model, pos_weights, device = train_mlp(
        Xs[tr], y[tr], Xs[va], y[va],
        epochs=args.epochs, seed=args.seed)

    homo_metrics, _ = evaluate(model, Xs[te], y[te], device,
                               'HOMOLOGY-CONTROLLED TEST')
    homo_val, _ = evaluate(model, Xs[va], y[va], device, 'HOMOLOGY VAL')

    # ---------------- 2) random split (comparison) -------------------------
    r_tr, r_va, r_te = random_split_indices(len(seqs), 0.7, 0.1, seed=args.seed)
    r_scaler = StandardScaler().fit(X[r_tr])
    Xr = r_scaler.transform(X).astype(np.float32)
    print('\n[comparison] training same model on a plain RANDOM 70/10/20 split')
    r_model, _, _ = train_mlp(Xr[r_tr], y[r_tr], Xr[r_va], y[r_va],
                              epochs=args.epochs, seed=args.seed)
    rand_metrics, _ = evaluate(r_model, Xr[r_te], y[r_te], device,
                               'RANDOM-SPLIT TEST (comparison only)')

    # ---------------- save artifacts ---------------------------------------
    save_admet_model(model, model_dir / 'admet_mlp.pt',
                     input_dim=X.shape[1], endpoints=ENDPOINTS,
                     pos_weights=pos_weights)
    torch.save(scaler, model_dir / 'scaler.pt')
    print(f'\nSaved model + scaler to {model_dir}/')

    macro_h = homo_metrics['MACRO']['auc']
    macro_r = rand_metrics['MACRO']['auc']
    metrics = {
        'model': 'ADMETMLP (PyTorch MLP, per-endpoint binary heads)',
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'input_dim': int(X.shape[1]),
        'feature_layout': '20 AAC + 400 DPC + 8 physchem',
        'training': {
            'epochs': args.epochs,
            'early_stopping': 'val BCE, patience 8',
            'pos_weights': [round(float(w), 3) for w in pos_weights.numpy()],
            'seed': args.seed,
        },
        'data': {
            'csv': args.csv,
            'n_samples': int(len(seqs)),
            'data_origin': origin,
            'provenance': ('SYNTHETIC DEMO DATA. Labels come from a latent '
                           'physicochemical model in prepare_data.py; they '
                           'are NOT experimental measurements.'),
        },
        'splits': {
            'primary': {
                'name': 'homology-controlled',
                'method': audit.get('method',
                                    '3-mer Jaccard family-level split'),
                'counts': {'train': int(len(tr)), 'val': int(len(va)),
                           'test': int(len(te))},
                'audit': audit,
                'val': homo_val,
                'test': homo_metrics,
            },
            'comparison': {
                'name': 'random stratified 70/10/20',
                'note': 'reported to show the homology-control delta; '
                        'random-split numbers are inflated by near-duplicate leakage',
                'counts': {'train': int(len(r_tr)), 'val': int(len(r_va)),
                           'test': int(len(r_te))},
                'test': rand_metrics,
            },
        },
        'headline': {
            'primary_macro_auc': macro_h,
            'comparison_macro_auc': macro_r,
            'homology_control_delta': round(macro_r - macro_h, 4),
        },
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(model_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'\nWrote {model_dir / "metrics.json"}')
    print(f'\nHEADLINE  primary (homology) macro AUC = {macro_h:.4f}   |   '
          f'random split macro AUC = {macro_r:.4f}   |   '
          f'delta = {macro_r - macro_h:+.4f}')
    print('The homology delta is the price of honest evaluation — '
          'it is exactly the leakage AMPBench-MT (arXiv:2607.25518) warns about.')


if __name__ == '__main__':
    main()
