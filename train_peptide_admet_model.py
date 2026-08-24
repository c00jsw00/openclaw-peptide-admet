#!/usr/bin/env python3
"""
train_peptide_admet_model.py
============================

Train the **v3.0 mixed multi-task** peptide ADMET model and produce
*measured*, reproducible metrics under two splits:

  1. **homology** — the AMPBench-MT-style homology-controlled split
     (sequences similar by 3-mer Jaccard never cross the boundary), and
  2. **random**   — a plain random 70/10/20 split, reported alongside so the
     reader can see how much the homology control lowers the (inflated) numbers.

Every number written to ``peptide_admet_model/metrics.json`` is computed here
at train time.  Nothing is hardcoded.

v3.0 endpoint set (see ``endpoint_config.py``)
----------------------------------------------
  * 6  binary     (GI, Caco2, BBB, Ames, hERG, toxicity_binary)
  * 2  multiclass (toxicity_type: 6, neurotoxicity_type: 4)
  * 1  regression (HC50)

Partial labels: a NaN endpoint cell = "not labelled for this row" and is
*masked out* of both the loss and the metrics (pepADMET convention).  The
loss is a sum over the nine endpoints, each reduced only over its labelled
rows.

Usage
-----
    python prepare_data.py                 # 1) build the demo CSV (9 endpoints)
    python homology_split.py               # 2) build data/split*.{npy,json}
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
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from admet_model import (MixedADMETMLP, predict_mixed, save_mixed_model)
from endpoint_config import (ENDPOINTS, ENDPOINT_NAMES, ENDPOINT_BY_NAME,
                             KIND_BINARY, KIND_MULTICLASS, KIND_REGRESSION)
from prepare_data import vectorized_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(csv_path: str):
    """
    Load sequences + the 9-endpoint label matrix with per-endpoint masks.

    Returns
    -------
    seqs : list[str]
    X : (N, 428) float32
    labels : dict endpoint -> (N,) float/int array (NaN where unlabelled)
    masks :  dict endpoint -> (N,) bool array  (True = has a label)
    origin : str
    """
    df = pd.read_csv(csv_path)
    required = {'sequence'} | set(ENDPOINT_NAMES)
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f'{csv_path} is missing columns: {sorted(missing)}')
    seqs = df['sequence'].astype(str).tolist()
    origin = (df['data_origin'].iloc[0] if 'data_origin' in df.columns else 'unknown')
    print(f'Loaded {len(seqs)} sequences from {csv_path} (data_origin={origin})')

    X = vectorized_features(seqs).astype(np.float32)

    labels, masks = {}, {}
    for ep in ENDPOINT_NAMES:
        col = df[ep].to_numpy(dtype=np.float64)
        m = ~np.isnan(col)
        labels[ep] = col
        masks[ep] = m
    return seqs, X, labels, masks, origin


def random_split_indices(n, train_frac, val_frac, seed):
    idx = np.arange(n)
    a, b = train_test_split(idx, test_size=(1 - train_frac),
                            random_state=seed, shuffle=True)
    b, c = train_test_split(b, test_size=val_frac / (val_frac + (1 - train_frac)),
                            random_state=seed, shuffle=True)
    return a, b, c


# ---------------------------------------------------------------------------
# Loss (mask-aware, per endpoint)
# ---------------------------------------------------------------------------
def mixed_loss(out, labels, masks, pos_weights, device):
    """Sum over endpoints, each reduced over its labelled rows only."""
    total = 0.0
    n_ep_active = 0
    for ep in ENDPOINT_NAMES:
        m = masks[ep]
        if not m.any():
            continue
        y = torch.as_tensor(labels[ep], dtype=torch.float32, device=device)[m]
        if y.dim() > 1:
            y = y.squeeze(-1)
        kind = ENDPOINT_BY_NAME[ep].kind
        o = out[ep][m]
        if kind == KIND_BINARY:
            o = o.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                o, y, pos_weight=pos_weights[ep].to(device), reduction='none')
        elif kind == KIND_MULTICLASS:
            loss = F.cross_entropy(o, y.long(), reduction='none')
        else:  # regression
            o = o.squeeze(-1)
            loss = F.mse_loss(o, y, reduction='none')
        total = total + loss.sum() / m.sum().clamp(min=1)
        n_ep_active += 1
    return total / max(n_ep_active, 1)


def compute_pos_weights(labels, masks, tr_idx):
    """Per-binary-endpoint pos_weight from the (labelled) train rows."""
    n = len(labels[ENDPOINT_NAMES[0]])
    tr_bool = np.zeros(n, dtype=bool)
    tr_bool[tr_idx] = True
    pw = {}
    for ep in ENDPOINT_NAMES:
        if ENDPOINT_BY_NAME[ep].kind != KIND_BINARY:
            pw[ep] = None
            continue
        m = masks[ep] & tr_bool
        y = labels[ep][m]
        pos = float(y.mean()) if len(y) else 0.5
        pos = float(np.clip(pos, 1e-4, 1 - 1e-4))
        pw[ep] = torch.tensor((1 - pos) / pos, dtype=torch.float32).clamp(max=20.0)
    return pw


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_mixed(X_train, labels, masks, X_val, val_masks,
                pos_weights, tr, va,
                lr=1e-3, epochs=60, patience=8, batch_size=128, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nTraining MixedADMETMLP on {device} ...')

    model = MixedADMETMLP(input_dim=X_train.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  params: {n_params:,}')

    # train tensors: X_train is ALREADY the train subset; index labels/masks
    # by tr to line them up
    Xt = torch.from_numpy(X_train).to(device)
    yt = {ep: torch.from_numpy(labels[ep][tr].astype(np.float32)).to(device)
          for ep in ENDPOINT_NAMES}
    mt = {ep: torch.from_numpy(masks[ep][tr].astype(bool)) for ep in ENDPOINT_NAMES}

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=3)

    # val tensors: X_val is already the val subset
    Xv = torch.from_numpy(X_val).to(device)
    yv = {ep: torch.from_numpy(labels[ep][va].astype(np.float32)).to(device)
          for ep in ENDPOINT_NAMES}
    mv = {ep: torch.from_numpy(masks[ep][va].astype(bool)) for ep in ENDPOINT_NAMES}

    best_val, best_state, bad = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xt[b])
            # build batch-local labels/masks
            bl = {ep: yt[ep][b] for ep in ENDPOINT_NAMES}
            bm = {ep: mt[ep][b] for ep in ENDPOINT_NAMES}
            loss = mixed_loss(out, bl, bm, pos_weights, device)
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vloss = mixed_loss(model(Xv), yv, mv, pos_weights, device).item()
        sched.step(vloss)
        if vloss < best_val - 1e-4:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f'  early stop at epoch {epoch} (val mixed loss {best_val:.4f})')
                break
        if epoch % 5 == 0 or epoch == 1:
            print(f'  epoch {epoch:3d}  train loss {total / len(Xt):.4f}  '
                  f'val loss {vloss:.4f}')

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, device


# ---------------------------------------------------------------------------
# Metrics (per endpoint kind)
# ---------------------------------------------------------------------------
def evaluate(model, X, labels, masks, device, tag):
    """Per-endpoint metrics, reduced only over labelled rows."""
    pred = predict_mixed(model, X)  # binary->prob, multiclass->class id, reg->value
    rows = {}
    for ep in ENDPOINT_NAMES:
        m = masks[ep]
        n_lab = int(m.sum())
        kind = ENDPOINT_BY_NAME[ep].kind
        rec = {'kind': kind, 'n_labelled': n_lab, 'n_total': int(len(m))}
        if n_lab < 2:
            rows[ep] = rec
            continue
        y = labels[ep][m]
        if kind == KIND_BINARY:
            p = pred[ep][m]
            yp = (p >= 0.5).astype(int)
            try:
                auc = float(roc_auc_score(y, p))
            except ValueError:
                auc = float('nan')
            rec['auc'] = round(auc, 4)
            rec['mcc'] = round(float(matthews_corrcoef(y, yp)), 4)
            rec['accuracy'] = round(float(accuracy_score(y, yp)), 4)
            rec['pos_rate'] = round(float(y.mean()), 4)
        elif kind == KIND_MULTICLASS:
            yp = pred[ep][m].astype(int)
            rec['accuracy'] = round(float(accuracy_score(y.astype(int), yp)), 4)
            rec['macro_f1'] = round(float(
                f1_score(y.astype(int), yp, average='macro',
                         zero_division=0)), 4)
            rec['class_distribution'] = {
                int(k): int(v) for k, v in
                pd.Series(y.astype(int)).value_counts().sort_index().items()}
        else:  # regression
            yp = pred[ep][m]
            rec['r2'] = round(float(r2_score(y, yp)), 4)
            rec['rmse'] = round(float(np.sqrt(mean_squared_error(y, yp))), 4)
            rec['mae'] = round(float(np.mean(np.abs(y - yp))), 4)
            rec['y_range'] = [round(float(y.min()), 3), round(float(y.max()), 3)]
            rec['pred_range'] = [round(float(yp.min()), 3), round(float(yp.max()), 3)]
        rows[ep] = rec

    # headline: mean of the primary metric across endpoints (AUC for binary,
    # macro_f1 for multiclass, r2 for regression)
    prim = []
    for ep, r in rows.items():
        v = r.get('auc', r.get('macro_f1', r.get('r2', np.nan)))
        if isinstance(v, float) and not np.isnan(v):
            prim.append(v)
    headline = round(float(np.mean(prim)), 4) if prim else float('nan')
    print(f'\n{tag} (mean primary metric {headline:.4f}):')
    for ep, r in rows.items():
        k = r.get('kind')
        if k == KIND_BINARY:
            print(f'  {ep:20s} AUC={r.get("auc"):.4f}  MCC={r.get("mcc"):.4f}  '
                  f'Acc={r.get("accuracy"):.4f}  pos={r.get("pos_rate"):.3f} '
                  f'({r["n_labelled"]}/{r["n_total"]})')
        elif k == KIND_MULTICLASS:
            print(f'  {ep:20s} Acc={r.get("accuracy"):.4f}  macroF1={r.get("macro_f1"):.4f} '
                  f'({r["n_labelled"]}/{r["n_total"]})')
        else:
            print(f'  {ep:20s} R2={r.get("r2"):.4f}  RMSE={r.get("rmse"):.4f} '
                  f'({r["n_labelled"]}/{r["n_total"]})')
    return rows, headline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Train the v3.0 mixed peptide ADMET model')
    ap.add_argument('--csv', type=str, default='data/peptide_admet_demo.csv')
    ap.add_argument('--split-dir', type=str, default='data')
    ap.add_argument('--model-dir', type=str, default='peptide_admet_model')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    seqs, X, labels, masks, origin = load_data(args.csv)
    print('Extracted 428-dim features')

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    n = len(seqs)

    # ---------------- 1) homology-controlled split (primary) --------------
    fam = np.load(Path(args.split_dir, 'split_families.npy'))
    mask = np.load(Path(args.split_dir, 'split_mask.npy'))
    audit_path = Path(args.split_dir, 'split_audit.json')
    audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else {}

    tr, va, te = (np.where(mask == m)[0] for m in (0, 1, 2))
    tr, va, te = (a.astype(np.int64) for a in (tr, va, te))
    print(f'\n[homology split] train={len(tr)} val={len(va)} test={len(te)}')

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)

    pos_weights = compute_pos_weights(labels, masks, tr)
    model, device = train_mixed(
        Xs[tr], labels, masks, Xs[va], masks,
        pos_weights, tr, va, epochs=args.epochs, seed=args.seed)

    te_mask = {ep: masks[ep][te] for ep in ENDPOINT_NAMES}
    va_mask = {ep: masks[ep][va] for ep in ENDPOINT_NAMES}
    homo_metrics, homo_headline = evaluate(
        model, Xs[te], {ep: labels[ep][te] for ep in ENDPOINT_NAMES},
        te_mask, device, 'HOMOLOGY-CONTROLLED TEST')
    evaluate(model, Xs[va], {ep: labels[ep][va] for ep in ENDPOINT_NAMES},
             va_mask, device, 'HOMOLOGY VAL')

    # ---------------- 2) random split (comparison) -------------------------
    r_tr, r_va, r_te = random_split_indices(n, 0.7, 0.1, seed=args.seed)
    r_tr, r_va, r_te = (a.astype(np.int64) for a in (r_tr, r_va, r_te))
    r_scaler = StandardScaler().fit(X[r_tr])
    Xr = r_scaler.transform(X).astype(np.float32)
    print('\n[comparison] training same model on a plain RANDOM 70/10/20 split')
    r_pw = compute_pos_weights(labels, masks, r_tr)
    r_model, _ = train_mixed(Xr[r_tr], labels, masks, Xr[r_va], masks,
                             r_pw, r_tr, r_va, epochs=args.epochs, seed=args.seed)
    rand_metrics, rand_headline = evaluate(
        r_model, Xr[r_te], {ep: labels[ep][r_te] for ep in ENDPOINT_NAMES},
        {ep: masks[ep][r_te] for ep in ENDPOINT_NAMES}, device,
        'RANDOM-SPLIT TEST (comparison only)')

    # ---------------- save artifacts ---------------------------------------
    pw_save = {ep: float(pos_weights[ep]) for ep in pos_weights
               if pos_weights[ep] is not None}
    save_mixed_model(model, model_dir / 'admet_mlp.pt', pos_weights=pw_save)
    torch.save(scaler, model_dir / 'scaler.pt')
    print(f'\nSaved model + scaler to {model_dir}/')

    metrics = {
        'model': 'MixedADMETMLP (shared trunk + per-task binary/multiclass/regression heads)',
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'input_dim': int(X.shape[1]),
        'feature_layout': '20 AAC + 400 DPC + 8 physchem',
        'endpoints': [e.name for e in ENDPOINTS],
        'endpoint_kinds': {e.name: e.kind for e in ENDPOINTS},
        'training': {
            'epochs': args.epochs,
            'early_stopping': 'val mixed loss, patience 8',
            'binary_pos_weights': pw_save,
            'partial_labels': 'NaN cell = endpoint not labelled for that row (masked out)',
            'seed': args.seed,
        },
        'data': {
            'csv': args.csv,
            'n_samples': int(n),
            'data_origin': origin,
            'provenance': ('SYNTHETIC DEMO DATA (and/or external rows with their '
                           'own data_origin). Labels come from a latent '
                           'physicochemical model / external file; they are NOT '
                           'experimental measurements unless the external file '
                           'is a real dataset.'),
        },
        'splits': {
            'primary': {
                'name': 'homology-controlled',
                'method': audit.get('method', '3-mer Jaccard family-level split'),
                'counts': {'train': int(len(tr)), 'val': int(len(va)), 'test': int(len(te))},
                'audit': audit,
                'test': homo_metrics,
            },
            'comparison': {
                'name': 'random 70/10/20',
                'note': 'reported to show the homology-control delta; random-split numbers are inflated by near-duplicate leakage',
                'counts': {'train': int(len(r_tr)), 'val': int(len(r_va)), 'test': int(len(r_te))},
                'test': rand_metrics,
            },
        },
        'headline': {
            'primary_mean_metric': homo_headline,
            'comparison_mean_metric': rand_headline,
            'homology_control_delta': round(rand_headline - homo_headline, 4),
            'metric_note': 'mean over endpoints of AUC(binary)/macroF1(multiclass)/R2(regression)',
        },
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(model_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'\nWrote {model_dir / "metrics.json"}')
    print(f'\nHEADLINE  primary (homology) mean metric = {homo_headline:.4f}   |   '
          f'random split = {rand_headline:.4f}   |   '
          f'delta = {rand_headline - homo_headline:+.4f}')
    print('The homology delta is the price of honest evaluation — it is exactly '
          'the leakage AMPBench-MT (arXiv:2607.25518) warns about.')


if __name__ == '__main__':
    main()
