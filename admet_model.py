#!/usr/bin/env python3
"""
admet_model.py
==============

Shared model definition for the peptide ADMET pipeline, so the trainer and
the predictor always agree on architecture and save format.

Architecture
------------
A single multilayer perceptron with one binary sigmoid head per ADMET
endpoint (5 heads), trained with BCEWithLogitsLoss.  This is a real
neural network (PyTorch) — it replaces the previous "nn_model" that was,
as the old training script admitted, "a placeholder model" (a second
Random Forest saved under a misleading name).

Save format (a single file ``admet_mlp.pt``):
    {
      'state_dict': ...,
      'input_dim': 428,
      'num_endpoints': 5,
      'endpoints': [...],
      'pos_weights': [...],   # per-endpoint positive weights used in training
    }
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ENDPOINTS = [
    'GI_absorption',
    'Caco2_permeability',
    'BBB_penetration',
    'Ames_mutagenicity',
    'hERG_inhibition',
]


class ADMETMLP(nn.Module):
    """MLP with per-endpoint binary heads."""

    def __init__(self, input_dim: int = 428, hidden=(256, 128),
                 num_endpoints: int = 5, dropout: float = 0.25):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_endpoints)

    def forward(self, x):
        return self.head(self.trunk(x))


def save_admet_model(model: nn.Module, path: Path, input_dim: int = 428,
                     endpoints=None, pos_weights=None):
    """Save model weights + metadata in one torch file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_endpoints': len(endpoints or ENDPOINTS),
        'endpoints': list(endpoints or ENDPOINTS),
        'pos_weights': (pos_weights.detach().cpu().tolist()
                        if isinstance(pos_weights, torch.Tensor) else list(pos_weights or [])),
    }, path)
    return path


def load_admet_model(path: str) -> tuple:
    """
    Load a saved ADMETMLP in eval mode.
    Returns (model, meta) where meta has 'endpoints' and 'pos_weights'.
    """
    path = Path(path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    model = ADMETMLP(input_dim=blob['input_dim'],
                     num_endpoints=blob['num_endpoints'])
    model.load_state_dict(blob['state_dict'])
    model.eval()
    return model, blob


def predict_proba(model: nn.Module, X: np.ndarray,
                  batch_size: int = 1024) -> np.ndarray:
    """
    Sigmoid probabilities for every endpoint: shape (n_samples, n_endpoints).
    """
    X = np.asarray(X, dtype=np.float32)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            t = torch.from_numpy(X[i:i + batch_size])
            out.append(torch.sigmoid(model(t)).numpy())
    return np.concatenate(out, axis=0)


if __name__ == '__main__':
    m = ADMETMLP()
    print(f'ADMETMLP params: {sum(p.numel() for p in m.parameters()):,}')
    x = torch.randn(8, 428)
    print('forward ok:', m(x).shape)
