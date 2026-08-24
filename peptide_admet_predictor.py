#!/usr/bin/env python3
"""
Peptide ADMET Prediction Inference Tool (v3.0)
==============================================

Inference for the peptide ADMET predictor.  v3.0 supports the **mixed
multi-task** model with 9 endpoints (6 binary + 2 multiclass + 1 regression),
and remains backward-compatible with the v2.0 5-binary model.

**Honesty note** (2026-08 revision):
  * This tool ships with a model trained on the clearly-labelled *synthetic
    demo dataset* (and/or external rows) produced by ``prepare_data.py``.
    Performance numbers printed below are *measured* on a homology-controlled
    test split (see ``peptide_admet_model/metrics.json`` and
    ``data/split_audit.json``) — they are NOT fixed constants.
  * Because the core training data is synthetic, the numbers demonstrate the
    pipeline, not a validated predictor for real peptides.  Before any
    biological use, retrain on real measured data (folded in via
    ``ingest_external.py --merge``) and re-evaluate with a homology-controlled
    split (AMPBench-MT, arXiv:2607.25518).

**Endpoints (v3.0)**:
  binary     — GI absorption, Caco-2, BBB, Ames, hERG, toxicity_binary
  multiclass — toxicity_type (6), neurotoxicity_type (4)
  regression — HC50 (half-maximal cytotoxicity, ~log scale; lower = more potent)

**Composite score**: geometric mean of each composite endpoint's *favourability*
in [0,1] (AMPGAN v3 / PepCraft-style joint ADMET ranking, arXiv:2606.17127), so
a candidate must be acceptable on all endpoints to rank well.

**Usage**:
    python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"
    python peptide_admet_predictor.py --sequences candidates.txt --rank
    python peptide_admet_predictor.py --interactive

**Author**: OpenClaw Team
**Date**: 2026-08-25 (v3.0 mixed 9-endpoint + external-data extensibility)
"""

import argparse
import sys
import json
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings('ignore')

from endpoint_config import (ENDPOINTS, ENDPOINT_NAMES, ENDPOINT_BY_NAME,
                             KIND_BINARY, KIND_MULTICLASS, KIND_REGRESSION,
                             COMPOSITE_NAMES)


# ============ Feature Extraction ============

class PeptideFeatureExtractor:
    """肽類特徵提取器 (must match training order: AAC 20 + DPC 400 + PhysChem 8)."""

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    def __init__(self):
        self.hydropathy = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }
        self.charge = {'R': 1.0, 'K': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0}

    def validate_sequence(self, sequence: str) -> bool:
        seq = sequence.upper().strip()
        if len(seq) == 0:
            return False
        return all(aa in self.AMINO_ACIDS for aa in seq)

    def amino_acid_composition(self, sequence: str) -> np.ndarray:
        from collections import Counter
        aa_counts = Counter(sequence.upper())
        total = len(sequence)
        if total == 0:
            return np.zeros(20)
        return np.array([aa_counts.get(aa, 0) / total for aa in self.AMINO_ACIDS])

    def dipeptide_composition(self, sequence: str) -> np.ndarray:
        from collections import Counter
        if len(sequence) < 2:
            return np.zeros(400)
        dpc_counts = Counter([sequence[i:i+2].upper() for i in range(len(sequence)-1)])
        total = len(sequence) - 1
        all_dipeptides = [a+b for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        return np.array([dpc_counts.get(dpp, 0) / total for dpp in all_dipeptides])

    def physicochemical_features(self, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        if len(seq) == 0:
            return np.zeros(8)
        mw = len(seq) * 110
        hydropathy_values = [self.hydropathy.get(aa, 0) for aa in seq]
        avg_hydropathy = np.mean(hydropathy_values)
        hydropathy_range = max(hydropathy_values) - min(hydropathy_values)
        net_charge = sum(self.charge.get(aa, 0) for aa in seq)
        basic_residues = sum(1 for aa in seq if aa in 'RK')
        acidic_residues = sum(1 for aa in seq if aa in 'DE')
        pi_estimate = 7.0 + (basic_residues - acidic_residues) / len(seq) * 2
        gravy = sum(hydropathy_values) / len(seq)
        hydrophobic_ratio = sum(1 for hv in hydropathy_values if hv > 0) / len(seq)
        charged_ratio = sum(1 for aa in seq if aa in self.charge) / len(seq)
        return np.array([mw, avg_hydropathy, hydropathy_range,
                         net_charge, pi_estimate, gravy,
                         hydrophobic_ratio, charged_ratio])

    def extract_all_features(self, sequence: str) -> np.ndarray:
        aac = self.amino_acid_composition(sequence)
        dpc = self.dipeptide_composition(sequence)
        physchem = self.physicochemical_features(sequence)
        return np.concatenate([aac, dpc, physchem])


# ============ Model loading + prediction ============

class PeptideAdmetPredictor:
    """
    Loads the trained model (v2.0 ADMETMLP or v3.0 MixedADMETMLP) + scaler +
    measured metrics.json from a model directory.
    """

    def __init__(self, model_dir: str = 'peptide_admet_model'):
        import torch
        self._torch = torch
        self.feature_extractor = PeptideFeatureExtractor()
        self.model_dir = Path(model_dir)

        model_path = self.model_dir / 'admet_mlp.pt'
        scaler_path = self.model_dir / 'scaler.pt'
        self.metrics_path = self.model_dir / 'metrics.json'

        if not model_path.exists():
            raise FileNotFoundError(
                f'Model file not found: {model_path}\n'
                f'Run: python prepare_data.py && python homology_split.py '
                f' && python train_peptide_admet_model.py')
        if not scaler_path.exists():
            raise FileNotFoundError(f'Scaler file not found: {scaler_path}')

        from admet_model import load_admet_model
        self.model, self.model_meta = load_admet_model(str(model_path))
        self.model_class = self.model_meta.get('model_class', 'ADMETMLP')
        self.is_mixed = (self.model_class == 'MixedADMETMLP')
        self.device = 'cpu'

        self.scaler = torch.load(scaler_path, map_location='cpu', weights_only=False)

        self.metrics = {}
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)

        # endpoint list: from the model (authoritative) 
        if self.is_mixed:
            self.endpoints = list(self.model.endpoints)
        else:
            self.endpoints = list(self.model_meta.get('endpoints', ENDPOINT_NAMES[:5]))

        # regression normalisation range (for composite) from metrics if present
        self.hc50_range = self._hc50_range()

        n_params = sum(p.numel() for p in self.model.parameters())
        origin = self.metrics.get('data', {}).get('data_origin', '?')
        print(f"✅ Model loaded from {self.model_dir} "
              f"({n_params:,} params, {self.model_class}, data: {origin})")

    def _hc50_range(self):
        """(lo, hi) HC50 label range from metrics, else a documented default."""
        try:
            test = self.metrics['splits']['primary']['test']
            yr = test.get('HC50', {}).get('y_range')
            if yr and len(yr) == 2:
                return (float(yr[0]), float(yr[1]))
        except (KeyError, TypeError):
            pass
        return (0.5, 3.0)

    # ------------------------------------------------------------------
    def _forward(self, sequences):
        """
        Return (preds, proball):
          preds   : dict endpoint -> (n,) array  (prob/class id/value)
          proball : dict endpoint -> (n,) or (n, C) probability
        """
        X = np.stack([self.feature_extractor.extract_all_features(s) for s in sequences])
        Xs = self.scaler.transform(X)
        xt = self._torch.from_numpy(Xs.astype(np.float32))
        with self._torch.no_grad():
            out = self.model(xt)
        preds, proball = {}, {}
        t = self._torch
        if self.is_mixed:
            for e in self.endpoints:
                kind = ENDPOINT_BY_NAME[e].kind
                o = out[e]
                if kind == KIND_BINARY:
                    p = t.sigmoid(o).squeeze(-1).numpy()
                    preds[e] = p
                    proball[e] = p
                elif kind == KIND_MULTICLASS:
                    probs = t.softmax(o, dim=1).numpy()
                    preds[e] = probs.argmax(axis=1)
                    proball[e] = probs
                else:
                    preds[e] = o.squeeze(-1).numpy()
                    proball[e] = o.squeeze(-1).numpy()
        else:
            # v2.0: single (n, k) sigmoid
            probs = t.sigmoid(out).numpy()
            for i, e in enumerate(self.endpoints):
                preds[e] = probs[:, i]
                proball[e] = probs[:, i]
        return preds, proball

    def predict(self, sequence: str) -> dict:
        if not self.feature_extractor.validate_sequence(sequence):
            raise ValueError(f"Invalid peptide sequence: {sequence}. Use only "
                             f"standard amino acids (ACDEFGHIKLMNPQRSTVWY).")
        preds, proball = self._forward([sequence])
        results = []
        for e in self.endpoints:
            kind = ENDPOINT_BY_NAME[e].kind
            rec = {'endpoint': e, 'kind': kind}
            if kind == KIND_BINARY:
                p = float(preds[e][0])
                rec['probability'] = round(p, 4)
                rec['prediction'] = int(p >= 0.5)
                rec['interpretation'] = self._interpret(e, rec['prediction'])
                rec['risk_level'] = self._risk(e, p)
            elif kind == KIND_MULTICLASS:
                cid = int(preds[e][0])
                labels = ENDPOINT_BY_NAME[e].pep_class_labels or {}
                rec['predicted_class'] = cid
                rec['predicted_label'] = labels.get(cid, f'class_{cid}')
                rec['class_probabilities'] = {
                    int(c): round(float(v), 4)
                    for c, v in zip(range(len(proball[e][0])), proball[e][0])}
                rec['confidence'] = round(float(proball[e][0][cid]), 4)
                rec['interpretation'] = f"Class {cid}: {rec['predicted_label']}"
                rec['risk_level'] = self._risk_multiclass(e, cid, proball[e][0])
            else:  # regression
                v = float(preds[e][0])
                rec['predicted_value'] = round(v, 4)
                lo, hi = self.hc50_range
                # lower HC50 = more potent (more toxic); higher is safer here
                frac = float(np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1))
                rec['interpretation'] = (f"HC50 ≈ {v:.2f} "
                                         f"(scale {lo:.1f}–{hi:.1f}; lower = more potent)")
                rec['risk_level'] = ('❌ 高毒性 (High potency)' if frac < 0.33
                                     else '⚠️ 中等 (Moderate)' if frac < 0.66
                                     else '✅ 低 (Low potency)')
            results.append(rec)

        comp = self.composite_score(preds, proball)
        return {'results': results,
                'composite_score': round(comp, 4),
                'endpoints': {e: (float(preds[e][0]) if preds[e].ndim else
                                  int(preds[e][0])) for e in self.endpoints}}

    # ---- composite -----------------------------------------------------
    def composite_score(self, preds, proball) -> float:
        """
        Geometric mean of each composite endpoint's favourability in [0,1].
        - binary higher_is_worse=False: favourable = p(positive)
        - binary higher_is_worse=True : favourable = 1 - p
        - multiclass: favourable = P(best class), best = class 0 (non-toxic)
        - regression (HC50): favourable = normalised value (higher HC50 = safer)
        """
        fav = []
        for e in self.endpoints:
            cfg = ENDPOINT_BY_NAME[e]
            if not cfg.in_composite:
                continue
            kind = cfg.kind
            if kind == KIND_BINARY:
                p = min(max(float(preds[e][0]), 1e-6), 1 - 1e-6)
                fav.append(p if not cfg.higher_is_worse else 1.0 - p)
            elif kind == KIND_MULTICLASS:
                # class 0 is the non-toxic / non-neurotoxic class
                fav.append(min(max(float(proball[e][0][0]), 1e-6), 1.0 - 1e-6))
            else:  # regression
                lo, hi = self.hc50_range
                frac = float(np.clip((float(preds[e][0]) - lo) / max(hi - lo, 1e-9),
                                     1e-6, 1.0 - 1e-6))
                fav.append(frac)
        if not fav:
            return 0.0
        return float(np.exp(np.mean(np.log(np.array(fav)))))

    # ---- text helpers --------------------------------------------------
    def _interpret(self, endpoint, prediction):
        interp = {
            'GI_absorption': {0: '低腸胃吸收 (Poor GI absorption)',
                              1: '高腸胃吸收 (Good GI absorption)'},
            'Caco2_permeability': {0: '低腸道穿透性 (Poor Caco-2)',
                                   1: '高腸道穿透性 (Good Caco-2)'},
            'BBB_penetration': {0: '無法穿透血腦屏障 (Poor BBB)',
                                1: '可穿透血腦屏障 (Good BBB)'},
            'Ames_mutagenicity': {0: '安全（非致突變）(Non-mutagenic)',
                                  1: '潛在致突變風險 (Mutagenicity risk)'},
            'hERG_inhibition': {0: '安全（低心毒性）(Low hERG risk)',
                                1: '潛在心毒性風險 (hERG risk)'},
            'toxicity_binary': {0: '非細胞毒性 (Non-toxic)',
                                1: '有細胞毒性 (Cytotoxic)'},
        }
        return interp.get(endpoint, {}).get(prediction, 'Unknown')

    def _risk(self, endpoint, probability):
        if ENDPOINT_BY_NAME[endpoint].higher_is_worse:
            if probability < 0.3: return '✅ 低風險 (Low Risk)'
            if probability < 0.5: return '⚠️ 中等風險 (Moderate)'
            return '❌ 高風險 (High Risk)'
        else:
            if probability > 0.7: return '✅ 優秀 (Excellent)'
            if probability > 0.5: return '⚠️ 良好 (Good)'
            return '⚠️ 需優化 (Needs Optimization)'

    def _risk_multiclass(self, endpoint, cid, probs):
        # class 0 = non-toxic / non-neurotoxic is the safe class
        safe = float(probs[0]) if len(probs) > 0 else 0.0
        if cid == 0:
            return f'✅ 無毒性/無神經毒性 (P={safe:.2f})'
        return f'❌ 毒性類型 {cid} (P={float(probs[cid]):.2f})'

    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        m = self.metrics
        if not m:
            return {'model_type': self.model_class,
                    'note': 'metrics.json not found — performance not stated. '
                            'Run train_peptide_admet_model.py for measured metrics.'}
        primary = m.get('splits', {}).get('primary', {})
        h = m.get('headline', {})
        return {
            'model_type': m.get('model', self.model_class),
            'trained_on': m.get('data', {}).get('data_origin'),
            'train_samples': primary.get('counts', {}).get('train'),
            'eval_split': 'homology-controlled (AMPBench-MT-style, arXiv:2607.25518)',
            'mean_metric_homology': h.get('primary_mean_metric'),
            'per_endpoint_homology': primary.get('test', {}),
            'disclaimer': ('Training data is the synthetic demo set / external rows — '
                           'numbers validate the pipeline, not real-peptide performance.')
        }


# ============ Output Formatting ============

def _status_icon(endpoint):
    if 'ames' in endpoint.lower(): return '🧬'
    if 'herg' in endpoint.lower() or 'toxic' in endpoint.lower(): return '❤️'
    if 'neuro' in endpoint.lower(): return '🧠'
    if 'hc50' in endpoint.lower(): return '☣️'
    return '📊'


def print_prediction_result(sequence, out, predictor):
    results = out['results']
    print("\n" + "="*70)
    print("Peptide ADMET Prediction Results (v3.0, 9 endpoints)")
    print("="*70)
    print(f"\nSequence: {sequence}")
    print(f"Length: {len(sequence)} amino acids")
    print(f"Features: 428 (AAC 20 + DPC 400 + PhysChem 8)\n" + "-"*70)

    for r in results:
        e = r['endpoint']
        print(f"\n{_status_icon(e)} {e}  [{r['kind']}]")
        if r['kind'] == KIND_BINARY:
            bar = '█' * int(30 * r['probability']) + '░' * (30 - int(30 * r['probability']))
            print(f"   Probability: {r['probability']:.4f}  [{bar}]")
            print(f"   Prediction: {r['interpretation']}")
            print(f"   Risk: {r['risk_level']}")
        elif r['kind'] == KIND_MULTICLASS:
            print(f"   Predicted: {r['interpretation']}  (confidence {r['confidence']:.3f})")
            probs = r['class_probabilities']
            top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
            print(f"   Top-3 classes: " +
                  ", ".join(f"c{c}:{p:.2f}" for c, p in top))
            print(f"   Risk: {r['risk_level']}")
        else:
            print(f"   {r['interpretation']}")
            print(f"   Risk: {r['risk_level']}")

    print("\n" + "-"*70)
    print(f"Composite multi-objective score: {out['composite_score']:.4f}  "
          f"(geometric mean of favourability across composite endpoints)")
    info = predictor.model_info()
    if 'mean_metric_homology' in info and info['mean_metric_homology'] is not None:
        print(f"Measured on {info['eval_split']} split "
              f"({info.get('trained_on')} data, {info.get('train_samples')} train): "
              f"mean metric = {info['mean_metric_homology']:.4f}")
        print("NOTE: " + info['disclaimer'])
    else:
        print(info.get('note', 'No measured metrics available.'))
    print("="*70)


def print_ranked_results(results_list):
    ranked = sorted(results_list, key=lambda r: r[1]['composite_score'], reverse=True)
    print("\n" + "="*70)
    print("Candidates ranked by composite multi-objective ADMET score")
    print("="*70)
    # compact column header: score + a few key endpoints
    cols = ['GI', 'Caco2', 'BBB', 'Ames', 'hERG', 'tox', 'toxType', 'HC50']
    keymap = {'GI': 'GI_absorption', 'Caco2': 'Caco2_permeability',
              'BBB': 'BBB_penetration', 'Ames': 'Ames_mutagenicity',
              'hERG': 'hERG_inhibition', 'tox': 'toxicity_binary',
              'toxType': 'toxicity_type', 'HC50': 'HC50'}
    header = f"{'#':<3} {'score':<7} {'sequence':<24}" + "".join(f"{c:>8}" for c in cols)
    print(header)
    print("-"*70)
    for i, (seq, out) in enumerate(ranked, 1):
        ep = out['endpoints']
        cells = []
        for c in cols:
            e = keymap[c]
            if e not in ep:
                cells.append(f"{'-':>8}")
            elif c == 'toxType':
                cells.append(f"{int(ep[e]):>8d}")
            elif c == 'HC50':
                cells.append(f"{ep[e]:>8.2f}")
            else:
                cells.append(f"{ep[e]:>8.3f}")
        print(f"{i:<3} {out['composite_score']:<7.4f} {seq[:22]:<24}" + "".join(cells))
    print("="*70)


def print_batch_results(results_list):
    print("\n" + "="*70)
    print(f"Peptide ADMET Batch Prediction Results ({len(results_list)} sequences)")
    print("="*70)
    for i, (sequence, out) in enumerate(results_list, 1):
        print(f"[{i}/{len(results_list)}] {sequence}  (len {len(sequence)})")
        for r in out['results']:
            if r['kind'] == KIND_BINARY:
                print(f"   {r['endpoint']:22s} p={r['probability']:.3f}  {r['risk_level']}")
            elif r['kind'] == KIND_MULTICLASS:
                print(f"   {r['endpoint']:22s} class={r['predicted_class']} "
                      f"({r['predicted_label']})  conf={r['confidence']:.2f}")
            else:
                print(f"   {r['endpoint']:22s} value={r['predicted_value']:.3f}")
        print(f"   composite = {out['composite_score']:.4f}\n")
    print("="*70)


# ============ Interactive ============

def interactive_mode(predictor):
    print("\n" + "="*70)
    print("Peptide ADMET Prediction - Interactive Mode (v3.0)")
    print("="*70)
    print("Enter peptide sequences to predict. 'quit' to exit.\n")
    while True:
        try:
            sequence = input("\nEnter peptide sequence: ").strip()
            if sequence.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            if not sequence:
                print("⚠️  Please enter a valid sequence.")
                continue
            out = predictor.predict(sequence)
            print_prediction_result(sequence, out, predictor)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(
        description='Peptide ADMET Prediction Tool (v3.0, 9 endpoints)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"
  python peptide_admet_predictor.py --sequences candidates.txt --rank
  python peptide_admet_predictor.py --interactive
  python peptide_admet_predictor.py --sequence "ACDE" --output results.json
        """)
    parser.add_argument('--sequence', '-s', type=str, help='Single peptide sequence')
    parser.add_argument('--sequences', '-f', type=str, help='File with sequences (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    parser.add_argument('--model-dir', '-m', type=str, default='peptide_admet_model')
    parser.add_argument('--rank', action='store_true',
                        help='With --sequences: rank by composite score')
    args = parser.parse_args()

    if not (args.sequence or args.sequences or args.interactive):
        parser.print_help()
        print("\n❌ Please specify a sequence, file, or use interactive mode.")
        sys.exit(1)

    try:
        predictor = PeptideAdmetPredictor(model_dir=args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        sys.exit(1)

    if args.interactive:
        interactive_mode(predictor)

    elif args.sequence:
        try:
            out = predictor.predict(args.sequence)
            print_prediction_result(args.sequence, out, predictor)
            if args.output:
                output_data = {
                    'sequence': args.sequence,
                    'length': len(args.sequence),
                    'composite_score': out['composite_score'],
                    'predictions': out['results'],
                    'endpoints': out['endpoints'],
                    'model_info': predictor.model_info()
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Results saved to {args.output}")
        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif args.sequences:
        try:
            seq_file = Path(args.sequences)
            if not seq_file.exists():
                print(f"❌ File not found: {args.sequences}")
                sys.exit(1)
            with open(seq_file, 'r', encoding='utf-8') as f:
                sequences = [line.strip() for line in f if line.strip()]
            results_list = []
            for seq in sequences:
                try:
                    out = predictor.predict(seq)
                    results_list.append((seq, out))
                except ValueError as e:
                    print(f"⚠️  Skipping invalid sequence '{seq[:20]}...': {e}")
            if args.rank:
                print_ranked_results(results_list)
            else:
                print_batch_results(results_list)
            if args.output:
                output_data = {
                    'total_sequences': len(results_list),
                    'successful_predictions': len(results_list),
                    'ranked_by_composite_score': args.rank,
                    'predictions': [
                        {'sequence': seq, 'length': len(seq),
                         'composite_score': out['composite_score'],
                         'endpoints': out['endpoints'],
                         'predictions': out['results']}
                        for seq, out in results_list
                    ],
                    'model_info': predictor.model_info()
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Results saved to {args.output}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
