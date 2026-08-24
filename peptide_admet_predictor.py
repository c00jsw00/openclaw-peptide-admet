#!/usr/bin/env python3
"""
Peptide ADMET Prediction Inference Tool
=======================================

Inference for the peptide ADMET predictor.

**Honesty note** (2026-08 revision):
  * This tool ships with a model trained on the clearly-labelled
    *synthetic demo dataset* produced by ``prepare_data.py``.  The
    performance numbers printed below are *measured* on a
    homology-controlled test split (see ``peptide_admet_model/metrics.json``
    and ``data/split_audit.json``) — they are NOT fixed constants.
  * Because the training data is synthetic, the numbers demonstrate the
    pipeline, not a validated predictor for real peptides.  Before any
    biological use, retrain on real measured data and re-evaluate with a
    homology-controlled split (see AMPBench-MT, arXiv:2607.25518, for the
    evaluation methodology).

**Usage**:
    python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"
    python peptide_admet_predictor.py --sequences sequences.txt
    python peptide_admet_predictor.py --interactive
    python peptide_admet_predictor.py --sequences candidates.csv --rank

**Output**:
    Predicts 5 ADMET endpoints + one composite multi-objective score:
    1. GI Absorption (腸胃吸收)          — higher is better
    2. Caco-2 Permeability (腸道穿透)    — higher is better
    3. BBB Penetration (血腦屏障穿透)    — higher is better
    4. Ames Mutagenicity (致突變性)      — lower is better
    5. hERG Inhibition (心毒性)          — lower is better

    Composite score (AMPGAN v3 / PepCraft-style multi-objective ranking,
    arXiv:2606.17127): geometric mean of the "favourable" probability of
    each endpoint, so a candidate must be decent on ALL endpoints to rank
    well; toxicity probabilities enter as (1 - p).

**Author**: Pinwan (OpenClaw Team)
**Date**: 2026-03-24 (initial); 2026-08-24 (honest-metrics + composite score revision)
"""

import argparse
import sys
import json
from pathlib import Path
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# ============ Feature Extraction ============

class PeptideFeatureExtractor:
    """肽類特徵提取器 (must match training order: AAC 20 + DPC 400 + PhysChem 8)"""

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    def __init__(self):
        # Kyte-Doolittle hydropathy scale
        self.hydropathy = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }

        # Charge at pH 7.0
        self.charge = {
            'R': 1.0, 'K': 1.0, 'H': 0.1,  # Positive
            'D': -1.0, 'E': -1.0,  # Negative
        }

    def validate_sequence(self, sequence: str) -> bool:
        """Validate peptide sequence"""
        seq = sequence.upper().strip()
        if len(seq) == 0:
            return False
        if not all(aa in self.AMINO_ACIDS for aa in seq):
            return False
        return True

    def amino_acid_composition(self, sequence: str) -> np.ndarray:
        """Amino acid composition (AAC) - 20 features"""
        from collections import Counter
        aa_counts = Counter(sequence.upper())
        total = len(sequence)
        if total == 0:
            return np.zeros(20)
        return np.array([aa_counts.get(aa, 0) / total for aa in self.AMINO_ACIDS])

    def dipeptide_composition(self, sequence: str) -> np.ndarray:
        """Dipeptide composition (DPC) - 400 features"""
        from collections import Counter
        if len(sequence) < 2:
            return np.zeros(400)

        dpc_counts = Counter([sequence[i:i+2].upper() for i in range(len(sequence)-1)])
        total = len(sequence) - 1
        all_dipeptides = [a+b for a in self.AMINO_ACIDS for b in self.AMINO_ACIDS]
        return np.array([dpc_counts.get(dpp, 0) / total for dpp in all_dipeptides])

    def physicochemical_features(self, sequence: str) -> np.ndarray:
        """Physicochemical properties - 8 features"""
        seq = sequence.upper()
        if len(seq) == 0:
            return np.zeros(8)

        # Molecular weight (approximate: 110 Da per amino acid)
        mw = len(seq) * 110

        # Hydropathy values
        hydropathy_values = [self.hydropathy.get(aa, 0) for aa in seq]
        avg_hydropathy = np.mean(hydropathy_values)
        hydropathy_range = max(hydropathy_values) - min(hydropathy_values)

        # Net charge at pH 7.0
        net_charge = sum(self.charge.get(aa, 0) for aa in seq)

        # Estimated pI (simplified)
        basic_residues = sum(1 for aa in seq if aa in 'RK')
        acidic_residues = sum(1 for aa in seq if aa in 'DE')
        pi_estimate = 7.0 + (basic_residues - acidic_residues) / len(seq) * 2

        # Grand average of hydropathy (GRAVY)
        gravy = sum(hydropathy_values) / len(seq)

        # Hydrophobic ratio (hydropathy > 0)
        hydrophobic_ratio = sum(1 for hv in hydropathy_values if hv > 0) / len(seq)

        # Charged ratio
        charged_ratio = sum(1 for aa in seq if aa in self.charge) / len(seq)

        return np.array([
            mw, avg_hydropathy, hydropathy_range,
            net_charge, pi_estimate, gravy,
            hydrophobic_ratio, charged_ratio
        ])

    def extract_all_features(self, sequence: str) -> np.ndarray:
        """Extract all features from peptide sequence (428 dimensions)"""
        aac = self.amino_acid_composition(sequence)
        dpc = self.dipeptide_composition(sequence)
        physchem = self.physicochemical_features(sequence)

        # Combine all features — SAME ORDER AS TRAINING (admet_model.py)
        all_features = np.concatenate([aac, dpc, physchem])
        return all_features


# ============ Composite multi-objective score ============

# Endpoint names and whether a HIGH probability is desirable.
ENDPOINT_NAMES = ['GI_absorption', 'Caco2_permeability', 'BBB_penetration',
                  'Ames_mutagenicity', 'hERG_inhibition']
HIGHER_IS_BETTER = {'GI_absorption': True, 'Caco2_permeability': True,
                    'BBB_penetration': True, 'Ames_mutagenicity': False,
                    'hERG_inhibition': False}


def composite_score(probs: dict) -> float:
    """
    Multi-objective composite score in [0, 1].

    Geometric mean of the *favourable* probability per endpoint
    (p for beneficial endpoints, 1-p for toxic ones).  The geometric mean
    is the natural "all endpoints must be acceptable" aggregator: a single
    badly-failing endpoint drags the whole score down, mirroring how
    AMPGAN v3 / PepCraft (arXiv:2606.17127) rank candidates by joint
    activity–selectivity–ADMET rather than any single endpoint.

    ``probs``: {endpoint_name: probability in [0,1]}
    """
    fav = []
    for name in ENDPOINT_NAMES:
        p = min(max(float(probs[name]), 1e-6), 1.0 - 1e-6)
        fav.append(p if HIGHER_IS_BETTER[name] else 1.0 - p)
    return float(np.exp(np.mean(np.log(np.array(fav)))))


# ============ Model Loading ============

class PeptideAdmetPredictor:
    """
    Loads the trained PyTorch MLP (admet_model.py format) + StandardScaler
    + measured metrics.json from a model directory.
    """

    def __init__(self, model_dir: str = 'peptide_admet_model'):
        import torch  # imported lazily so --help works without torch

        self._torch = torch
        self.feature_extractor = PeptideFeatureExtractor()
        self.model_dir = Path(model_dir)

        model_path = self.model_dir / 'admet_mlp.pt'
        scaler_path = self.model_dir / 'scaler.pt'
        self.metrics_path = self.model_dir / 'metrics.json'

        if not model_path.exists():
            raise FileNotFoundError(
                f'Model file not found: {model_path}\n'
                f'Run first:  python prepare_data.py && python homology_split.py && python train_peptide_admet_model.py')
        if not scaler_path.exists():
            raise FileNotFoundError(f'Scaler file not found: {scaler_path}')

        from admet_model import load_admet_model
        self.model, self.model_meta = load_admet_model(str(model_path))
        self.device = 'cpu'

        self.scaler = torch.load(scaler_path, map_location='cpu', weights_only=False)

        # Measured performance (never hardcoded)
        self.metrics = {}
        if self.metrics_path.exists():
            with open(self.metrics_path, 'r', encoding='utf-8') as f:
                self.metrics = json.load(f)

        n_params = sum(p.numel() for p in self.model.parameters())
        origin = self.metrics.get('data', {}).get('data_origin', '?')
        print(f"✅ Model loaded from {self.model_dir} "
              f"({n_params:,} params, data: {origin})")

    # ------------------------------------------------------------------
    def _probabilities(self, sequences) -> np.ndarray:
        """Return (n, 5) probability matrix, one column per endpoint."""
        X = np.stack([self.feature_extractor.extract_all_features(s) for s in sequences])
        Xs = self.scaler.transform(X)
        xt = self._torch.from_numpy(Xs.astype(np.float32))
        with self._torch.no_grad():
            logits = self.model(xt)
        probs = self._torch.sigmoid(logits).numpy()
        return probs

    def predict(self, sequence: str) -> dict:
        """
        Predict ADMET properties for a single peptide sequence.

        Returns {'results': [...per endpoint...], 'composite_score': float}
        """
        if not self.feature_extractor.validate_sequence(sequence):
            raise ValueError(f"Invalid peptide sequence: {sequence}. "
                             f"Use only standard amino acids (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)")

        probs = self._probabilities([sequence])[0]
        prob_map = {ENDPOINT_NAMES[i]: float(probs[i]) for i in range(5)}

        results = []
        for i, endpoint in enumerate(ENDPOINT_NAMES):
            prob = prob_map[endpoint]
            binary_pred = 1 if prob >= 0.5 else 0
            results.append({
                'endpoint': endpoint,
                'probability': round(prob, 4),
                'prediction': int(binary_pred),
                'interpretation': self._get_interpretation(endpoint, binary_pred),
                'risk_level': self._get_risk_level(endpoint, prob)
            })

        return {'results': results, 'composite_score': round(composite_score(prob_map), 4),
                'probabilities': {k: round(v, 4) for k, v in prob_map.items()}}

    def _get_interpretation(self, endpoint: str, prediction: int) -> str:
        """Get interpretation for prediction"""
        interpretations = {
            'GI_absorption': {
                0: '低腸胃吸收 (Poor GI absorption)',
                1: '高腸胃吸收 (Good GI absorption)'
            },
            'Caco2_permeability': {
                0: '低腸道穿透性 (Poor Caco-2 permeability)',
                1: '高腸道穿透性 (Good Caco-2 permeability)'
            },
            'BBB_penetration': {
                0: '無法穿透血腦屏障 (Poor BBB penetration)',
                1: '可穿透血腦屏障 (Good BBB penetration)'
            },
            'Ames_mutagenicity': {
                0: '安全（非致突變）(Safe, non-mutagenic)',
                1: '潛在致突變風險 (Potential mutagenicity risk)'
            },
            'hERG_inhibition': {
                0: '安全（低心毒性風險）(Safe, low cardiotoxicity risk)',
                1: '潛在心毒性風險 (Potential cardiotoxicity risk)'
            }
        }
        return interpretations.get(endpoint, {}).get(prediction, 'Unknown')

    def _get_risk_level(self, endpoint: str, probability: float) -> str:
        """Get risk level based on probability"""
        if endpoint in ['Ames_mutagenicity', 'hERG_inhibition']:
            # For toxicity endpoints, higher probability = higher risk
            if probability < 0.3:
                return '✅ 低風險 (Low Risk)'
            elif probability < 0.5:
                return '⚠️ 中等風險 (Moderate Risk)'
            else:
                return '❌ 高風險 (High Risk)'
        else:
            # For absorption/permeability, higher probability = better
            if probability > 0.7:
                return '✅ 優秀 (Excellent)'
            elif probability > 0.5:
                return '⚠️ 良好 (Good)'
            else:
                return '⚠️ 需優化 (Needs Optimization)'

    # ------------------------------------------------------------------
    def model_info(self) -> dict:
        """Return the *measured* model info (from metrics.json), with an
        explicit placeholder-free message when metrics are unavailable."""
        m = self.metrics
        if not m:
            return {'model_type': 'PyTorch MLP (per-endpoint heads)',
                    'note': 'metrics.json not found — performance not stated. '
                            'Run train_peptide_admet_model.py to produce measured metrics.'}
        primary = m.get('splits', {}).get('primary', {})
        test = primary.get('test', {})
        h = m.get('headline', {})
        return {
            'model_type': m.get('model', 'PyTorch MLP (per-endpoint heads)'),
            'trained_on': m.get('data', {}).get('data_origin'),
            'train_samples': primary.get('counts', {}).get('train'),
            'eval_split': 'homology-controlled (AMPBench-MT-style, arXiv:2607.25518)',
            'mean_auc_homology_split': h.get('primary_macro_auc'),
            'mean_acc_homology_split': round(
                float(np.mean([test[e]['accuracy'] for e in ENDPOINT_NAMES
                               if e in test])), 4) if test else None,
            'per_endpoint_auc_homology': {
                e: test[e]['auc'] for e in ENDPOINT_NAMES if e in test},
            'disclaimer': ('Training data is the synthetic demo set — numbers '
                           'validate the pipeline, not real-peptide performance.')
        }


# ============ Output Formatting ============

def print_prediction_result(sequence: str, out: dict, predictor: PeptideAdmetPredictor):
    """Print prediction results in a readable format"""
    results = out['results']
    print("\n" + "="*70)
    print(f"Peptide ADMET Prediction Results")
    print("="*70)
    print(f"\nSequence: {sequence}")
    print(f"Length: {len(sequence)} amino acids")
    print(f"Feature Dimensions: 428 (AAC: 20 + DPC: 400 + PhysChem: 8)")
    print("\n" + "-"*70)

    for result in results:
        endpoint = result['endpoint'].replace('_', ' ').title()
        prob = result['probability']
        pred = result['prediction']
        interp = result['interpretation']
        risk = result['risk_level']

        # Color-coded output
        if 'mutagenicity' in result['endpoint'].lower():
            status = "🧬" if pred == 1 else "✅"
        elif 'herg' in result['endpoint'].lower():
            status = "❤️" if pred == 1 else "✅"
        else:
            status = "📊"

        print(f"\n{status} {endpoint}:")
        print(f"   Probability: {prob:.4f}")
        print(f"   Prediction: {interp}")
        print(f"   Risk Level: {risk}")

        # Progress bar
        bar_length = 30
        filled_length = int(bar_length * prob)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   [{bar}] {prob*100:.1f}%")

    print("\n" + "-"*70)
    print(f"Composite multi-objective score: {out['composite_score']:.4f}  (geometric mean of favourable endpoint probabilities)")
    info = predictor.model_info()
    if 'mean_auc_homology_split' in info:
        print(f"Measured on {info['eval_split']} split ({info.get('trained_on')} data, "
              f"{info.get('train_samples')} train samples): "
              f"mean AUC = {info['mean_auc_homology_split']:.4f}, "
              f"mean accuracy = {info['mean_acc_homology_split']:.4f}")
        print("NOTE: " + info['disclaimer'])
    else:
        print(info.get('note', 'No measured metrics available.'))
    print("="*70)


def print_ranked_results(results_list: list):
    """Print candidates ranked by composite multi-objective score (desc)."""
    ranked = sorted(results_list, key=lambda r: r[1]['composite_score'], reverse=True)
    print("\n" + "="*70)
    print("Candidates ranked by composite multi-objective ADMET score")
    print("="*70)
    print(f"{'#':<3} {'score':<8} {'sequence':<26} {'GI':>5} {'Caco2':>6} {'BBB':>5} {'Ames':>5} {'hERG':>5}")
    print("-"*70)
    for i, (seq, out) in enumerate(ranked, 1):
        p = out['probabilities']
        print(f"{i:<3} {out['composite_score']:<8.4f} {seq[:24]:<26} "
              f"{p['GI_absorption']:>5.3f} {p['Caco2_permeability']:>6.3f} "
              f"{p['BBB_penetration']:>5.3f} {p['Ames_mutagenicity']:>5.3f} "
              f"{p['hERG_inhibition']:>5.3f}")
    print("="*70)


def print_batch_results(results_list: list):
    """Print batch prediction results"""
    print("\n" + "="*70)
    print(f"Peptide ADMET Batch Prediction Results")
    print("="*70)
    print(f"Total sequences: {len(results_list)}\n")

    for i, (sequence, out) in enumerate(results_list, 1):
        print(f"[{i}/{len(results_list)}] Sequence: {sequence}")
        print(f"   Length: {len(sequence)} AA")

        for result in out['results']:
            endpoint = result['endpoint'].replace('_', ' ').title()
            prob = result['probability']
            risk = result['risk_level']
            print(f"   {endpoint}: {prob:.4f} [{risk}]")
        print(f"   Composite score: {out['composite_score']:.4f}")
        print()

    print("="*70)


# ============ Main Functions ============

def interactive_mode(predictor: PeptideAdmetPredictor):
    """Interactive prediction mode"""
    print("\n" + "="*70)
    print("Peptide ADMET Prediction - Interactive Mode")
    print("="*70)
    print("Enter peptide sequences to predict ADMET properties.")
    print("Enter 'quit' or 'exit' to terminate.\n")

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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Peptide ADMET Prediction Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sequence prediction
  python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"

  # Batch prediction from file
  python peptide_admet_predictor.py --sequences sequences.txt

  # Rank candidates by composite multi-objective score
  python peptide_admet_predictor.py --sequences sequences.txt --rank

  # Interactive mode
  python peptide_admet_predictor.py --interactive

  # JSON output
  python peptide_admet_predictor.py --sequence "ACDE" --output results.json
        """
    )

    parser.add_argument('--sequence', '-s', type=str, help='Single peptide sequence')
    parser.add_argument('--sequences', '-f', type=str, help='File containing peptide sequences (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--output', '-o', type=str, help='Output file (JSON format)')
    parser.add_argument('--model-dir', '-m', type=str, default='peptide_admet_model',
                       help='Directory containing trained models (default: peptide_admet_model)')
    parser.add_argument('--rank', action='store_true',
                       help='With --sequences: rank candidates by composite multi-objective score')

    args = parser.parse_args()

    # Check if any mode is specified
    if not (args.sequence or args.sequences or args.interactive):
        parser.print_help()
        print("\n❌ Please specify a sequence, file, or use interactive mode.")
        sys.exit(1)

    # Initialize predictor
    try:
        predictor = PeptideAdmetPredictor(model_dir=args.model_dir)
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        sys.exit(1)

    # Process predictions
    if args.interactive:
        interactive_mode(predictor)

    elif args.sequence:
        try:
            out = predictor.predict(args.sequence)
            print_prediction_result(args.sequence, out, predictor)

            # Save to JSON if requested
            if args.output:
                output_data = {
                    'sequence': args.sequence,
                    'length': len(args.sequence),
                    'composite_score': out['composite_score'],
                    'predictions': out['results'],
                    'probabilities': out['probabilities'],
                    'model_info': predictor.model_info()
                }
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"\n✅ Results saved to {args.output}")

        except ValueError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif args.sequences:
        # Read sequences from file
        try:
            seq_file = Path(args.sequences)
            if not seq_file.exists():
                print(f"❌ File not found: {args.sequences}")
                sys.exit(1)

            with open(seq_file, 'r', encoding='utf-8') as f:
                sequences = [line.strip() for line in f if line.strip()]

            # Predict for each sequence
            results_list = []
            for seq in sequences:
                try:
                    out = predictor.predict(seq)
                    results_list.append((seq, out))
                except ValueError as e:
                    print(f"⚠️  Skipping invalid sequence '{seq[:20]}...': {e}")

            # Print results
            if args.rank:
                print_ranked_results(results_list)
            else:
                print_batch_results(results_list)

            # Save to JSON if requested
            if args.output:
                output_data = {
                    'total_sequences': len(results_list),
                    'successful_predictions': len(results_list),
                    'ranked_by_composite_score': args.rank,
                    'predictions': [
                        {
                            'sequence': seq,
                            'length': len(seq),
                            'composite_score': out['composite_score'],
                            'probabilities': out['probabilities'],
                            'predictions': out['results']
                        }
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
