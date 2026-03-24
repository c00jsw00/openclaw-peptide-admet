# Usage Guide for Peptide ADMET Prediction Model

## Installation

### 1. Install Dependencies

```bash
# Basic dependencies
pip install torch scikit-learn pandas numpy joblib

# Optional: for visualization
pip install matplotlib seaborn
```

### 2. Clone Repository

```bash
git clone https://github.com/c00jsw00/openclaw-peptide-admet.git
cd openclaw-peptide-admet
```

---

## Quick Start

### Basic Usage

```python
from peptide_admet_inference import PeptideADMETPredictor

# Initialize predictor
predictor = PeptideADMETPredictor(model_dir='peptide_admet_model')

# Predict for a single sequence
sequence = "ACDEFGHIKLMNPQRSTVWY"
results = predictor.predict(sequence)
predictor.print_result(results)
```

### Output Format

```
============================================================
Peptide ADMET Prediction Results
============================================================

GI Absorption:
  Probability: 0.9823
  Prediction: ✅ GOOD
  Interpretation: High GI absorption

Caco-2 Permeability:
  Probability: 0.9901
  Prediction: ✅ GOOD
  Interpretation: High Caco-2 permeability

BBB Penetration:
  Probability: 0.0234
  Prediction: ⚠️ POOR
  Interpretation: Poor BBB penetration

Ames Mutagenicity:
  Probability: 0.0156
  Prediction: ✅ SAFE
  Interpretation: Safe (non-mutagenic)

hERG Inhibition:
  Probability: 0.0089
  Prediction: ✅ SAFE
  Interpretation: Safe (low hERG risk)

============================================================
```

---

## API Reference

### PeptideADMETPredictor Class

#### Constructor

```python
predictor = PeptideADMETPredictor(model_dir='path/to/models')
```

**Parameters:**
- `model_dir` (str): Path to directory containing trained model files

**Model Files Required:**
- `feature_extractor.pkl` - Feature extraction logic
- `rf_model.pkl` - Random Forest model
- `nn_model.pkl` - Neural Network model
- `scaler.pkl` - Standardization scaler
- `feature_names.txt` - Feature names

#### predict() Method

```python
results = predictor.predict(sequences, endpoints=None)
```

**Parameters:**
- `sequences` (str or list): Single peptide sequence or list of sequences
- `endpoints` (list, optional): List of specific endpoints to predict
  - Available: `['GI_absorption', 'Caco2_permeability', 'BBB_penetration', 'Ames_mutagenicity', 'hERG_inhibition']`

**Returns:**
- Single dict if input is string
- List of dicts if input is list

**Example:**

```python
# Single sequence
seq = "ACDEFGHIKLMNPQRSTVWY"
results = predictor.predict(seq)

# Batch prediction
sequences = ["ACDEFGH", "GAGAGA", "KKKKK"]
results = predictor.predict(sequences)

# Specific endpoints
results = predictor.predict(seq, endpoints=['GI_absorption', 'BBB_penetration'])
```

#### print_result() Method

```python
predictor.print_result(results)
```

**Parameters:**
- `results` (dict): Prediction results from `predict()` method

**Example:**

```python
results = predictor.predict("ACDEFGHIKLMNPQRSTVWY")
predictor.print_result(results)
```

---

## Command Line Usage

### Single Sequence

```bash
python peptide_admet_inference.py --sequence "ACDEFGHIKLMNPQRSTVWY"
```

### Batch Prediction

Create a file `sequences.txt`:

```
ACDEFGHIKLMNPQRSTVWY
GAGAGAGAGAGA
KKKKKKKKKK
VVVVVVVVVV
```

Run:

```bash
python peptide_admet_inference.py --sequences sequences.txt
```

---

## Feature Engineering Details

### Amino Acid Composition (AAC) - 20 features

Frequency of each amino acid in the sequence:

```python
# Example for sequence "ACDE"
AAC = [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Corresponding to: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
```

### Dipeptide Composition (DPC) - 400 features

Frequency of all possible dipeptide combinations:

```python
# Example for sequence "ACDE"
DPC = [0.33, 0.33, 0.33, 0, 0, ..., 0]
# Corresponding to: AA, AC, AD, AE, AF, AG, ... YY
```

### Physicochemical Properties - 8 features

1. **molecular_weight**: Length × 110 Da
2. **avg_hydropathy**: Mean Kyte-Doolittle hydropathy
3. **hydropathy_range**: Max - Min hydropathy
4. **net_charge**: Sum of charges at pH 7.0
5. **pi_estimate**: Estimated isoelectric point
6. **grand_average_hydropathy** (GRAVY): Mean hydropathy
7. **hydrophobic_ratio**: Fraction of hydrophobic residues
8. **charged_ratio**: Fraction of charged residues

---

## Interpretation Guide

### GI Absorption

- **Prediction 0 (Low)**: Poor oral bioavailability expected
  - Likely causes: High molecular weight, high polarity
  - Suggestions: Consider prodrug strategy, parenteral administration

- **Prediction 1 (High)**: Good oral bioavailability expected
  - Favorable properties: Moderate MW, balanced hydrophobicity
  - Advantages: Can be developed as oral formulation

### Caco-2 Permeability

- **Prediction 0 (Low)**: Poor intestinal absorption
  - Likely causes: High polarity, high charge
  - Suggestions: Modify sequence to increase hydrophobicity

- **Prediction 1 (High)**: Good intestinal absorption
  - Favorable properties: Balanced hydrophobicity, moderate charge
  - Advantages: Suitable for oral delivery

### BBB Penetration

- **Prediction 0 (Poor)**: Unlikely to cross blood-brain barrier
  - Likely causes: High MW, high polarity, charged residues
  - Applications: Good for systemic drugs (avoid CNS side effects)

- **Prediction 1 (Good)**: Can cross blood-brain barrier
  - Favorable properties: Moderate MW, hydrophobic, low charge
  - Applications: Suitable for CNS-targeted therapeutics

### Ames Mutagenicity

- **Prediction 0 (Safe)**: Low mutagenicity risk
  - Favorable: No known mutagenic motifs
  - Confidence: High (97.27% accuracy)

- **Prediction 1 (Risk)**: Potentially mutagenic
  - Concern: Contains mutagenic structural alerts
  - Actions: Structure optimization recommended

### hERG Inhibition

- **Prediction 0 (Safe)**: Low cardiotoxicity risk
  - Favorable: Low cationic content, balanced hydrophobicity
  - Confidence: High (97.91% accuracy)

- **Prediction 1 (Risk)**: Potentially cardiotoxic
  - Concern: hERG channel inhibition risk
  - Actions: Critical - must optimize to reduce cardiotoxicity

---

## Best Practices

### 1. Sequence Quality

- Use valid amino acid sequences (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- Sequence length: 8-25 amino acids recommended
- Avoid non-standard amino acids

### 2. Interpretation

- Consider all 5 ADMET endpoints together
- Prioritize safety endpoints (Ames, hERG) for optimization
- Balance efficacy and ADMET properties

### 3. Optimization

- **Low GI absorption**: Increase hydrophobicity, reduce MW
- **Poor BBB penetration**: Add hydrophobic residues, reduce polarity
- **Ames risk**: Remove known mutagenic motifs
- **hERG risk**: Reduce cationic residues, balance hydrophobicity

### 4. Validation

- Experimental validation recommended for final candidates
- Use multiple prediction tools for consensus
- Consider structural similarity to known peptides

---

## Performance Tips

### Speed Optimization

```python
# Batch prediction is faster than individual predictions
sequences = ["SEQ1", "SEQ2", "SEQ3", "SEQ4", "SEQ5"]
results = predictor.predict(sequences)  # Faster than 5 individual calls
```

### Memory Efficiency

```python
# For large batches, process in chunks
large_batches = [...]  # 1000+ sequences
results = []
for i in range(0, len(large_batches), 100):
    chunk = large_batches[i:i+100]
    results.extend(predictor.predict(chunk))
```

---

## Troubleshooting

### Error: Model files not found

```
FileNotFoundError: [Errno 2] No such file or directory: 'peptide_admet_model/rf_model.pkl'
```

**Solution**: Ensure model files exist in the specified directory

```bash
ls peptide_admet_model/
# Should show: rf_model.pkl, nn_model.pkl, scaler.pkl, feature_extractor.pkl
```

### Error: Invalid sequence

```
ValueError: Sequence contains invalid amino acids
```

**Solution**: Use only standard amino acids (A-Z excluding B, J, O, U, X, Z)

```python
# Valid
sequence = "ACDEFGHIKLMNPQRSTVWY"

# Invalid
sequence = "ACBX"  # X, B are not standard
```

### Error: Sequence too short/long

```
Warning: Sequence length {len} outside optimal range (8-25)
```

**Solution**: Performance optimized for 8-25 amino acids. Results may be less accurate for very short or long sequences.

---

## Advanced Usage

### Custom Feature Extraction

```python
from peptide_admet_inference import PeptideFeatureExtractor

extractor = PeptideFeatureExtractor()

# Extract features
sequence = "ACDEFGHIKLMNPQRSTVWY"
features = extractor.extract_all_features(sequence)
print(f"Feature shape: {features.shape}")  # (428,)
print(f"Feature names: {extractor.feature_names}")
```

### Access Raw Predictions

```python
# Get raw probabilities without interpretation
results = predictor.predict("ACDEFGHIKLMNPQRSTVWY")

for item in results:
    print(f"{item['endpoint']}: prob={item['probability']:.4f}, pred={item['prediction']}")
```

### Save Predictions to File

```python
import json

sequences = ["SEQ1", "SEQ2", "SEQ3"]
results = predictor.predict(sequences)

# Save to JSON
with open('predictions.json', 'w') as f:
    json.dump({
        'predictions': results,
        'metadata': {
            'model_version': '1.0',
            'accuracy': 0.9770
        }
    }, f, indent=2)
```

---

## References

1. **Manuscript**: See `peptide_admet_manuscript.md` for full research paper
2. **Training Code**: See `peptide_admet_model.py` for training framework
3. **Model Comparison**: See `model_comparison_pepADMET.md` for detailed comparisons

---

**Version**: 1.0  
**Last Updated**: 2026-03-24  
**Author**: Pinwan (OpenClaw Team)
