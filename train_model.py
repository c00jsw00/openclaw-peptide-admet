#!/usr/bin/env python3
"""
Peptide ADMET Model Training Script
====================================

Train the ensemble model for peptide ADMET prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import joblib

from peptide_admet_model import PeptideFeatureExtractor, PeptideADMETModel


def load_training_data(data_dir: str = 'real_peptide_data'):
    """Load training data"""
    data_path = Path(data_dir)
    
    # Load features
    X_train = np.load(data_path / 'X_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    X_test = np.load(data_path / 'X_test.npy')
    
    # Load labels
    y_train = np.load(data_path / 'y_train.npy')
    y_val = np.load(data_path / 'y_val.npy')
    y_test = np.load(data_path / 'y_test.npy')
    
    print(f"✅ Loaded training data:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_ensemble_model(X_train, y_train, X_val, y_val):
    """Train ensemble model (Random Forest + Neural Network)"""
    
    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf_model = PeptideADMETModel(model_type='random_forest', num_features=X_train.shape[1])
    rf_model.train(X_train, y_train, X_val, y_val)
    
    # Train Neural Network
    print("\n🧠 Training Neural Network...")
    nn_model = PeptideADMETModel(model_type='neural_network', num_features=X_train.shape[1])
    nn_model.train(X_train, y_train, X_val, y_val, epochs=30)
    
    return rf_model, nn_model


def save_models(rf_model, nn_model, scaler, feature_extractor, output_dir='peptide_admet_model'):
    """Save trained models"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save models
    with open(output_path / 'rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model.model, f)
    
    with open(output_path / 'nn_model.pkl', 'wb') as f:
        pickle.dump(nn_model.model, f)
    
    # Save scaler
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature extractor
    with open(output_path / 'feature_extractor.pkl', 'wb') as f:
        pickle.dump(feature_extractor, f)
    
    # Save feature names
    with open(output_path / 'feature_names.txt', 'w') as f:
        for name in feature_extractor.feature_names:
            f.write(name + '\n')
    
    print(f"\n✅ Models saved to {output_dir}/")


def main():
    """Main training pipeline"""
    print("="*70)
    print("Peptide ADMET Model Training")
    print("="*70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data()
    
    # Train ensemble
    rf_model, nn_model = train_ensemble_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\n📊 Evaluating models...")
    rf_metrics = rf_model.evaluate(X_test, y_test)
    nn_metrics = nn_model.evaluate(X_test, y_test)
    
    print(f"\nRandom Forest Test Metrics:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nNeural Network Test Metrics:")
    for metric, value in nn_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save models
    scaler = joblib.StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    feature_extractor = PeptideFeatureExtractor()
    save_models(rf_model, nn_model, scaler, feature_extractor)
    
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()
