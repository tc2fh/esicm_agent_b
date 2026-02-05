"""
HMM Batch Inference Script
===========================
Runs inference on ALL patients and ALL time points in the dataset.
Outputs a CSV with current state, forecast state, probabilities, and raw features.

Usage:
    python hmm_batch_inference.py

Output:
    hmm_batch_predictions.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# --- Feature Configuration (must match hmm_prototype.py) ---
BASE_FEATURES = ['peep_mean', 'peak_mean', 'sbp_mean', 'fio2_mean']
TREND_SOURCE_FEATURES = ['peep_mean', 'peak_mean', 'sbp_mean', 'fio2_mean']

def get_all_feature_names():
    """Returns the full list of features including trends."""
    features = BASE_FEATURES.copy()
    for feat in TREND_SOURCE_FEATURES:
        features.append(f"{feat}_delta1h")
        features.append(f"{feat}_slope4h")
    return features

FEATURES = get_all_feature_names()  # 12 features total

# State mapping
STATE_MAP = {0: 'Acute', 1: 'Recovery', 2: 'Weaning', 3: 'Liberation'}

# Physiological bounds for clipping
PHYSIOLOGICAL_BOUNDS = {
    'peep_mean': (0, 25),      # PEEP: 0-25 cmH2O
    'peak_mean': (5, 60),      # Peak pressure: 5-60 cmH2O
    'sbp_mean': (40, 250),     # SBP: 40-250 mmHg
    'fio2_mean': (21, 100),    # FiO2: 21-100%
}


def make_trend_features(df):
    """
    Engineer trend features for HMM:
    - 1-hour delta: diff(1)
    - 4-hour rolling slope: diff(4) / 4
    """
    df = df.copy()
    
    for feat in TREND_SOURCE_FEATURES:
        if feat not in df.columns:
            continue
        
        # 1-hour delta
        delta_col = f"{feat}_delta1h"
        df[delta_col] = df.groupby('person_id')[feat].diff(1)
        
        # 4-hour rolling slope
        slope_col = f"{feat}_slope4h"
        df[slope_col] = df.groupby('person_id')[feat].transform(
            lambda x: x.diff(4) / 4.0
        )
    
    # Fill NaNs from trend calculation with 0 (no change at start)
    trend_cols = [c for c in df.columns if 'delta' in c or 'slope' in c]
    df[trend_cols] = df[trend_cols].fillna(0)
    
    return df


def forward_algorithm(model, X):
    """
    Compute true filtered posterior P(S_t | x_0:t)
    
    Returns:
        filtered_probs: (T, n_components) array of P(S_t | x_0:t)
    """
    n_samples = len(X)
    n_components = model.n_components
    
    log_startprob = np.log(model.startprob_ + 1e-10)
    log_transmat = np.log(model.transmat_ + 1e-10)
    
    # Precompute log emission probabilities
    log_emit = model._compute_log_likelihood(X)

    # Forward pass in log space
    log_alpha = np.zeros((n_samples, n_components))
    
    # Initialize: alpha_0 = pi * P(x_0 | S_0)
    log_alpha[0] = log_startprob + log_emit[0]
    
    for t in range(1, n_samples):
        log_alpha[t] = log_emit[t] + np.logaddexp.reduce(
            log_alpha[t-1, :, np.newaxis] + log_transmat, axis=0
        )
    
    # Normalize to get filtered probabilities
    log_normalizer = np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
    filtered_probs = np.exp(log_alpha - log_normalizer)
    
    return filtered_probs


def load_and_preprocess(filepath):
    """
    Load data, impute missing values, clip outliers, and add trend features.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        df = df.reset_index()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

    print(f"Data loaded. Shape: {df.shape}")

    # Keep base features + identifiers
    cols_to_keep = ['person_id', 'measure_time'] + BASE_FEATURES
    
    # Check if columns exist
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    df = df[cols_to_keep].copy()

    # Sort by person_id and time
    df.sort_values(by=['person_id', 'measure_time'], inplace=True)
    
    # Imputation: Forward Fill -> Backward Fill per person
    print("Handling missing values...")
    df[BASE_FEATURES] = df.groupby('person_id')[BASE_FEATURES].ffill().bfill()
    
    # Drop rows that are still NaN
    n_before = len(df)
    df.dropna(subset=BASE_FEATURES, inplace=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with remaining NaNs.")
    
    # Clip physiologically impossible values
    print("Clipping outliers...")
    for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df.columns:
            n_clipped = ((df[col] < low) | (df[col] > high)).sum()
            df[col] = df[col].clip(low, high)
            if n_clipped > 0:
                print(f"  {col}: clipped {n_clipped} values to [{low}, {high}]")
    
    # Add trend features
    print("Engineering trend features...")
    df = make_trend_features(df)
    
    print(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def run_batch_inference(df, model, scaler):
    """
    Run inference on all patients and all time points.
    
    Returns:
        DataFrame with inference results
    """
    results = []
    patient_ids = df['person_id'].unique()
    
    print(f"\nRunning inference on {len(patient_ids)} patients...")
    
    for person_id in tqdm(patient_ids, desc="Processing patients"):
        # Get patient data sorted by time
        patient_data = df[df['person_id'] == person_id].sort_values('measure_time')
        
        if len(patient_data) == 0:
            continue
        
        # Scale features for model input
        X_patient = scaler.transform(patient_data[FEATURES].values)
        
        # Run forward algorithm to get filtered probabilities
        filtered_probs = forward_algorithm(model, X_patient)
        
        # Current state: argmax of filtered probabilities
        current_states = np.argmax(filtered_probs, axis=1)
        current_probs = np.max(filtered_probs, axis=1)
        
        # Forecast: P(S_{t+1} | x_0:t) = P(S_t | x_0:t) @ Transition_Matrix
        forecast_probs = filtered_probs @ model.transmat_
        forecast_states = np.argmax(forecast_probs, axis=1)
        forecast_max_probs = np.max(forecast_probs, axis=1)
        
        # Collect results for each time point
        for i, row in enumerate(patient_data.itertuples()):
            results.append({
                'patient_id': person_id,
                'measure_time': row.measure_time,
                # Current state
                'current_state': current_states[i],
                'current_state_label': STATE_MAP[current_states[i]],
                'current_state_probability': round(current_probs[i], 4),
                # Forecast state (1 hour ahead)
                'forecast_state_1h': forecast_states[i],
                'forecast_state_1h_label': STATE_MAP[forecast_states[i]],
                'forecast_state_probability': round(forecast_max_probs[i], 4),
                # Raw feature values
                'peep_mean': row.peep_mean,
                'peak_mean': row.peak_mean,
                'sbp_mean': row.sbp_mean,
                'fio2_mean': row.fio2_mean,
            })
    
    return pd.DataFrame(results)


def main():
    # File paths
    model_path = 'hmm_model.pkl'
    scaler_path = 'scaler.pkl'
    data_path = "clinical_data/data_v3_max_72_h.parquet"
    output_path = "hmm_batch_predictions.csv"
    
    # Check for required files
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run hmm_prototype.py first to train the model.")
        sys.exit(1)
    
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file '{scaler_path}' not found.")
        print("Please run hmm_prototype.py first to train the model.")
        sys.exit(1)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file '{data_path}' not found.")
        sys.exit(1)
    
    # Load model and scaler
    print("Loading model and scaler...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Model loaded: {model.n_components} states, {len(FEATURES)} features")
    
    # Load and preprocess data
    df = load_and_preprocess(data_path)
    
    # Run batch inference
    results_df = run_batch_inference(df, model, scaler)
    
    # Summary statistics
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    print(f"Total patients: {results_df['patient_id'].nunique()}")
    print(f"Total time points: {len(results_df)}")
    print(f"\nState distribution:")
    state_counts = results_df['current_state_label'].value_counts()
    for state, count in state_counts.items():
        pct = count / len(results_df) * 100
        print(f"  {state}: {count} ({pct:.1f}%)")
    
    print(f"\nMean current state probability: {results_df['current_state_probability'].mean():.3f}")
    print(f"Mean forecast probability: {results_df['forecast_state_probability'].mean():.3f}")
    
    # Save to CSV
    print(f"\nSaving results to {output_path}...")
    results_df.to_csv(output_path, index=False)
    print(f"Done! Output saved to {output_path}")
    
    # Show sample output
    print("\nSample output (first 10 rows):")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

'''
============================================================
INFERENCE SUMMARY
============================================================
Total patients: 65015
Total time points: 3270335

State distribution:
  Weaning: 1311005 (40.1%)
  Recovery: 1308299 (40.0%)
  Liberation: 377213 (11.5%)
  Acute: 273818 (8.4%)

Mean current state probability: 0.903
Mean forecast probability: 0.896

Saving results to hmm_batch_predictions.csv...
Done! Output saved to hmm_batch_predictions.csv
'''