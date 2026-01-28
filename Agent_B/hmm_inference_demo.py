import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import os
from scipy.stats import multivariate_normal

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

FEATURES = get_all_feature_names()  # 9 features total


def make_trend_features(df):
    """
    Engineer trend features for HMM (same as hmm_prototype.py).
    """
    df = df.copy()
    
    for feat in TREND_SOURCE_FEATURES:
        if feat not in df.columns:
            continue
        delta_col = f"{feat}_delta1h"
        df[delta_col] = df.groupby('person_id')[feat].diff(1)
        
        slope_col = f"{feat}_slope4h"
        df[slope_col] = df.groupby('person_id')[feat].transform(
            lambda x: x.diff(4) / 4.0
        )
    
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


def visualize_patient_trajectory(model, patient_data, X_patient, state_map, filename):
    """
    Visualization using TRUE FILTERED inference (forward algorithm).
    """
    time = patient_data['measure_time'].values
    
    filtered_probs = forward_algorithm(model, X_patient)
    current_states = np.argmax(filtered_probs, axis=1)
    
    forecast_probs = filtered_probs @ model.transmat_
    forecast_states = np.argmax(forecast_probs, axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), constrained_layout=True)
    
    ax1 = axes[0, 0]
    ax1.plot(time, current_states, label='Current State (Filtered)', marker='o', linestyle='-', color='blue')
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([state_map[i] for i in range(4)])
    ax1.set_title('Current State (Filtered P(S_t|x_0:t))')
    ax1.set_xlabel('Hours after intubation')
    ax1.grid(True)
    
    ax2 = axes[0, 1]
    ax2.plot(time, forecast_states, label='Forecast (t+1)', marker='x', linestyle='--', color='orange')
    ax2.set_yticks(range(4))
    ax2.set_yticklabels([state_map[i] for i in range(4)])
    ax2.set_title('Forecasted State (t+1)')
    ax2.set_xlabel('Hours after intubation')
    ax2.grid(True)
    
    ax3 = axes[0, 2]
    ax3.plot(time, current_states, label='Current (Filtered)', color='blue', alpha=0.6, linewidth=2)
    ax3.plot(time, forecast_states, label='Forecast', color='orange', alpha=0.6, linestyle='--', linewidth=2)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels([state_map[i] for i in range(4)])
    ax3.set_title('Overlay: Current vs Forecast')
    ax3.set_xlabel('Hours after intubation')
    ax3.legend()
    ax3.grid(True)
    
    ax4 = axes[1, 0]
    confidence_current = np.max(filtered_probs, axis=1)
    sc1 = ax4.scatter(time, confidence_current, c=confidence_current, cmap='viridis', vmin=0, vmax=1)
    ax4.set_title('Confidence of Current Classification')
    ax4.set_xlabel('Hours after intubation')
    ax4.set_ylabel('Probability')
    ax4.set_ylim(-0.05, 1.05)
    plt.colorbar(sc1, ax=ax4)
    
    ax5 = axes[1, 1]
    confidence_forecast = np.max(forecast_probs, axis=1)
    sc2 = ax5.scatter(time, confidence_forecast, c=confidence_forecast, cmap='viridis', vmin=0, vmax=1)
    ax5.set_title('Confidence of Forecast (t+1)')
    ax5.set_xlabel('Hours after intubation')
    ax5.set_ylabel('Probability')
    ax5.set_ylim(-0.05, 1.05)
    plt.colorbar(sc2, ax=ax5)
    
    ax6 = axes[1, 2]
    ax6.axis('off')
    handles, labels = ax3.get_legend_handles_labels()
    ax6.legend(handles, labels, loc='center', title="State Trajectories", fontsize='large')
    
    plt.savefig(filename)
    print(f"Visualization saved to {filename}")
    plt.close()


def find_subjects_with_state_changes(df, model, scaler, state_map, n_subjects=3):
    """
    Find subjects who have at least one state change during their time series.
    Returns a list of (person_id, n_transitions) tuples.
    """
    print("\nSearching for subjects with state changes...")
    candidates = []
    
    for person_id in df['person_id'].unique():
        patient_data = df[df['person_id'] == person_id].sort_values('measure_time')
        if len(patient_data) < 5:  # Need enough data points
            continue
        
        X_patient = scaler.transform(patient_data[FEATURES])
        states = model.predict(X_patient)
        
        # Count transitions (state changes)
        n_transitions = np.sum(np.diff(states) != 0)
        if n_transitions > 0:
            candidates.append((person_id, n_transitions))
    
    # Sort by number of transitions (most transitions first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(candidates)} subjects with at least one state change.")
    return candidates[:n_subjects]


def main():
    model_path = 'hmm_model.pkl'
    scaler_path = 'scaler.pkl'
    data_path = "clinical_data/data_v2_max_72_h.parquet"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Error: Model or Scaler file not found. Please run hmm_prototype.py first.")
        return

    # 1. Load Model and Scaler
    print("Loading model and scaler...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # 2. Load and Preprocess Data
    print("Loading data...")
    df = pd.read_parquet(data_path)
    df = df.reset_index()
    
    state_map = {0: 'Acute', 1: 'Recovery', 2: 'Weaning', 3: 'Liberation'}

    # Keep base features + identifiers
    df = df[['person_id', 'measure_time'] + BASE_FEATURES].copy()
    df.sort_values(by=['person_id', 'measure_time'], inplace=True)
    df[BASE_FEATURES] = df.groupby('person_id')[BASE_FEATURES].ffill().bfill()
    df.dropna(subset=BASE_FEATURES, inplace=True)
    
    # Add trend features (to match model training)
    df = make_trend_features(df)

    # =========================================================================
    # SECTION 1: Find and Plot 3 Subjects with State Changes
    # =========================================================================
    print("\n" + "="*60)
    print("Finding subjects with state changes...")
    print("="*60)
    
    subjects_with_changes = find_subjects_with_state_changes(df, model, scaler, state_map, n_subjects=3)
    
    if len(subjects_with_changes) == 0:
        print("No subjects with state changes found. Falling back to random selection.")
        subjects_with_changes = [(random.choice(df['person_id'].unique()), 0)]
    
    for i, (person_id, n_trans) in enumerate(subjects_with_changes):
        print(f"\nSubject {i+1}: person_id={person_id}, state_transitions={n_trans}")
        
        patient_data = df[df['person_id'] == person_id].sort_values('measure_time').copy()
        X_patient = scaler.transform(patient_data[FEATURES])
        
        hidden_states = model.predict(X_patient)
        patient_data['predicted_state'] = hidden_states
        patient_data['state_label'] = patient_data['predicted_state'].map(state_map)
        
        print(f"  Time points: {len(patient_data)}")
        print(f"  States visited: {patient_data['state_label'].unique().tolist()}")
        
        filename = f"demo_state_change_subject_{i+1}_{person_id}.png"
        visualize_patient_trajectory(model, patient_data, X_patient, state_map, filename=filename)

    # =========================================================================
    # SECTION 2: Also plot a random subject for comparison
    # =========================================================================
    print("\n" + "="*60)
    print("Random subject demo...")
    print("="*60)
    
    random.seed(None)  # Use system time for randomness
    sample_id = random.choice(df['person_id'].unique())
    print(f"\nPerforming inference for Random Subject: {sample_id}")
    
    patient_data = df[df['person_id'] == sample_id].sort_values('measure_time').copy()
    X_patient = scaler.transform(patient_data[FEATURES])

    hidden_states = model.predict(X_patient)
    patient_data['predicted_state'] = hidden_states
    patient_data['state_label'] = patient_data['predicted_state'].map(state_map)

    print("\nInference Result (first 10 hours):")
    print(patient_data[['measure_time', 'peep_mean', 'peak_mean', 'sbp_mean', 'state_label']].head(10))

    visualize_patient_trajectory(model, patient_data, X_patient, state_map, filename=f"demo_inference_{sample_id}.png")
    
    print("\n" + "="*60)
    print("Done! Generated visualizations:")
    for i, (person_id, _) in enumerate(subjects_with_changes):
        print(f"  - demo_state_change_subject_{i+1}_{person_id}.png")
    print(f"  - demo_inference_{sample_id}.png")
    print("="*60)

if __name__ == "__main__":
    main()



'''

Debugging with Claude Opus 4.5 with copilot because everything was broken...


================================================================================
HMM MODELING DECISIONS & CHANGES - Session Summary
================================================================================

PROBLEMS DIAGNOSED:
1. Corrupted data - PEEP values of 1637 (physiologically impossible)
2. Covariance explosion - Trained covariances grew to 10^11, causing one state 
   to absorb all observations (100% classified as "Recovery")
3. Clinical centroids mismatched data - Initial means were too far from actual 
   data distribution (e.g., Acute PEEP=14 vs data mean=7.1)

--------------------------------------------------------------------------------
FIXES APPLIED:
--------------------------------------------------------------------------------

| Issue                  | Solution                                            |
|------------------------|-----------------------------------------------------|
| Outliers/bad data      | Added physiological clipping in load_and_preprocess |
| Covariance explosion   | Changed params='stc' → params='st' (freeze covars)  |
| Centroid mismatch      | Adjusted clinical centroids to match data range     |
| Viterbi uses future    | Replaced with forward_algorithm() for real-time     |

--------------------------------------------------------------------------------
FINAL MODEL CONFIGURATION:
--------------------------------------------------------------------------------

    model = GaussianHMM(
        n_components=4,
        covariance_type="diag",
        init_params="",    # Don't auto-initialize anything
        params='st',       # Only train: start probs + transitions
        n_iter=100
    )

What's FROZEN (clinically defined):
  - Means: 4 states based on PROMIZING Protocol cutoffs
  - Covariances: Fixed at 1.0 (base features), 2.0 (trend features)

What's LEARNED from data:
  - Transition matrix: How patients move between states
  - Start probabilities: Initial state distribution

--------------------------------------------------------------------------------
PHYSIOLOGICAL BOUNDS ADDED:
--------------------------------------------------------------------------------

| Feature       | Range         | Rationale              |
|---------------|---------------|------------------------|
| PEEP          | 0-25 cmH₂O    | Clinical max           |
| Peak pressure | 5-60 cmH₂O    | Ventilator limits      |
| SBP           | 40-250 mmHg   | Viable BP range        |
| FiO₂          | 21-100%       | Room air to pure O₂    |

--------------------------------------------------------------------------------
INFERENCE CHANGE:
--------------------------------------------------------------------------------

| Before                      | After                                |
|-----------------------------|--------------------------------------|
| model.predict() (Viterbi)   | forward_algorithm() (filtered)       |
| Uses all data incl. future  | Uses only past observations          |
| Good for retrospective      | Valid for real-time bedside use      |

--------------------------------------------------------------------------------
KEY TAKEAWAY:
--------------------------------------------------------------------------------
This is now a SEMI-SUPERVISED HMM: clinical knowledge defines *what* the states 
mean (fixed emissions), while the data teaches *how* patients transition between 
them (learned dynamics).

================================================================================
'''