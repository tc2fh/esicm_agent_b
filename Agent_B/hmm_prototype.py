import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import sys
import matplotlib.pyplot as plt
import pickle
import random

# --- 1. CONFIGURATION ---
# Base features (vitals from PROMIZING Protocol)
BASE_FEATURES = ['peep_mean', 'peak_mean', 'sbp_mean', 'fio2_mean']

# Trend features will be added dynamically
TREND_SOURCE_FEATURES = ['peep_mean', 'peak_mean', 'sbp_mean', 'fio2_mean']

def get_all_feature_names():
    """Returns the full list of features including trends."""
    features = BASE_FEATURES.copy()
    for feat in TREND_SOURCE_FEATURES:
        features.append(f"{feat}_delta1h")
        features.append(f"{feat}_slope4h")
    return features

FEATURES = get_all_feature_names()  # Will be: base + 8 trend features = 12 features

# Indices for the "Driver" features (base features only)
IDX_PEEP = 0
IDX_PEAK = 1
IDX_SBP  = 2
IDX_FIO2 = 3

def load_and_preprocess(filepath):
    """
    Load data, impute missing values, and handle FiO2 scaling/conversion
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        df = df.reset_index()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

    print("Data loaded. Shape:", df.shape)

    # Filter for selected features + identifiers
    cols_to_keep = ['person_id', 'measure_time'] + BASE_FEATURES
    
    # Check if columns exist
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in dataset: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        sys.exit(1)

    df_subset = df[cols_to_keep].copy()

    # Sort by person_id and time
    df_subset.sort_values(by=['person_id', 'measure_time'], inplace=True)
    
    # Imputation: Forward Fill -> Backward Fill per person
    print("Handling missing values...")
    df_subset[BASE_FEATURES] = df_subset.groupby('person_id')[BASE_FEATURES].ffill().bfill()
    
    # Drop rows that are still NaN
    df_subset.dropna(subset=BASE_FEATURES, inplace=True)
    
    # ADD: Clip physiologically impossible values
    print("Clipping outliers...")
    PHYSIOLOGICAL_BOUNDS = {
        'peep_mean': (0, 25),      # PEEP: 0-25 cmH2O
        'peak_mean': (5, 60),      # Peak pressure: 5-60 cmH2O
        'sbp_mean': (40, 250),     # SBP: 40-250 mmHg
        'fio2_mean': (21, 100),    # FiO2: 21-100%
    }
    
    for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df_subset.columns:
            n_clipped = ((df_subset[col] < low) | (df_subset[col] > high)).sum()
            df_subset[col] = df_subset[col].clip(low, high)
            if n_clipped > 0:
                print(f"  {col}: clipped {n_clipped} values to [{low}, {high}]")
    
    print("\nData quality check:")
    for col in BASE_FEATURES:
        print(f"{col}: min={df_subset[col].min():.1f}, max={df_subset[col].max():.1f}, "
              f"mean={df_subset[col].mean():.1f}, NaN={df_subset[col].isna().sum()}")

    return df_subset



def make_trend_features(df):
    """
    Engineer trend features for HMM:
    - 1-hour delta: diff(1)
    - 4-hour rolling slope: diff(4) / 4
    """
    print("Engineering trend features...")
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
    
    print(f"  Added {len(trend_cols)} trend features.")
    return df


def prepare_hmm_sequences(df, X_scaled):
    """Prepare sequences and lengths for hmmlearn."""
    lengths = df.groupby('person_id').size().to_numpy()
    return X_scaled, lengths


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


def initialize_hmm_with_clinical_thresholds(scaler, n_features, n_components=4):
    """
    States:
    0: Acute
    1: Recovery
    2: Weaning (Ready for SBT: PEEP<=8, FiO2<=40%)
    3: Liberation
    """
    print("Initializing HMM using PROMIZING Protocol cutoffs...")
    
    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, 
                        init_params="", params='st', verbose=False, random_state=42)
    
    # Scaler was fit on ALL features, so we need to get stats for base features
    mus = scaler.mean_
    sigmas = scaler.scale_
    
    def get_z(feature_idx, raw_value):
        return (raw_value - mus[feature_idx]) / sigmas[feature_idx]

    # --- CLINICAL CENTROIDS (Raw Values) ---
    # Based on PROMIZING / Standard Weaning Phases
    # FiO2 is in % (0-100)
    
    # State 0: ACUTE (Instability)
    # High PEEP, High FiO2, High Peak, Variable BP (often hypotensive or pressor dependent)
    C_ACUTE = {
        IDX_PEEP: 12.0,  # >10
        IDX_PEAK: 35.0,  # >30
        IDX_SBP:  95.0, # Hypotensive risk or controlled
        IDX_FIO2: 75.0   # >50%
    }
    
    # State 1: RECOVERY (Stabilization)
    # Better but not ready to wean.
    C_REC = {
        IDX_PEEP: 8.0,  # 8-10
        IDX_PEAK: 26.0,
        IDX_SBP:  115.0,
        IDX_FIO2: 55.0   # 40-50%
    }
    
    # State 2: WEANING (Weaning Criteria Met)
    # PEEP <= 8, FiO2 <= 40% (0.4)
    C_WEAN = {
        IDX_PEEP: 6.0,   # <=8
        IDX_PEAK: 22.0,
        IDX_SBP:  125.0,
        IDX_FIO2: 40.0   # <=40%
    }
    
    # State 3: LIBERATION (Minimal / Extubation Ready)
    # Lowest settings
    C_LIB = {
        IDX_PEEP: 5.0,   # 5
        IDX_PEAK: 18.0,
        IDX_SBP:  125.0,
        IDX_FIO2: 28.0   # ~21-30%
    }
    
    means_init = np.zeros((n_components, n_features))
    
   # Fill base feature means
    for idx, feature_idx in enumerate([IDX_PEEP, IDX_PEAK, IDX_SBP, IDX_FIO2]):
        means_init[0, feature_idx] = get_z(feature_idx, C_ACUTE[feature_idx])
        means_init[1, feature_idx] = get_z(feature_idx, C_REC[feature_idx])
        means_init[2, feature_idx] = get_z(feature_idx, C_WEAN[feature_idx])
        means_init[3, feature_idx] = get_z(feature_idx, C_LIB[feature_idx])
        
    model.means_ = means_init

    # Use 1.0 for base features (1 std dev spread), larger for trends
    covars_init = np.ones((n_components, n_features))
    # Base features: moderate spread
    covars_init[:, :4] = 1.0
    # Trend features: allow more variance (trends are noisy)
    covars_init[:, 4:] = 2.0
    model.covars_ = covars_init
    
    # Transition Matrix (Left-to-right dominance with relapse allowed)
    trans_init = np.array([
        [0.85, 0.10, 0.05, 0.00],
        [0.05, 0.80, 0.15, 0.00],
        [0.02, 0.08, 0.80, 0.10],
        [0.01, 0.02, 0.07, 0.90],
    ])
    trans_init = trans_init / trans_init.sum(axis=1, keepdims=True)
    model.transmat_ = trans_init
    
    model.startprob_ = np.array([0.2, 0.4, 0.3, 0.1])
    
    return model


def predict_next_state_probs(model, current_state_idx):
    """Predict P(S_{t+1} | S_t = current_state_idx)."""
    return model.transmat_[current_state_idx]


def visualize_patient_trajectory(model, patient_data, X_patient, state_map, filename="hmm_visualization.png"):
    """
    Visualization using TRUE FILTERED inference (forward algorithm).
    """
    time = patient_data['measure_time'].values
    
    # 1. Filtered Inference (True bedside)
    filtered_probs = forward_algorithm(model, X_patient)
    current_states = np.argmax(filtered_probs, axis=1)
    
    # 2. True Forecast: P(S_{t+1} | x_0:t) = P(S_t | x_0:t) @ T
    forecast_probs = filtered_probs @ model.transmat_
    forecast_states = np.argmax(forecast_probs, axis=1)
    
    # Plotting
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


def main():
    print('main function started...')
    data_path = "clinical_data/data_v2_max_72_h.parquet" # must be parquet file
    
    # 1. Load and Preprocess
    df = load_and_preprocess(data_path)
    
    # 2. Add Trend Features
    df = make_trend_features(df)
    
    # 3. Train/Test Split by person_id
    person_ids = df['person_id'].unique()
    random.seed(42)
    random.shuffle(person_ids)
    
    split_idx = int(len(person_ids) * 0.8)
    train_ids = person_ids[:split_idx]
    test_ids = person_ids[split_idx:]
    
    print(f"Total subjects: {len(person_ids)}")
    print(f"Training subjects: {len(train_ids)}")
    print(f"Testing subjects: {len(test_ids)}")
    
    df_train = df[df['person_id'].isin(train_ids)].copy()
    df_test = df[df['person_id'].isin(test_ids)].copy()
    
    # 4. Standardization (fit on train only)
    print("Standardizing features...")
    scaler = StandardScaler()
    df_train[FEATURES] = scaler.fit_transform(df_train[FEATURES])
    df_test[FEATURES] = scaler.transform(df_test[FEATURES])
    
    X_train, lengths_train = prepare_hmm_sequences(df_train, df_train[FEATURES].values)
    X_test, lengths_test = prepare_hmm_sequences(df_test, df_test[FEATURES].values)
    
    X_train = np.asarray(X_train, dtype=np.float64, order="C")
    X_test  = np.asarray(X_test, dtype=np.float64, order="C")

    # DEBUG: Check data distribution vs clinical centroids
    print("\nData distribution (raw, before scaling):")
    raw_train = scaler.inverse_transform(df_train[FEATURES].values)
    for i, feat in enumerate(FEATURES[:4]):
        print(f"  {feat}: mean={raw_train[:, i].mean():.1f}, std={raw_train[:, i].std():.1f}, "
                f"min={raw_train[:, i].min():.1f}, max={raw_train[:, i].max():.1f}")

    # 5. Architecture & Init
    n_features = len(FEATURES)
    model = initialize_hmm_with_clinical_thresholds(scaler, n_features)
    
    # 6. Fit
    print("Fitting model...")
    try:
        model.fit(X_train, lengths_train)
        print("Model converged:", model.monitor_.converged)
    except Exception as e:
        print(f"Error during fitting: {e}")
        sys.exit(1)
        
    print(f"\nLearned Means (Components x {n_features} Features):")
    print(f"Features: {FEATURES}")
    print(model.means_)
    
    print("\nLearned Transition Matrix:")
    print(model.transmat_)
    
    # 7. Evaluation on Test Set
    log_likelihood = model.score(X_test, lengths_test)
    print(f"\nTest Set Log-Likelihood: {log_likelihood:.4f}")
    
    print("\nDecoding test set states (using filtered inference)...")
    
    # Use forward algorithm per patient instead of Viterbi
    all_filtered_states = []
    all_forecast_states = []
    
    idx = 0
    for length in lengths_test:
        X_patient = X_test[idx:idx + length]
        
        # Filtered inference: P(S_t | x_0:t)
        filtered_probs = forward_algorithm(model, X_patient)
        filtered_states = np.argmax(filtered_probs, axis=1)
        
        # Forecast: P(S_{t+1} | x_0:t) = P(S_t | x_0:t) @ T
        forecast_probs = filtered_probs @ model.transmat_
        forecast_states = np.argmax(forecast_probs, axis=1)
        
        all_filtered_states.extend(filtered_states)
        all_forecast_states.extend(forecast_states)
        
        idx += length
    
    df_test['predicted_state'] = all_filtered_states
    df_test['forecast_state_tplus1'] = all_forecast_states
    
    state_map = {0: 'Acute', 1: 'Recovery', 2: 'Weaning', 3: 'Liberation'}
    df_test['state_label'] = df_test['predicted_state'].map(state_map)

    print("\nState distribution in test set:")
    print(df_test['predicted_state'].value_counts(normalize=True))
    
    print("\nMean raw values per state (PEEP, FiO2):")
    for state in range(4):
        mask = df_test['predicted_state'] == state
        if mask.sum() > 0:
            # Inverse transform to get raw values
            raw = scaler.inverse_transform(df_test.loc[mask, FEATURES].values)
            print(f"  {state_map[state]}: PEEP={raw[:, IDX_PEEP].mean():.1f}, FiO2={raw[:, IDX_FIO2].mean():.1f}")

    
    print("\nLearned Covariances (diagonal, first 4 features):")
    for i in range(4):
        print(f"  State {i}: {model.covars_[i, :4]}")

    # Calculate 1-hour forecast accuracy on test set
    # Compare forecast at t with FILTERED state at t+1
    print("\nCalculating 1-hour forecast accuracy on test set...")
    forecast_matches = []
    for person_id in test_ids:
        person_preds = df_test[df_test['person_id'] == person_id].sort_values('measure_time')
        if len(person_preds) < 2:
            continue
        for i in range(len(person_preds) - 1):
            forecast = person_preds.iloc[i]['forecast_state_tplus1']
            actual_next = person_preds.iloc[i + 1]['predicted_state']
            forecast_matches.append(forecast == actual_next)
    
    if forecast_matches:
        accuracy = np.mean(forecast_matches)
        print(f"1-hour forecast accuracy (test set): {accuracy:.1%}")
    else:
        print("Not enough data to calculate forecast accuracy.")
    
    print("\nSample Output (first 20 rows of test set):")
    print(df_test[['person_id', 'measure_time', 'peep_mean', 'peak_mean', 'state_label']].head(20))
    
    # 8. Save Model & Scaler
    print("\nSaving model and scaler...")
    with open('hmm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 9. Visualization Demo on a Test Patient
    print("\nForecasting Demo:")
    sample_patient = test_ids[0]
    patient_data = df_test[df_test['person_id'] == sample_patient]
    last_state = patient_data.iloc[-1]['predicted_state']
    
    next_probs = predict_next_state_probs(model, last_state)
    
    print(f"Patient {sample_patient} currently in state: {state_map[last_state]} ({last_state})")
    print(f"Forecast for t+1 (1 hour):")
    for s_idx, prob in enumerate(next_probs):
        print(f"  State {state_map[s_idx]}: {prob:.4f}")
    
    print(f"\nVisualizing Test Patient {sample_patient}...")
    X_patient = patient_data[FEATURES].values
    visualize_patient_trajectory(model, patient_data, X_patient, state_map, filename="hmm_test_patient_visualization.png")
    
    # Save test predictions
    output_file = "hmm_test_predictions.csv"
    print(f"\nSaving test predictions to {output_file}...")
    df_test.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()

