"""
NOTE: please don't use, this is not yet ready or debugged
HMM Prototype with Vent Mode Labels (Dual Emission Channels)

This prototype implements a Hidden Markov Model pipeline for mechanical ventilation 
phase classification and 1-hour forecasting using:
  1) Continuous physiologic features (Gaussian emissions)
  2) Imperfect observed vent_mode labels (categorical/noisy label emissions)

The 5-state latent space represents clinical ventilation phases:
  0: Acute
  1: Acute and Recovery
  2: Recovery and Weaning
  3: Weaning
  4: Liberation

Key Design Decisions:
- Emissions (means/covars) are FROZEN to clinically meaningful centroids
- Only transitions and start probabilities are learned from data
- Inference uses manual forward algorithm combining both evidence types
- "Unclassified / Artefacts" labels treated as uninformative (not a latent state)

"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp
import sys
import matplotlib.pyplot as plt
import pickle
import random
import warnings

# --- 1. CONFIGURATION ---
# Base features (vitals from PROMIZING Protocol)
BASE_FEATURES = ['peep_mean', 'peak_mean', 'sbp_mean', 'fio2_mean']

# Trend features will be added dynamically
TREND_SOURCE_FEATURES = ['peep_mean', 'peak_mean', 'sbp_mean', 'fio2_mean']

# Number of latent states (5-state model)
N_STATES = 5

# State names for the 5-state model
STATE_NAMES = {
    0: 'Acute',
    1: 'Acute and Recovery',
    2: 'Recovery and Weaning',
    3: 'Weaning',
    4: 'Liberation'
}

# Valid vent_mode tokens (must match STATE_NAMES values exactly)
VALID_VENT_MODE_TOKENS = set(STATE_NAMES.values())

# Token to state mapping
TOKEN_TO_STATE = {v: k for k, v in STATE_NAMES.items()}

# Indices for the "Driver" features (base features only)
IDX_PEEP = 0
IDX_PEAK = 1
IDX_SBP = 2
IDX_FIO2 = 3

# Label strength hyperparameter (how much weight to give vent_mode labels)
# Default 1.5 = somewhat strong but not absolute
LABEL_STRENGTH = 1.5

# Physiological bounds for clipping outliers
PHYSIOLOGICAL_BOUNDS = {
    'peep_mean': (0, 25),      # PEEP: 0-25 cmH2O
    'peak_mean': (5, 60),      # Peak pressure: 5-60 cmH2O
    'sbp_mean': (40, 250),     # SBP: 40-250 mmHg
    'fio2_mean': (21, 100),    # FiO2: 21-100% (percent scale)
}


def get_all_feature_names():
    """Returns the full list of features including trends."""
    features = BASE_FEATURES.copy()
    for feat in TREND_SOURCE_FEATURES:
        features.append(f"{feat}_delta1h")
        features.append(f"{feat}_slope4h")
    return features


FEATURES = get_all_feature_names()  # Will be: base + 8 trend features = 12 features


def load_and_preprocess(filepath):
    """
    Load data, impute missing values, handle FiO2 scaling, and clip outliers.
    
    Args:
        filepath: Path to the parquet file
        
    Returns:
        DataFrame with preprocessed data including vent_mode column
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        if 'person_id' not in df.columns or 'measure_time' not in df.columns:
            # Try resetting index if columns are in the index
            df = df.reset_index()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

    print("Data loaded. Shape:", df.shape)

    # Filter for selected features + identifiers + vent_mode
    cols_to_keep = ['person_id', 'measure_time'] + BASE_FEATURES
    if 'vent_mode' in df.columns:
        cols_to_keep.append('vent_mode')
    else:
        print("Warning: 'vent_mode' column not found. Will create with NaN values.")
    
    # Check if columns exist
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        # If vent_mode is the only missing, that's okay
        if missing_cols == ['vent_mode']:
            cols_to_keep.remove('vent_mode')
        else:
            print(f"Error: Missing columns in dataset: {missing_cols}")
            print("Available columns:", df.columns.tolist())
            sys.exit(1)

    df_subset = df[cols_to_keep].copy()
    
    # Add vent_mode if it wasn't present
    if 'vent_mode' not in df_subset.columns:
        df_subset['vent_mode'] = np.nan

    # Ensure measure_time is datetime (timezone-aware UTC if possible)
    df_subset['measure_time'] = pd.to_datetime(df_subset['measure_time'], utc=True, errors='coerce')

    # Sort by person_id and time
    df_subset.sort_values(by=['person_id', 'measure_time'], inplace=True)
    
    # Handle FiO2 scaling: determine current scale
    fio2_max = df_subset['fio2_mean'].max()
    if fio2_max <= 1.5:
        # Data is in 0-1 scale, convert to 21-100 percent
        print("FiO2 detected in 0-1 scale. Converting to 21-100%...")
        df_subset['fio2_mean'] = df_subset['fio2_mean'] * 100
        # fio2 of 0.21 -> 21%, fio2 of 1.0 -> 100%
    else:
        # Already in percent scale (21-100)
        print(f"FiO2 appears to be in percent scale (max={fio2_max:.1f}). Keeping as-is.")
    
    # Clip physiologic outliers
    print("Clipping outliers...")
    for col, (low, high) in PHYSIOLOGICAL_BOUNDS.items():
        if col in df_subset.columns:
            n_clipped = ((df_subset[col] < low) | (df_subset[col] > high)).sum()
            df_subset[col] = df_subset[col].clip(low, high)
            if n_clipped > 0:
                print(f"  {col}: clipped {n_clipped} values to [{low}, {high}]")
    
    # Imputation: Forward Fill -> Backward Fill per person for continuous features
    print("Handling missing values...")
    df_subset[BASE_FEATURES] = df_subset.groupby('person_id')[BASE_FEATURES].ffill().bfill()
    
    # Drop rows that are still NaN in continuous features
    initial_rows = len(df_subset)
    df_subset.dropna(subset=BASE_FEATURES, inplace=True)
    rows_dropped = initial_rows - len(df_subset)
    if rows_dropped > 0:
        print(f"  Dropped {rows_dropped} rows with remaining NaN in continuous features.")
    
    # Keep vent_mode as-is (NaNs allowed)
    
    print("\nData quality check:")
    for col in BASE_FEATURES:
        print(f"{col}: min={df_subset[col].min():.1f}, max={df_subset[col].max():.1f}, "
              f"mean={df_subset[col].mean():.1f}, NaN={df_subset[col].isna().sum()}")
    
    # Vent mode summary
    n_vent_mode_nan = df_subset['vent_mode'].isna().sum()
    print(f"vent_mode: {len(df_subset) - n_vent_mode_nan} non-null, {n_vent_mode_nan} NaN")

    return df_subset


def make_trend_features(df):
    """
    Engineer trend features for HMM:
    - 1-hour delta: diff(1)
    - 4-hour rolling slope: diff(4) / 4
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with added trend features
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


def parse_vent_mode(raw_string):
    """
    Parse a vent_mode string into a set of valid state tokens.
    
    - If raw is NaN/None/empty => return empty set (uninformative)
    - Split by "&", strip whitespace
    - Remove "Unclassified / Artefacts" tokens
    - Return set of remaining valid tokens
    
    Args:
        raw_string: Raw vent_mode value (string or NaN)
        
    Returns:
        set of valid token strings
    """
    if pd.isna(raw_string) or raw_string is None or str(raw_string).strip() == '':
        return set()
    
    # Split by "&" and strip whitespace
    tokens = [t.strip() for t in str(raw_string).split('&')]
    
    # Filter out artefacts and empty tokens
    tokens = [t for t in tokens if t and t != "Unclassified / Artefacts"]
    
    # Validate tokens against known states
    valid_tokens = set()
    for t in tokens:
        if t in VALID_VENT_MODE_TOKENS:
            valid_tokens.add(t)
    
    return valid_tokens


def build_log_py(df, label_strength=LABEL_STRENGTH):
    """
    Build the label likelihood matrix log_py[t, k] for all rows.
    
    For each row, determines P(y_t | state=k) based on vent_mode:
    - If token set is empty: uniform distribution (all states equally likely)
    - If single valid token matching state k0: P(y|k0)=0.90, others share 0.10
    - If multiple valid tokens: distribute 0.90 across included states
    
    Args:
        df: DataFrame with 'vent_mode' column
        label_strength: Multiplier for label influence (default 1.5)
        
    Returns:
        log_py: (T, N_STATES) array of log likelihoods
        parse_stats: dict with parsing statistics
    """
    T = len(df)
    log_py = np.zeros((T, N_STATES))
    
    # Parsing statistics
    parse_stats = {
        'nan_count': 0,
        'artefact_only_count': 0,
        'single_token_count': 0,
        'multi_token_count': 0,
        'unknown_token_count': 0
    }
    
    eps = 1e-12
    log_uniform = np.log(1.0 / N_STATES)
    
    for i, raw_vent_mode in enumerate(df['vent_mode'].values):
        # Check if original was NaN
        was_nan = pd.isna(raw_vent_mode)
        
        tokens = parse_vent_mode(raw_vent_mode)
        
        if was_nan:
            parse_stats['nan_count'] += 1
        
        if len(tokens) == 0:
            # Uninformative: uniform distribution
            if not was_nan:
                # Was non-NaN but parsed to empty (artefact only or unknown)
                if raw_vent_mode and str(raw_vent_mode).strip():
                    # Check if it contained only artefacts
                    raw_tokens = [t.strip() for t in str(raw_vent_mode).split('&')]
                    artefact_tokens = [t for t in raw_tokens if t == "Unclassified / Artefacts"]
                    if artefact_tokens and len(artefact_tokens) == len([t for t in raw_tokens if t]):
                        parse_stats['artefact_only_count'] += 1
                    else:
                        parse_stats['unknown_token_count'] += 1
            
            log_py[i, :] = log_uniform
            
        elif len(tokens) == 1:
            # Single valid token
            parse_stats['single_token_count'] += 1
            token = list(tokens)[0]
            k0 = TOKEN_TO_STATE[token]
            
            # High prob for matching state, low for others
            probs = np.full(N_STATES, 0.10 / (N_STATES - 1))
            probs[k0] = 0.90
            log_py[i, :] = np.log(np.clip(probs, eps, 1.0))
            
        else:
            # Multiple valid tokens
            parse_stats['multi_token_count'] += 1
            n_included = len(tokens)
            n_excluded = N_STATES - n_included
            
            # Distribute 0.90 across included states, 0.10 across excluded
            probs = np.zeros(N_STATES)
            for token in tokens:
                k = TOKEN_TO_STATE[token]
                probs[k] = 0.90 / n_included
            
            if n_excluded > 0:
                for k in range(N_STATES):
                    if STATE_NAMES[k] not in tokens:
                        probs[k] = 0.10 / n_excluded
            
            # Normalize to ensure probabilities sum to 1.0 (handles edge case where all tokens present)
            probs = np.clip(probs, eps, None)
            probs = probs / probs.sum()
            log_py[i, :] = np.log(probs)
    
    # Apply label strength multiplier
    log_py *= label_strength
    
    print("\nVent mode parsing statistics:")
    print(f"  NaN: {parse_stats['nan_count']}")
    print(f"  Artefact-only: {parse_stats['artefact_only_count']}")
    print(f"  Single token: {parse_stats['single_token_count']}")
    print(f"  Multi-token: {parse_stats['multi_token_count']}")
    print(f"  Unknown token: {parse_stats['unknown_token_count']}")
    
    return log_py, parse_stats


def prepare_hmm_sequences(df, X_scaled):
    """Prepare sequences and lengths for hmmlearn."""
    lengths = df.groupby('person_id').size().to_numpy()
    return X_scaled, lengths


def initialize_hmm_5states(scaler, n_features, n_components=N_STATES):
    """
    Initialize a 5-state HMM with clinically meaningful emission parameters.
    
    States (ordered from most acute to liberated):
      0: Acute - High support (FiO2 > 50%, PEEP > 10)
      1: Acute and Recovery - Transitioning, still elevated
      2: Recovery and Weaning - Improving, approaching weaning criteria
      3: Weaning - Meets SBT criteria (PEEP <= 8, FiO2 <= 40%)
      4: Liberation - Minimal support, ready for extubation
    
    Emissions (means/covars) are FROZEN.
    Only transitions and start probabilities will be learned.
    
    Args:
        scaler: Fitted StandardScaler
        n_features: Total number of features
        n_components: Number of states (should be 5)
        
    Returns:
        Configured GaussianHMM model
    """
    print(f"Initializing {n_components}-state HMM with clinical thresholds...")
    
    model = GaussianHMM(
        n_components=n_components, 
        covariance_type="diag", 
        n_iter=100,
        init_params="",      # Don't auto-initialize
        params='st',         # Only learn start probs and transitions
        verbose=False, 
        random_state=42
    )
    
    # Get scaler statistics
    mus = scaler.mean_
    sigmas = scaler.scale_
    
    def get_z(feature_idx, raw_value):
        """Convert raw value to z-score."""
        return (raw_value - mus[feature_idx]) / sigmas[feature_idx]

    # --- CLINICAL CENTROIDS (Raw Values) ---
    # Based on PROMIZING Protocol and standard weaning criteria
    # FiO2 is in % (21-100)
    
    # State 0: ACUTE (High instability, high support)
    C_ACUTE = {
        IDX_PEEP: 14.0,   # High PEEP
        IDX_PEAK: 38.0,   # High peak pressure
        IDX_SBP:  95.0,   # Hypotensive or controlled
        IDX_FIO2: 80.0    # High FiO2
    }
    
    # State 1: ACUTE AND RECOVERY (Transitional, improving but elevated support)
    C_ACUTE_REC = {
        IDX_PEEP: 11.0,   # Still elevated
        IDX_PEAK: 32.0,
        IDX_SBP:  105.0,
        IDX_FIO2: 65.0
    }
    
    # State 2: RECOVERY AND WEANING (Approaching weaning thresholds)
    C_REC_WEAN = {
        IDX_PEEP: 9.0,    # Near 8
        IDX_PEAK: 27.0,
        IDX_SBP:  115.0,
        IDX_FIO2: 50.0
    }
    
    # State 3: WEANING (Meets SBT criteria: PEEP <= 8, FiO2 <= 40%)
    C_WEAN = {
        IDX_PEEP: 6.5,    # <= 8
        IDX_PEAK: 22.0,
        IDX_SBP:  120.0,
        IDX_FIO2: 38.0    # <= 40%
    }
    
    # State 4: LIBERATION (Minimal support, extubation ready)
    C_LIB = {
        IDX_PEEP: 5.0,    # PEEP 5
        IDX_PEAK: 18.0,
        IDX_SBP:  120.0,
        IDX_FIO2: 26.0    # Near room air (21%)
    }
    
    centroids = [C_ACUTE, C_ACUTE_REC, C_REC_WEAN, C_WEAN, C_LIB]
    
    # Initialize means
    means_init = np.zeros((n_components, n_features))
    
    for state_idx, centroid in enumerate(centroids):
        for feature_idx in [IDX_PEEP, IDX_PEAK, IDX_SBP, IDX_FIO2]:
            means_init[state_idx, feature_idx] = get_z(feature_idx, centroid[feature_idx])
        
        # Trend features: expect different trends per state
        # Acute: possibly worsening or stable (slightly positive or zero trends)
        # Liberation: stable or improving (slightly negative trends for pressure/FiO2)
        # Leave trends at 0 (neutral) as baseline - scaler will have normalized them
    
    model.means_ = means_init

    # Covariances: fixed values
    # Base features: moderate spread (1.0)
    # Trend features: allow more variance (2.0) since trends may be noisy
    covars_init = np.ones((n_components, n_features))
    covars_init[:, :4] = 1.0   # Base features
    covars_init[:, 4:] = 2.0   # Trend features
    model.covars_ = covars_init
    
    # --- Transition Matrix ---
    # Left-to-right with relapse allowed but discourage long jumps
    # States: 0=Acute, 1=Acute+Rec, 2=Rec+Wean, 3=Wean, 4=Liberation
    
    trans_init = np.array([
        # To:    0      1      2      3      4
        [0.80, 0.15, 0.04, 0.01, 0.00],  # From Acute
        [0.08, 0.75, 0.14, 0.03, 0.00],  # From Acute+Recovery
        [0.02, 0.08, 0.75, 0.13, 0.02],  # From Recovery+Weaning
        [0.01, 0.03, 0.08, 0.78, 0.10],  # From Weaning
        [0.00, 0.01, 0.02, 0.07, 0.90],  # From Liberation
    ])
    
    # Normalize rows
    trans_init = trans_init / trans_init.sum(axis=1, keepdims=True)
    model.transmat_ = trans_init
    
    # Start probabilities: emphasize early states
    # Most ICU admissions start in Acute or Acute+Recovery
    model.startprob_ = np.array([0.30, 0.30, 0.25, 0.10, 0.05])
    
    print("  Clinical centroids (raw values):")
    for i, name in STATE_NAMES.items():
        c = centroids[i]
        print(f"    {name}: PEEP={c[IDX_PEEP]:.1f}, Peak={c[IDX_PEAK]:.1f}, "
              f"SBP={c[IDX_SBP]:.1f}, FiO2={c[IDX_FIO2]:.1f}")
    
    return model


def _logsumexp_axis1(a):
    """Fast logsumexp along axis=1 without scipy overhead."""
    a_max = np.max(a, axis=1, keepdims=True)
    # Handle -inf max (entire row is -inf)
    a_max = np.where(np.isfinite(a_max), a_max, 0)
    return np.log(np.sum(np.exp(a - a_max), axis=1)) + a_max.ravel()


def forward_backward_with_labels(model, X_patient, log_py_patient):
    """
    Vectorized forward-backward algorithm with dual evidence channels.
    
    Args:
        model: GaussianHMM with current parameters
        X_patient: (T, n_features) scaled features
        log_py_patient: (T, N_STATES) log label likelihoods
        
    Returns:
        log_alpha: (T, N_STATES) forward log-probabilities
        log_beta: (T, N_STATES) backward log-probabilities  
        log_emit_total: (T, N_STATES) combined emissions
        log_likelihood: scalar log P(X, Y)
    """
    n_samples = len(X_patient)
    n_components = model.n_components
    eps = 1e-10
    
    log_startprob = np.log(model.startprob_ + eps)
    log_transmat = np.log(model.transmat_ + eps)
    
    # Combined emissions
    log_emit_x = model._compute_log_likelihood(X_patient)
    log_emit_total = log_emit_x + log_py_patient
    
    # Forward pass (vectorized over states)
    log_alpha = np.zeros((n_samples, n_components))
    log_alpha[0] = log_startprob + log_emit_total[0]
    
    for t in range(1, n_samples):
        # log_alpha[t-1, :] is (K,), log_transmat is (K, K)
        # We want: for each k, logsumexp over j of (log_alpha[t-1,j] + log_transmat[j,k])
        # = logsumexp(log_alpha[t-1, :, None] + log_transmat, axis=0)
        log_alpha[t] = log_emit_total[t] + _logsumexp_axis1(
            (log_alpha[t - 1, :, None] + log_transmat).T
        )
    
    # Backward pass (vectorized over states)
    log_beta = np.zeros((n_samples, n_components))
    
    for t in range(n_samples - 2, -1, -1):
        # For each j: logsumexp over k of (log_transmat[j,k] + log_emit[t+1,k] + log_beta[t+1,k])
        log_beta[t] = _logsumexp_axis1(
            log_transmat + log_emit_total[t + 1] + log_beta[t + 1]
        )
    
    log_likelihood = logsumexp(log_alpha[-1])
    
    return log_alpha, log_beta, log_emit_total, log_likelihood


def fit_with_dual_channels(model, df, X_scaled, log_py, lengths, n_iter=50, tol=1e-4):
    """
    Vectorized EM loop that learns startprob and transmat using both
    Gaussian emissions and vent_mode label likelihoods.
    
    Emissions (means/covars) remain FROZEN.
    
    Args:
        model: Initialized GaussianHMM
        df: DataFrame with person_id column
        X_scaled: Scaled feature array
        log_py: Label likelihood array
        lengths: Sequence lengths per patient
        n_iter: Maximum EM iterations
        tol: Convergence tolerance for log-likelihood
        
    Returns:
        model: Updated model with learned startprob/transmat
        history: List of log-likelihoods per iteration
    """
    n_components = model.n_components
    eps = 1e-10
    history = []
    prev_ll = -np.inf
    
    # Pre-compute patient indices once (avoid repeated groupby)
    patient_indices = []
    for person_id, group in df.groupby('person_id', sort=False):
        positions = np.array([df.index.get_loc(i) for i in group.index])
        if len(positions) > 0:
            patient_indices.append(positions)
    
    print(f"Running dual-channel EM (max {n_iter} iterations, {len(patient_indices)} patients)...")
    
    for iteration in range(n_iter):
        # E-step: accumulate expected counts
        total_start_counts = np.zeros(n_components)
        total_trans_counts = np.zeros((n_components, n_components))
        total_ll = 0.0
        
        log_transmat = np.log(model.transmat_ + eps)
        
        # Process each patient sequence
        for positions in patient_indices:
            X_patient = X_scaled[positions]
            log_py_patient = log_py[positions]
            T = len(X_patient)
            
            # Forward-backward
            log_alpha, log_beta, log_emit_total, ll = forward_backward_with_labels(
                model, X_patient, log_py_patient
            )
            total_ll += ll
            
            # Posterior: gamma[t,k] = P(S_t=k | X, Y)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            
            # Start counts: gamma[0]
            total_start_counts += gamma[0]
            
            # Transition counts (VECTORIZED): xi[t,j,k] = P(S_t=j, S_{t+1}=k | X, Y)
            if T > 1:
                # Compute all xi for t=0..T-2 at once
                # xi[t,j,k] = exp(log_alpha[t,j] + log_transmat[j,k] + log_emit[t+1,k] + log_beta[t+1,k] - ll)
                # Shape: (T-1, K, K)
                log_xi = (
                    log_alpha[:-1, :, None] +          # (T-1, K, 1)
                    log_transmat[None, :, :] +         # (1, K, K)
                    log_emit_total[1:, None, :] +      # (T-1, 1, K)
                    log_beta[1:, None, :] -            # (T-1, 1, K)
                    ll
                )
                # Sum over time to get expected transition counts
                total_trans_counts += np.exp(log_xi).sum(axis=0)
        
        history.append(total_ll)
        
        # Check convergence
        if iteration > 0 and abs(total_ll - prev_ll) < tol:
            print(f"  Converged at iteration {iteration + 1} (ΔLL={total_ll - prev_ll:.6f})")
            break
        prev_ll = total_ll
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration + 1}: LL={total_ll:.2f}")
        
        # M-step: update startprob and transmat
        total_start_counts = np.clip(total_start_counts, eps, None)
        model.startprob_ = total_start_counts / total_start_counts.sum()
        
        total_trans_counts = np.clip(total_trans_counts, eps, None)
        model.transmat_ = total_trans_counts / total_trans_counts.sum(axis=1, keepdims=True)
    
    print(f"  Final LL: {history[-1]:.2f} after {len(history)} iterations")
    return model, history


def forward_filter_with_labels(model, X_patient, log_py_patient):
    """
    Compute filtered posterior P(S_t | x_0:t, y_0:t) using forward algorithm
    with dual evidence channels (Gaussian emissions + label likelihoods).
    
    This is the core inference routine combining:
      log_emit_total[t,k] = log P(x_t | state=k) + log P(y_t | state=k)
    
    Args:
        model: Trained GaussianHMM
        X_patient: (T, n_features) scaled feature array for one patient
        log_py_patient: (T, N_STATES) log label likelihoods for one patient
        
    Returns:
        filtered_probs: (T, N_STATES) filtered posterior probabilities
    """
    n_samples = len(X_patient)
    n_components = model.n_components
    
    # Log parameters (add small epsilon for numerical stability)
    eps = 1e-10
    log_startprob = np.log(model.startprob_ + eps)
    log_transmat = np.log(model.transmat_ + eps)
    
    # Log emission probabilities from Gaussian model
    log_emit_x = model._compute_log_likelihood(X_patient)  # (T, n_components)
    
    # Combine evidence: Gaussian + label likelihoods
    log_emit_total = log_emit_x + log_py_patient
    
    # Forward pass in log space
    log_alpha = np.zeros((n_samples, n_components))
    
    # Initialize: alpha_0 = pi * P(x_0, y_0 | S_0)
    log_alpha[0] = log_startprob + log_emit_total[0]
    
    for t in range(1, n_samples):
        # log_alpha[t, k] = log( sum_j [alpha[t-1, j] * A[j,k]] ) + log P(x_t, y_t | k)
        for k in range(n_components):
            log_alpha[t, k] = log_emit_total[t, k] + logsumexp(
                log_alpha[t - 1, :] + log_transmat[:, k]
            )
    
    # Normalize to get filtered probabilities
    log_normalizer = logsumexp(log_alpha, axis=1, keepdims=True)
    filtered_probs = np.exp(log_alpha - log_normalizer)
    
    return filtered_probs


def run_inference_on_dataset(model, df, X_scaled, log_py):
    """
    Run filtered inference for all patients in a dataset.
    
    Args:
        model: Trained GaussianHMM
        df: DataFrame with patient data (must have person_id)
        X_scaled: Scaled feature array aligned with df
        log_py: Label likelihood array aligned with df
        
    Returns:
        results: dict with arrays of results aligned with df rows
    """
    all_filtered_probs = np.zeros((len(df), N_STATES))
    all_forecast_probs = np.zeros((len(df), N_STATES))
    
    # Use groupby with sort=False to preserve original order and get correct indices
    # This is robust to non-contiguous patient blocks and arbitrary ordering
    for person_id, group in df.groupby('person_id', sort=False):
        # Get the actual row indices for this patient
        ix = group.index.to_numpy()
        
        # Map DataFrame indices to array positions (handles non-default index)
        # Use iloc-style positions for array slicing
        positions = np.array([df.index.get_loc(i) for i in ix])
        
        X_patient = X_scaled[positions]
        log_py_patient = log_py[positions]
        
        # Filtered inference: P(S_t | x_0:t, y_0:t)
        filtered_probs = forward_filter_with_labels(model, X_patient, log_py_patient)
        
        # Forecast: P(S_{t+1} | x_0:t, y_0:t) = P(S_t | x_0:t, y_0:t) @ T
        forecast_probs = filtered_probs @ model.transmat_
        
        all_filtered_probs[positions] = filtered_probs
        all_forecast_probs[positions] = forecast_probs
    
    # Get predictions
    predicted_states = np.argmax(all_filtered_probs, axis=1)
    forecast_states = np.argmax(all_forecast_probs, axis=1)
    pmax_current = np.max(all_filtered_probs, axis=1)
    pmax_forecast = np.max(all_forecast_probs, axis=1)
    
    results = {
        'filtered_probs': all_filtered_probs,
        'forecast_probs': all_forecast_probs,
        'predicted_state': predicted_states,
        'forecast_state_tplus1': forecast_states,
        'pmax_current': pmax_current,
        'pmax_forecast': pmax_forecast
    }
    
    return results


def visualize_patient_trajectory(model, patient_data, X_patient, log_py_patient, 
                                  state_map, filename="hmm_visualization.png"):
    """
    Visualization using filtered inference with dual evidence channels.
    
    Args:
        model: Trained HMM
        patient_data: DataFrame slice for the patient
        X_patient: Scaled features for patient
        log_py_patient: Label likelihoods for patient
        state_map: Dict mapping state indices to names
        filename: Output PNG filename
    """
    time = patient_data['measure_time'].values
    
    # Filtered Inference with labels
    filtered_probs = forward_filter_with_labels(model, X_patient, log_py_patient)
    current_states = np.argmax(filtered_probs, axis=1)
    
    # Forecast: P(S_{t+1} | x_0:t, y_0:t)
    forecast_probs = filtered_probs @ model.transmat_
    forecast_states = np.argmax(forecast_probs, axis=1)
    
    # Plotting
    n_states = len(state_map)
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), constrained_layout=True)
    
    # Plot 1: Current State (Filtered)
    ax1 = axes[0, 0]
    ax1.plot(time, current_states, label='Current State (Filtered)', 
             marker='o', linestyle='-', color='blue', alpha=0.7)
    ax1.set_yticks(range(n_states))
    ax1.set_yticklabels([state_map[i] for i in range(n_states)], fontsize=8)
    ax1.set_title('Current State (Filtered P(S_t|x,y))', fontsize=10)
    ax1.set_xlabel('Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast State
    ax2 = axes[0, 1]
    ax2.plot(time, forecast_states, label='Forecast (t+1)', 
             marker='x', linestyle='--', color='orange', alpha=0.7)
    ax2.set_yticks(range(n_states))
    ax2.set_yticklabels([state_map[i] for i in range(n_states)], fontsize=8)
    ax2.set_title('Forecasted State (t+1)', fontsize=10)
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overlay
    ax3 = axes[0, 2]
    ax3.plot(time, current_states, label='Current (Filtered)', 
             color='blue', alpha=0.6, linewidth=2)
    ax3.plot(time, forecast_states, label='Forecast', 
             color='orange', alpha=0.6, linestyle='--', linewidth=2)
    ax3.set_yticks(range(n_states))
    ax3.set_yticklabels([state_map[i] for i in range(n_states)], fontsize=8)
    ax3.set_title('Overlay: Current vs Forecast', fontsize=10)
    ax3.set_xlabel('Time')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Current State Confidence
    ax4 = axes[1, 0]
    confidence_current = np.max(filtered_probs, axis=1)
    sc1 = ax4.scatter(time, confidence_current, c=confidence_current, 
                      cmap='viridis', vmin=0.2, vmax=1, s=30)
    ax4.set_title('Confidence (Current Classification)', fontsize=10)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Max Probability')
    ax4.set_ylim(-0.05, 1.05)
    plt.colorbar(sc1, ax=ax4)
    
    # Plot 5: Forecast Confidence
    ax5 = axes[1, 1]
    confidence_forecast = np.max(forecast_probs, axis=1)
    sc2 = ax5.scatter(time, confidence_forecast, c=confidence_forecast, 
                      cmap='viridis', vmin=0.2, vmax=1, s=30)
    ax5.set_title('Confidence (Forecast)', fontsize=10)
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Max Probability')
    ax5.set_ylim(-0.05, 1.05)
    plt.colorbar(sc2, ax=ax5)
    
    # Plot 6: Vent mode labels
    ax6 = axes[1, 2]
    vent_modes = patient_data['vent_mode'].values
    # Encode vent_mode for visualization
    y_labels = []
    for vm in vent_modes:
        tokens = parse_vent_mode(vm)
        if len(tokens) == 0:
            y_labels.append(-1)  # Uninformative
        elif len(tokens) == 1:
            y_labels.append(TOKEN_TO_STATE[list(tokens)[0]])
        else:
            # Multi-token: use average
            y_labels.append(np.mean([TOKEN_TO_STATE[t] for t in tokens]))
    
    ax6.scatter(time, y_labels, c='green', alpha=0.6, s=30, label='Vent Mode Label')
    ax6.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5, label='Uninformative')
    ax6.set_yticks(list(range(n_states)) + [-1])
    ax6.set_yticklabels([state_map.get(i, 'NA') for i in range(n_states)] + ['NA'], fontsize=8)
    ax6.set_title('Observed Vent Mode Labels', fontsize=10)
    ax6.set_xlabel('Time')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=150)
    print(f"Visualization saved to {filename}")
    plt.close()


def visualize_trajectory_simple(patient_data, predicted_states, forecast_states,
                                 pmax_current, pmax_forecast, state_map,
                                 filename="hmm_simple_visualization.png"):
    """
    Simplified trajectory visualization showing states and confidence.
    
    Args:
        patient_data: DataFrame slice for patient
        predicted_states: Array of current state predictions
        forecast_states: Array of forecast state predictions  
        pmax_current: Array of current confidence scores
        pmax_forecast: Array of forecast confidence scores
        state_map: Dict mapping state indices to names
        filename: Output PNG filename
    """
    time = patient_data['measure_time'].values
    n_states = len(state_map)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    
    # Plot 1: States
    ax1 = axes[0]
    ax1.plot(time, predicted_states, label='Current State', 
             marker='o', color='blue', linewidth=2, markersize=4)
    ax1.plot(time, forecast_states, label='Forecast (t+1)', 
             marker='x', linestyle='--', color='orange', linewidth=2, markersize=4)
    ax1.set_yticks(range(n_states))
    ax1.set_yticklabels([state_map[i] for i in range(n_states)], fontsize=9)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('State')
    ax1.set_title('Predicted State Trajectory')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Confidence
    ax2 = axes[1]
    ax2.plot(time, pmax_current, label='Current Confidence', 
             color='blue', linewidth=2, alpha=0.7)
    ax2.plot(time, pmax_forecast, label='Forecast Confidence', 
             color='orange', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Max Probability')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Classification Confidence')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
    
    plt.savefig(filename, dpi=150)
    print(f"Simple visualization saved to {filename}")
    plt.close()


def main():
    """Main execution function."""
    print('='*80)
    print('HMM Prototype with Vent Mode Labels (5-State Model)')
    print('='*80)
    
    data_path = "clinical_data/data_v3_max_72_h.parquet"
    
    # 1. Load and Preprocess
    df = load_and_preprocess(data_path)
    
    # 2. Add Trend Features
    df = make_trend_features(df)
    
    # 3. Train/Test Split by person_id
    person_ids = df['person_id'].unique()
    random.seed(42)
    person_ids_list = list(person_ids)
    random.shuffle(person_ids_list)
    
    split_idx = int(len(person_ids_list) * 0.8)
    train_ids = set(person_ids_list[:split_idx])
    test_ids = set(person_ids_list[split_idx:])
    
    print(f"\nTotal subjects: {len(person_ids_list)}")
    print(f"Training subjects: {len(train_ids)}")
    print(f"Testing subjects: {len(test_ids)}")
    
    df_train = df[df['person_id'].isin(train_ids)].copy().reset_index(drop=True)
    df_test = df[df['person_id'].isin(test_ids)].copy().reset_index(drop=True)
    
    # Re-sort by person_id and time after split to ensure contiguity
    df_train = df_train.sort_values(['person_id', 'measure_time']).reset_index(drop=True)
    df_test = df_test.sort_values(['person_id', 'measure_time']).reset_index(drop=True)
    
    print(f"Training rows: {len(df_train)}")
    print(f"Testing rows: {len(df_test)}")
    
    # 4. Standardization (fit on train only)
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_raw = df_train[FEATURES].values
    X_test_raw = df_test[FEATURES].values
    
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Prepare sequences for hmmlearn
    _, lengths_train = prepare_hmm_sequences(df_train, X_train)
    _, lengths_test = prepare_hmm_sequences(df_test, X_test)
    
    X_train = np.asarray(X_train, dtype=np.float64, order="C")
    X_test = np.asarray(X_test, dtype=np.float64, order="C")

    # 5. Build label likelihoods
    print("\nBuilding label likelihoods from vent_mode...")
    log_py_train, train_parse_stats = build_log_py(df_train, label_strength=LABEL_STRENGTH)
    log_py_test, test_parse_stats = build_log_py(df_test, label_strength=LABEL_STRENGTH)

    # 6. Initialize Model
    n_features = len(FEATURES)
    model = initialize_hmm_5states(scaler, n_features, n_components=N_STATES)
    
    # 7. Fit Model using dual-channel EM (learns startprob_ and transmat_ using both emissions)
    print("\nFitting model with dual-channel EM (Gaussian + vent_mode labels)...")
    try:
        model, em_history = fit_with_dual_channels(
            model, df_train, X_train, log_py_train, lengths_train,
            n_iter=50, tol=1e-4
        )
        print(f"Dual-channel EM completed with {len(em_history)} iterations.")
    except Exception as e:
        print(f"Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print(f"\nLearned Transition Matrix:")
    print(np.array2string(model.transmat_, precision=3, suppress_small=True))
    
    print(f"\nLearned Start Probabilities:")
    print(np.array2string(model.startprob_, precision=3, suppress_small=True))
    
    # 8. Inference on Test Set
    print("\nRunning inference on test set...")
    test_results = run_inference_on_dataset(model, df_test, X_test, log_py_test)
    
    # Add predictions to dataframe
    df_test['predicted_state'] = test_results['predicted_state']
    df_test['forecast_state_tplus1'] = test_results['forecast_state_tplus1']
    df_test['pmax_current'] = test_results['pmax_current']
    df_test['pmax_forecast'] = test_results['pmax_forecast']
    df_test['predicted_label'] = df_test['predicted_state'].map(STATE_NAMES)
    df_test['forecast_label_tplus1'] = df_test['forecast_state_tplus1'].map(STATE_NAMES)
    
    # 9. Evaluation
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nState distribution in test set:")
    state_dist = df_test['predicted_state'].value_counts(normalize=True).sort_index()
    for state, prop in state_dist.items():
        print(f"  {STATE_NAMES[state]}: {prop:.1%}")
    
    print("\nMean raw values per predicted state:")
    for state in range(N_STATES):
        mask = df_test['predicted_state'] == state
        if mask.sum() > 0:
            # Use X_test (scaled features) for inverse_transform, not df_test[FEATURES] (raw)
            raw = scaler.inverse_transform(X_test[mask.values])
            print(f"  {STATE_NAMES[state]} (n={mask.sum()}): "
                  f"PEEP={raw[:, IDX_PEEP].mean():.1f}, "
                  f"Peak={raw[:, IDX_PEAK].mean():.1f}, "
                  f"FiO2={raw[:, IDX_FIO2].mean():.1f}")
    
    # Calculate 1-hour forecast accuracy (self-consistency)
    print("\nCalculating 1-hour forecast accuracy (self-consistency)...")
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
        print(f"1-hour forecast self-consistency: {accuracy:.1%} ({sum(forecast_matches)}/{len(forecast_matches)})")
    else:
        print("Not enough data to calculate forecast accuracy.")
    
    # Ground truth evaluation: compare predictions to vent_mode labels
    print("\nGround truth evaluation (vs vent_mode labels)...")
    
    # Single-token exact match accuracy
    single_token_matches = []
    multi_token_membership = []  # prediction ∈ label_set
    unknown_tokens_examples = set()
    
    for idx_row, row in df_test.iterrows():
        tokens = parse_vent_mode(row['vent_mode'])
        pred_state = row['predicted_state']
        pred_label = STATE_NAMES[pred_state]
        
        if len(tokens) == 1:
            # Single unambiguous label: exact match
            ground_truth = list(tokens)[0]
            single_token_matches.append(pred_label == ground_truth)
        elif len(tokens) > 1:
            # Multi-token: set membership (prediction in label set)
            multi_token_membership.append(pred_label in tokens)
        else:
            # Check for unknown tokens (not just NaN/artefact)
            raw = row['vent_mode']
            if pd.notna(raw) and str(raw).strip():
                raw_tokens = [t.strip() for t in str(raw).split('&')]
                for t in raw_tokens:
                    if t and t != "Unclassified / Artefacts" and t not in VALID_VENT_MODE_TOKENS:
                        unknown_tokens_examples.add(t)
    
    if single_token_matches:
        acc = np.mean(single_token_matches)
        print(f"  Single-token exact match: {acc:.1%} ({sum(single_token_matches)}/{len(single_token_matches)})")
    else:
        print("  Single-token exact match: N/A (no single-token labels)")
    
    if multi_token_membership:
        acc = np.mean(multi_token_membership)
        print(f"  Multi-token set membership: {acc:.1%} ({sum(multi_token_membership)}/{len(multi_token_membership)})")
    else:
        print("  Multi-token set membership: N/A (no multi-token labels)")
    
    if unknown_tokens_examples:
        print(f"  Unknown tokens found ({len(unknown_tokens_examples)}): {list(unknown_tokens_examples)[:10]}")
    else:
        print("  No unknown tokens found.")
    
    # 10. Save Outputs
    output_columns = [
        'person_id', 'measure_time', 'vent_mode',
        'predicted_state', 'predicted_label',
        'forecast_state_tplus1', 'forecast_label_tplus1',
        'pmax_current', 'pmax_forecast'
    ]
    
    output_file = "hmm_test_predictions_v3.csv"
    print(f"\nSaving test predictions to {output_file}...")
    df_test[output_columns].to_csv(output_file, index=False)
    
    # Save model and scaler
    print("Saving model and scaler...")
    with open('hmm_model_5state.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler_5state.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 11. Visualization
    print("\nGenerating visualizations...")
    test_ids_list = list(test_ids)
    sample_patient = test_ids_list[0]
    
    patient_mask = df_test['person_id'] == sample_patient
    patient_data = df_test[patient_mask].copy()
    patient_idx = patient_mask.values.nonzero()[0]
    
    X_patient = X_test[patient_idx]
    log_py_patient = log_py_test[patient_idx]
    
    # Detailed visualization
    visualize_patient_trajectory(
        model, patient_data, X_patient, log_py_patient,
        STATE_NAMES, filename="hmm_test_patient_v3_detailed.png"
    )
    
    # Simple visualization
    visualize_trajectory_simple(
        patient_data,
        patient_data['predicted_state'].values,
        patient_data['forecast_state_tplus1'].values,
        patient_data['pmax_current'].values,
        patient_data['pmax_forecast'].values,
        STATE_NAMES,
        filename="hmm_test_patient_v3_simple.png"
    )
    
    # Print sample output
    print("\nSample Output (first 10 rows of test predictions):")
    print(df_test[output_columns].head(10).to_string())
    
    print("\n" + "="*80)
    print("DONE - HMM with Vent Mode Labels Complete")
    print("="*80)


if __name__ == "__main__":
    main()
