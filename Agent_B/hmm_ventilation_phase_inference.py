#!/usr/bin/env python3
"""
==============================================================================
HMM-based Ventilation Phase Inference & 1-Hour Forecasting Prototype
==============================================================================

This script infers a patient's ventilation "phase" (4-state Hidden Markov Model)
each hour after intubation and forecasts the phase 1 hour ahead.

CLINICAL PHASES (Hidden States):
    S0 = Acute      (Assist/Control–like, higher support/instability)
    S1 = Recovery   (PSV/PAV+–like, improving but supported)
    S2 = Weaning    (pre-SBT/SBT–like, low support + stable)
    S3 = Liberation (extubation/vent-off proxy)

IMPORTANT: These phases are LATENT COURSE STAGES correlated with available
proxies (PEEP, peak pressure, hemodynamics, labs). We do NOT have direct
ventilator mode, FiO2, SpO2, PaO2, pH, vasopressor dose, PS, or SBT results.

Author: Agent B (HMM Specialist)
Date: January 2026
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM

import os

# =============================================================================
# CONFIGURATION - Easy to tweak cutoffs
# =============================================================================

# Get the directory of this script to resolve relative paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# File paths (use parquet for faster loading)
INPUT_FILE = os.path.join(_PROJECT_ROOT, "clinical_data", "data_v1_max_72_h.parquet")
OUTPUT_CSV = os.path.join(_SCRIPT_DIR, "hmm_phase_predictions.csv")

# Phase label names (mapped to HMM component indices after fitting)
PHASE_NAMES = {0: "Acute", 1: "Recovery", 2: "Weaning", 3: "Liberation"}

# Weak label heuristic cutoffs (for HMM initialization)
PEEP_ACUTE_THRESHOLD = 12       # PEEP >= this suggests Acute
PEEP_WEANING_THRESHOLD = 8      # PEEP <= this suggests Weaning-ready
PEAK_ACUTE_THRESHOLD = 30       # Peak pressure >= this suggests Acute
MAP_UNSTABLE_THRESHOLD = 65     # MAP < this suggests hemodynamic instability
# PEAK_WEANING_CUTOFF will be computed as 75th percentile of peak among low-PEEP rows

# Missing data handling
MAX_FORWARD_FILL_HOURS = 4      # Forward-fill up to this many hours within visit

# HMM configuration
N_COMPONENTS = 4
COVARIANCE_TYPE = "full"
N_ITER = 100
RANDOM_STATE = 42

# Sampling for faster iteration (set to 1.0 to use all data)
SAMPLE_RATE = 0.1  # Use 10% of visits for fast prototyping, set to 1.0 for full run

# Transition matrix initialization (encourage forward progression)
# Structure: Acute -> Recovery -> Weaning -> Liberation (with small backward allowed)
INIT_TRANSMAT = np.array([
    # From Acute:      Acute, Recovery, Weaning, Liberation
    [0.85, 0.12, 0.02, 0.01],
    # From Recovery:   backward to Acute allowed for deterioration
    [0.08, 0.82, 0.08, 0.02],
    # From Weaning:    backward to Recovery allowed
    [0.02, 0.08, 0.82, 0.08],
    # From Liberation: mostly stays, tiny backward to Weaning
    [0.01, 0.02, 0.07, 0.90]
])

# Features to use
BASE_FEATURES = [
    'peep_mean', 'peak_mean', 'map_mean', 'sbp_mean', 'dbp_mean', 'temp_mean',
    'wbc_mean', 'crp_mean', 'creatinine_mean', 'glucose_mean',
    'sodium_mean', 'potassium_mean', 'chloride_mean',
    'hemoglobin_mean', 'platelets_mean'
]

# Features for trend calculation
TREND_FEATURES = ['peep_mean', 'peak_mean', 'map_mean']

# Features for missingness indicators
MISSINGNESS_FEATURES = ['peep_mean', 'peak_mean']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def detect_time_format(measure_time: pd.Series) -> str:
    """
    Detect whether measure_time is:
    - epoch_seconds: Unix timestamp in seconds (~10 digits, > 1e9)
    - epoch_ms: Unix timestamp in milliseconds (~13 digits, > 1e12)
    - hour_index: Small integers representing hours since some reference
    
    Returns: 'epoch_seconds', 'epoch_ms', or 'hour_index'
    """
    sample_vals = measure_time.dropna().head(100)
    if len(sample_vals) == 0:
        return 'hour_index'
    
    median_val = sample_vals.median()
    
    if median_val > 1e12:
        return 'epoch_ms'
    elif median_val > 1e9:
        return 'epoch_seconds'
    else:
        return 'hour_index'


def convert_to_datetime(measure_time: pd.Series, time_format: str) -> pd.Series:
    """Convert measure_time to pandas datetime based on detected format."""
    if time_format == 'epoch_seconds':
        return pd.to_datetime(measure_time, unit='s', errors='coerce')
    elif time_format == 'epoch_ms':
        return pd.to_datetime(measure_time, unit='ms', errors='coerce')
    else:
        # hour_index - return as-is (will use as numeric)
        return measure_time


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Load data and prepare with hourly indexing per visit.
    
    For each visit_occurrence_id:
    - Sort by measure_time
    - Detect time format and convert appropriately
    - Create hourly index t_hr (hours since first measurement)
    - Resample to 1-hour bins if needed
    
    Returns: DataFrame with t_hr column added
    """
    print(f"Loading data from {filepath}...")
    
    # Load data - handle both CSV and parquet
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    # Handle multi-index (visit_occurrence_id, measure_time) from parquet
    if isinstance(df.index, pd.MultiIndex):
        print("  Detected multi-index, resetting...")
        df = df.reset_index()
    
    # Ensure required columns exist
    if 'visit_occurrence_id' not in df.columns:
        raise ValueError("Data must have 'visit_occurrence_id' column")
    if 'measure_time' not in df.columns:
        raise ValueError("Data must have 'measure_time' column")
    
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Unique visits: {df['visit_occurrence_id'].nunique():,}")
    
    # Sample visits if SAMPLE_RATE < 1.0 for faster iteration
    if SAMPLE_RATE < 1.0:
        all_visits = df['visit_occurrence_id'].unique()
        np.random.seed(RANDOM_STATE)
        n_sample = max(1, int(len(all_visits) * SAMPLE_RATE))
        sampled_visits = np.random.choice(all_visits, size=n_sample, replace=False)
        df = df[df['visit_occurrence_id'].isin(sampled_visits)]
        print(f"  Sampled {n_sample:,} visits ({SAMPLE_RATE*100:.0f}%): {len(df):,} rows")
    
    # Detect time format
    time_format = detect_time_format(df['measure_time'])
    print(f"  Detected time format: {time_format}")
    
    # Vectorized time conversion
    df = df.sort_values(['visit_occurrence_id', 'measure_time']).copy()
    
    if time_format == 'hour_index':
        # Already hour index - compute t_hr relative to first measurement per visit
        df['t_hr'] = df.groupby('visit_occurrence_id')['measure_time'].transform(
            lambda x: x - x.min()
        ).astype(float)
    else:
        # Convert to datetime
        unit = 's' if time_format == 'epoch_seconds' else 'ms'
        df['datetime'] = pd.to_datetime(df['measure_time'], unit=unit, errors='coerce')
        df['t_hr'] = df.groupby('visit_occurrence_id')['datetime'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / 3600.0
        )
    
    # Round to hourly bins
    df['t_hr_bin'] = df['t_hr'].round().astype(int)
    
    # Resample to hourly using groupby aggregation (vectorized, much faster)
    print("  Resampling to hourly bins...")
    
    # Identify column types
    id_cols = ['visit_occurrence_id', 'person_id']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in numeric_cols if c not in ['t_hr_bin', 'measure_time', 't_hr', 'visit_occurrence_id', 'person_id']]
    
    # Group by visit and hour, aggregate
    agg_dict = {col: 'median' for col in agg_cols}
    agg_dict['person_id'] = 'first'
    
    df_hourly = df.groupby(['visit_occurrence_id', 't_hr_bin']).agg(agg_dict).reset_index()
    df_hourly['t_hr'] = df_hourly['t_hr_bin'].astype(float)
    
    print(f"  After hourly resampling: {len(df_hourly):,} rows")
    
    return df_hourly


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Engineer features for HMM:
    - Base features (vitals, pressures, labs)
    - Trend features: 1-hour delta and 4-hour rolling slope
    - Missingness indicators
    
    Returns: (DataFrame with features, list of feature column names)
    """
    print("Engineering features...")
    
    df = df.copy()
    feature_cols = []
    
    # 1. Base features - check which exist in data
    available_base = [f for f in BASE_FEATURES if f in df.columns]
    feature_cols.extend(available_base)
    print(f"  Base features available: {len(available_base)}/{len(BASE_FEATURES)}")
    
    # 2. Trend features per visit
    for feat in TREND_FEATURES:
        if feat not in df.columns:
            continue
            
        # 1-hour delta
        delta_col = f"{feat}_delta1h"
        df[delta_col] = df.groupby('visit_occurrence_id')[feat].diff(1)
        feature_cols.append(delta_col)
        
        # 4-hour rolling slope (approximated as diff over 4 hours)
        slope_col = f"{feat}_slope4h"
        df[slope_col] = df.groupby('visit_occurrence_id')[feat].transform(
            lambda x: x.diff(4) / 4  # Simple slope: (x[t] - x[t-4]) / 4
        )
        feature_cols.append(slope_col)
    
    print(f"  Added trend features: {len([f for f in feature_cols if 'delta' in f or 'slope' in f])}")
    
    # 3. Missingness indicators (important for Liberation detection)
    for feat in MISSINGNESS_FEATURES:
        if feat not in df.columns:
            continue
        miss_col = f"{feat}_missing"
        df[miss_col] = df[feat].isna().astype(float)
        feature_cols.append(miss_col)
    
    print(f"  Added missingness indicators: {len([f for f in feature_cols if 'missing' in f])}")
    print(f"  Total features: {len(feature_cols)}")
    
    return df, feature_cols


def create_weak_labels(df: pd.DataFrame, use_raw_thresholds: bool = True) -> pd.Series:
    """
    Create simple heuristic labels to seed HMM initialization.
    
    Labels:
    - 0 (Acute-like): high PEEP OR high peak OR low MAP
    - 1 (Recovery-like): default when not matching other criteria
    - 2 (Weaning-like): low PEEP AND stable MAP AND moderate peak
    - 3 (Liberation-like): missing vent measurements
    
    NOTE: These are rough proxies since we lack direct ventilator mode data.
    The HMM will refine these based on observation patterns.
    
    If use_raw_thresholds=False, uses z-score based thresholds for standardized data.
    """
    print("Creating weak labels for HMM initialization...")
    
    labels = pd.Series(index=df.index, dtype=float)
    labels[:] = 1  # Default: Recovery-like
    
    # Determine if data is standardized (check if mean is close to 0)
    if 'peep_mean' in df.columns:
        data_mean = df['peep_mean'].mean()
        is_standardized = abs(data_mean) < 1.0  # Standardized data has mean ~0
    else:
        is_standardized = False
    
    if is_standardized or not use_raw_thresholds:
        # Use percentile-based thresholds for standardized data
        print("  Using percentile-based thresholds (standardized data)")
        peep_acute_pctl = 70   # Top 30%
        peep_weaning_pctl = 40  # Bottom 40%
        map_unstable_pctl = 30  # Bottom 30%
        peak_acute_pctl = 70
    else:
        # Use raw thresholds
        print("  Using raw value thresholds")
    
    # Get peak cutoff for weaning
    if 'peep_mean' in df.columns and 'peak_mean' in df.columns:
        if is_standardized:
            peep_weaning_thresh = df['peep_mean'].quantile(peep_weaning_pctl / 100)
            low_peep_rows = df['peep_mean'] <= peep_weaning_thresh
        else:
            low_peep_rows = df['peep_mean'] <= PEEP_WEANING_THRESHOLD
        peak_weaning_cutoff = df.loc[low_peep_rows, 'peak_mean'].quantile(0.75)
        if pd.isna(peak_weaning_cutoff):
            peak_weaning_cutoff = df['peak_mean'].quantile(0.5)  # Fallback to median
    else:
        peak_weaning_cutoff = 0  # z-score
    
    print(f"  Peak weaning cutoff: {peak_weaning_cutoff:.2f}")
    
    # Liberation-like: missing vent measurements
    if 'peep_mean_missing' in df.columns:
        lib_mask = df['peep_mean_missing'] > 0.5
        labels[lib_mask] = 3
        print(f"  Liberation-like (missing vent): {lib_mask.sum()} rows")
    elif 'peep_mean' in df.columns:
        lib_mask = df['peep_mean'].isna()
        if 'peak_mean' in df.columns:
            lib_mask = lib_mask | df['peak_mean'].isna()
        labels[lib_mask] = 3
        print(f"  Liberation-like (missing vent): {lib_mask.sum()} rows")
    
    # Acute-like: high PEEP OR high peak OR low MAP
    acute_mask = pd.Series(False, index=df.index)
    if 'peep_mean' in df.columns:
        if is_standardized:
            thresh = df['peep_mean'].quantile(peep_acute_pctl / 100)
            acute_mask = acute_mask | (df['peep_mean'] >= thresh)
        else:
            acute_mask = acute_mask | (df['peep_mean'] >= PEEP_ACUTE_THRESHOLD)
    if 'peak_mean' in df.columns:
        if is_standardized:
            thresh = df['peak_mean'].quantile(peak_acute_pctl / 100)
            acute_mask = acute_mask | (df['peak_mean'] >= thresh)
        else:
            acute_mask = acute_mask | (df['peak_mean'] >= PEAK_ACUTE_THRESHOLD)
    if 'map_mean' in df.columns:
        if is_standardized:
            thresh = df['map_mean'].quantile(map_unstable_pctl / 100)
            acute_mask = acute_mask | (df['map_mean'] <= thresh)
        else:
            acute_mask = acute_mask | (df['map_mean'] < MAP_UNSTABLE_THRESHOLD)
    
    # Don't override Liberation
    acute_mask = acute_mask & (labels != 3)
    labels[acute_mask] = 0
    print(f"  Acute-like (high support/instability): {acute_mask.sum()} rows")
    
    # Weaning-like: low PEEP AND stable MAP AND moderate peak
    weaning_mask = pd.Series(True, index=df.index)
    if 'peep_mean' in df.columns:
        if is_standardized:
            thresh = df['peep_mean'].quantile(peep_weaning_pctl / 100)
            weaning_mask = weaning_mask & (df['peep_mean'] <= thresh)
        else:
            weaning_mask = weaning_mask & (df['peep_mean'] <= PEEP_WEANING_THRESHOLD)
    if 'map_mean' in df.columns:
        if is_standardized:
            thresh = df['map_mean'].quantile((100 - map_unstable_pctl) / 100)
            weaning_mask = weaning_mask & (df['map_mean'] >= thresh)
        else:
            weaning_mask = weaning_mask & (df['map_mean'] >= MAP_UNSTABLE_THRESHOLD)
    if 'peak_mean' in df.columns:
        weaning_mask = weaning_mask & (df['peak_mean'] <= peak_weaning_cutoff)
    
    # Don't override Liberation or Acute
    weaning_mask = weaning_mask & (labels == 1)
    labels[weaning_mask] = 2
    print(f"  Weaning-like (low support + stable): {weaning_mask.sum()} rows")
    
    recovery_count = (labels == 1).sum()
    print(f"  Recovery-like (intermediate): {recovery_count} rows")
    
    return labels


def handle_missing_data(
    df: pd.DataFrame, 
    feature_cols: List[str],
    train_mask: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, StandardScaler, Dict[str, float]]:
    """
    Handle missing data:
    1. Forward-fill within each visit (up to MAX_FORWARD_FILL_HOURS)
    2. Impute remaining with global median (fit on train only)
    3. Standardize features
    
    Returns: (filled_df, fitted_scaler, median_impute_values)
    """
    print("Handling missing data...")
    
    df = df.copy()
    
    # 1. Forward-fill within each visit (limited to MAX_FORWARD_FILL_HOURS)
    for col in feature_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby('visit_occurrence_id')[col].transform(
            lambda x: x.ffill(limit=MAX_FORWARD_FILL_HOURS)
        )
    
    # 2. Compute global medians from training data
    if train_mask is None:
        train_mask = pd.Series(True, index=df.index)
    
    medians = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        medians[col] = df.loc[train_mask, col].median()
        if pd.isna(medians[col]):
            medians[col] = 0.0  # Fallback
    
    # 3. Impute remaining missing values
    for col in feature_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(medians[col])
    
    # Check for any remaining issues
    missing_after = df[feature_cols].isna().sum().sum()
    print(f"  Missing values after imputation: {missing_after}")
    
    # 4. Standardize
    scaler = StandardScaler()
    df_train_features = df.loc[train_mask, feature_cols]
    scaler.fit(df_train_features)
    
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    print(f"  Features standardized using {train_mask.sum()} training rows")
    
    return df, scaler, medians


def init_hmm_from_weak_labels(
    X_train: np.ndarray,
    weak_labels: np.ndarray,
    n_components: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize HMM means and covariances from weak labels.
    
    Falls back to KMeans if weak labels don't cover all components.
    
    Returns: (means_, covars_)
    """
    print("Initializing HMM parameters from weak labels...")
    
    n_features = X_train.shape[1]
    means = np.zeros((n_components, n_features))
    covars = np.zeros((n_components, n_features, n_features))
    
    # Check coverage of each label
    label_counts = pd.Series(weak_labels).value_counts()
    print(f"  Weak label distribution: {dict(label_counts)}")
    
    use_kmeans = False
    for i in range(n_components):
        mask = weak_labels == i
        if mask.sum() < max(10, n_features + 1):  # Need enough samples for covariance
            print(f"  Warning: Label {i} has only {mask.sum()} samples, using KMeans fallback")
            use_kmeans = True
            break
        
        means[i] = X_train[mask].mean(axis=0)
        # Compute covariance with regularization for numerical stability
        cov = np.cov(X_train[mask].T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Ensure positive definiteness
        cov = cov + 0.01 * np.eye(n_features)
        covars[i] = cov
    
    if use_kmeans:
        print("  Falling back to KMeans initialization...")
        kmeans = KMeans(n_clusters=n_components, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X_train)
        
        for i in range(n_components):
            mask = kmeans.labels_ == i
            if mask.sum() < n_features + 1:
                # Not enough samples, use global covariance
                means[i] = X_train[mask].mean(axis=0) if mask.sum() > 0 else X_train.mean(axis=0)
                cov = np.cov(X_train.T)
            else:
                means[i] = X_train[mask].mean(axis=0)
                cov = np.cov(X_train[mask].T)
            
            if cov.ndim == 0:
                cov = np.array([[cov]])
            # Ensure positive definiteness with stronger regularization
            cov = cov + 0.01 * np.eye(n_features)
            covars[i] = cov
    
    # Final check: ensure all covariances are positive definite
    for i in range(n_components):
        try:
            np.linalg.cholesky(covars[i])
        except np.linalg.LinAlgError:
            print(f"  Warning: Covariance {i} not positive definite, adding regularization")
            covars[i] = covars[i] + 0.1 * np.eye(n_features)
    
    return means, covars


def fit_hmm(
    X_sequences: List[np.ndarray],
    lengths: List[int],
    means_init: Optional[np.ndarray] = None,
    covars_init: Optional[np.ndarray] = None,
    fix_transmat: bool = False
) -> GaussianHMM:
    """
    Fit Gaussian HMM on concatenated sequences.
    
    Args:
        X_sequences: List of feature arrays, one per visit
        lengths: Length of each sequence
        means_init: Initial means (from weak labels or KMeans)
        covars_init: Initial covariances
        fix_transmat: If True, don't update transition matrix during training
    
    Returns: Fitted GaussianHMM model
    """
    print("Fitting HMM...")
    
    # Concatenate all sequences
    X_concat = np.vstack(X_sequences)
    print(f"  Total training samples: {len(X_concat)}")
    print(f"  Number of sequences: {len(lengths)}")
    
    # Initialize model
    model = GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type=COVARIANCE_TYPE,
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    # Set initial parameters
    model.startprob_ = np.array([0.7, 0.2, 0.08, 0.02])  # Most start in Acute
    model.transmat_ = INIT_TRANSMAT.copy()
    
    if means_init is not None:
        model.means_ = means_init
    if covars_init is not None:
        model.covars_ = covars_init
    
    # Control what parameters to update
    # 's' = startprob, 't' = transmat, 'm' = means, 'c' = covariances
    if fix_transmat:
        model.params = 'mc'  # Only update means and covariances
        model.init_params = ''  # Don't reinitialize anything
        print("  Transition matrix: FIXED (not updated during training)")
    else:
        model.params = 'stmc'
        model.init_params = ''  # We set initial params manually
        print("  Transition matrix: Will be updated during training")
    
    # Fit
    model.fit(X_concat, lengths)
    
    print(f"  Training converged: {model.monitor_.converged}")
    print(f"  Final log-likelihood: {model.score(X_concat, lengths):.2f}")
    
    # Print learned transition matrix
    print("\n  Learned transition matrix:")
    print("  " + " ".join([f"{PHASE_NAMES[i]:>10}" for i in range(N_COMPONENTS)]))
    for i in range(N_COMPONENTS):
        row = " ".join([f"{model.transmat_[i,j]:>10.3f}" for j in range(N_COMPONENTS)])
        print(f"  {PHASE_NAMES[i]:<10} {row}")
    
    return model


def infer_and_forecast(
    model: GaussianHMM,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    For each visit:
    - Compute filtered posterior P(S_t | x_0:t)
    - Compute 1-hour-ahead forecast: P(S_{t+1} | x_0:t)
    
    NOTE: Using model.predict_proba gives P(S_t | x_0:T) (smoothed), not filtered.
    For true filtered inference, we implement a forward pass.
    
    Returns: DataFrame with predictions
    """
    print("\nRunning inference and forecasting...")
    
    results = []
    
    for visit_id, visit_df in df.groupby('visit_occurrence_id'):
        visit_df = visit_df.sort_values('t_hr').reset_index(drop=True)
        X = visit_df[feature_cols].values
        
        if len(X) == 0:
            continue
        
        # Get transition matrix and emission parameters
        transmat = model.transmat_
        
        # Forward algorithm for filtered posteriors
        # alpha[t] = P(S_t, x_0:t) (unnormalized)
        n_samples = len(X)
        n_components = model.n_components
        
        # Compute log emission probabilities
        # log P(x_t | S_t = k) for all t, k
        log_emission = np.zeros((n_samples, n_components))
        for k in range(n_components):
            log_emission[:, k] = model._compute_log_likelihood(X)[:, k] if hasattr(model, '_compute_log_likelihood') else 0
        
        # Use score_samples for log-likelihoods per state
        try:
            # This gives log P(x_t | S_t) for each state
            _, log_gamma = model.score_samples(X)
            # log_gamma is log P(S_t | X) - we need to convert to filtered
            # For simplicity, we use the forward pass approximation
            
            # Simple forward pass
            log_alpha = np.zeros((n_samples, n_components))
            
            # Initialize: alpha_0 = pi * P(x_0 | S_0)
            log_startprob = np.log(model.startprob_ + 1e-10)
            log_transmat = np.log(transmat + 1e-10)
            
            # Get emission log-probs
            from scipy.stats import multivariate_normal
            log_emit = np.zeros((n_samples, n_components))
            for k in range(n_components):
                try:
                    mvn = multivariate_normal(
                        mean=model.means_[k],
                        cov=model.covars_[k],
                        allow_singular=True
                    )
                    log_emit[:, k] = mvn.logpdf(X)
                except:
                    log_emit[:, k] = -100  # Very low probability
            
            # Forward pass
            log_alpha[0] = log_startprob + log_emit[0]
            
            for t in range(1, n_samples):
                for j in range(n_components):
                    log_alpha[t, j] = log_emit[t, j] + np.logaddexp.reduce(
                        log_alpha[t-1] + log_transmat[:, j]
                    )
            
            # Normalize to get filtered probabilities
            log_normalizer = np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
            filtered_probs = np.exp(log_alpha - log_normalizer)
            
        except Exception as e:
            print(f"  Warning: Forward pass failed for visit {visit_id}, using predict_proba: {e}")
            filtered_probs = model.predict_proba(X)
        
        # Inferred phase (argmax of filtered)
        phase_hat = np.argmax(filtered_probs, axis=1)
        
        # 1-hour ahead forecast: P(S_{t+1} | x_0:t) = sum_k P(S_{t+1} | S_t=k) * P(S_t=k | x_0:t)
        forecast_probs = filtered_probs @ transmat
        phase_forecast = np.argmax(forecast_probs, axis=1)
        
        # Build results for this visit
        for i, row in visit_df.iterrows():
            result = {
                'visit_occurrence_id': visit_id,
                't_hr': row['t_hr'],
                'phase_hat': phase_hat[visit_df.index.get_loc(i)] if i in visit_df.index else phase_hat[0],
                'phase_name': PHASE_NAMES[phase_hat[visit_df.index.get_loc(i)] if i in visit_df.index else phase_hat[0]],
            }
            
            idx = list(visit_df.index).index(i) if i in visit_df.index else 0
            
            # Current phase probabilities
            for k in range(n_components):
                result[f'prob_{PHASE_NAMES[k]}'] = filtered_probs[idx, k]
            
            # Forecast
            result['phase_hat_tplus1'] = phase_forecast[idx]
            result['phase_name_tplus1'] = PHASE_NAMES[phase_forecast[idx]]
            
            # Forecast probabilities
            for k in range(n_components):
                result[f'prob_{PHASE_NAMES[k]}_tplus1'] = forecast_probs[idx, k]
            
            results.append(result)
    
    results_df = pd.DataFrame(results)
    print(f"  Generated predictions for {results_df['visit_occurrence_id'].nunique()} visits")
    print(f"  Total prediction rows: {len(results_df)}")
    
    return results_df


def plot_sanity_checks(
    df_original: pd.DataFrame,
    predictions_df: pd.DataFrame,
    n_sample_visits: int = 3
) -> None:
    """
    Generate sanity check plots:
    (a) Average PEEP and peak pressure by inferred phase
    (b) Sample visit timelines showing phase progression
    """
    print("\nGenerating sanity check plots...")
    
    # Merge predictions with original data
    df_plot = df_original.merge(
        predictions_df[['visit_occurrence_id', 't_hr', 'phase_hat', 'phase_name']],
        on=['visit_occurrence_id', 't_hr'],
        how='inner'
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot (a): Average vitals by phase
    ax1 = axes[0, 0]
    if 'peep_mean' in df_plot.columns:
        phase_peep = df_plot.groupby('phase_name')['peep_mean'].mean()
        phase_peep = phase_peep.reindex(['Acute', 'Recovery', 'Weaning', 'Liberation'])
        bars1 = ax1.bar(phase_peep.index, phase_peep.values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        ax1.set_ylabel('Mean PEEP (cmH2O)')
        ax1.set_title('Average PEEP by Inferred Phase')
        ax1.set_ylim(bottom=0)
    
    ax2 = axes[0, 1]
    if 'peak_mean' in df_plot.columns:
        phase_peak = df_plot.groupby('phase_name')['peak_mean'].mean()
        phase_peak = phase_peak.reindex(['Acute', 'Recovery', 'Weaning', 'Liberation'])
        bars2 = ax2.bar(phase_peak.index, phase_peak.values, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
        ax2.set_ylabel('Mean Peak Pressure (cmH2O)')
        ax2.set_title('Average Peak Pressure by Inferred Phase')
        ax2.set_ylim(bottom=0)
    
    # Plot (b): Sample visit timelines
    # Select visits with sufficient length and phase variation
    visit_lengths = df_plot.groupby('visit_occurrence_id').size()
    good_visits = visit_lengths[visit_lengths >= 6].index.tolist()
    
    if len(good_visits) > 0:
        sample_visits = np.random.choice(
            good_visits, 
            size=min(n_sample_visits, len(good_visits)), 
            replace=False
        )
        
        colors = {'Acute': '#d62728', 'Recovery': '#ff7f0e', 'Weaning': '#2ca02c', 'Liberation': '#1f77b4'}
        
        for idx, visit_id in enumerate(sample_visits[:2]):  # Plot 2 sample visits
            ax = axes[1, idx]
            visit_data = df_plot[df_plot['visit_occurrence_id'] == visit_id].sort_values('t_hr')
            
            # Plot phase timeline
            for phase in colors:
                phase_data = visit_data[visit_data['phase_name'] == phase]
                ax.scatter(phase_data['t_hr'], [phase] * len(phase_data), 
                          c=colors[phase], s=100, label=phase if idx == 0 else None)
            
            ax.set_xlabel('Hours since intubation')
            ax.set_ylabel('Inferred Phase')
            ax.set_title(f'Visit {visit_id} Phase Timeline')
            ax.set_xlim(left=-0.5)
        
        # Add legend to first timeline
        axes[1, 0].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('hmm_sanity_checks.png', dpi=150, bbox_inches='tight')
    print("  Saved: hmm_sanity_checks.png")
    plt.show()
    
    # Additional plot: Phase distribution
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    phase_counts = predictions_df['phase_name'].value_counts()
    phase_counts = phase_counts.reindex(['Acute', 'Recovery', 'Weaning', 'Liberation'])
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    ax3.bar(phase_counts.index, phase_counts.values, color=colors)
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution of Inferred Phases')
    
    # Add percentages
    total = phase_counts.sum()
    for i, (phase, count) in enumerate(phase_counts.items()):
        ax3.annotate(f'{count/total*100:.1f}%', 
                    xy=(i, count), 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('hmm_phase_distribution.png', dpi=150, bbox_inches='tight')
    print("  Saved: hmm_phase_distribution.png")
    plt.show()


def main():
    """
    Main pipeline:
    1. Load and prepare data
    2. Feature engineering
    3. Handle missing data
    4. Create weak labels for initialization
    5. Fit HMM
    6. Infer phases and forecast
    7. Save outputs and plot
    """
    print("=" * 70)
    print("HMM Ventilation Phase Inference Pipeline")
    print("=" * 70)
    
    # 1. Load and prepare
    df = load_and_prepare(INPUT_FILE)
    
    # 2. Feature engineering
    df, feature_cols = make_features(df)
    
    # Filter to only use features that exist
    feature_cols = [f for f in feature_cols if f in df.columns]
    print(f"\nUsing {len(feature_cols)} features: {feature_cols[:5]}... (truncated)")
    
    # 3. Train/test split by visit (80/20)
    visits = df['visit_occurrence_id'].unique()
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(visits)
    split_idx = int(0.8 * len(visits))
    train_visits = set(visits[:split_idx])
    test_visits = set(visits[split_idx:])
    
    train_mask = df['visit_occurrence_id'].isin(train_visits)
    print(f"\nTrain visits: {len(train_visits)}, Test visits: {len(test_visits)}")
    
    # 4. Handle missing data
    df, scaler, medians = handle_missing_data(df, feature_cols, train_mask)
    
    # 5. Create weak labels on training data
    weak_labels = create_weak_labels(df[train_mask])
    
    # 6. Prepare sequences for HMM
    train_sequences = []
    train_lengths = []
    train_weak_labels = []
    
    for visit_id in train_visits:
        visit_df = df[df['visit_occurrence_id'] == visit_id].sort_values('t_hr')
        if len(visit_df) < 2:  # Skip very short visits
            continue
        X = visit_df[feature_cols].values
        train_sequences.append(X)
        train_lengths.append(len(X))
        # Get weak labels for this visit
        visit_weak = weak_labels.loc[visit_df.index].values
        train_weak_labels.extend(visit_weak)
    
    print(f"\nPrepared {len(train_sequences)} training sequences")
    
    # 7. Initialize HMM from weak labels
    X_train_concat = np.vstack(train_sequences)
    train_weak_labels = np.array(train_weak_labels)
    means_init, covars_init = init_hmm_from_weak_labels(
        X_train_concat, train_weak_labels, N_COMPONENTS
    )
    
    # 8. Fit HMM
    model = fit_hmm(
        train_sequences, 
        train_lengths,
        means_init=means_init,
        covars_init=covars_init,
        fix_transmat=False  # Allow transition matrix to be learned
    )
    
    # 9. Inference on all data
    predictions_df = infer_and_forecast(model, df, feature_cols)
    
    # 10. Save predictions
    predictions_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to: {OUTPUT_CSV}")
    
    # 11. Sanity checks
    plot_sanity_checks(df, predictions_df)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total visits processed: {predictions_df['visit_occurrence_id'].nunique()}")
    print(f"Total hourly predictions: {len(predictions_df)}")
    print("\nPhase distribution:")
    print(predictions_df['phase_name'].value_counts())
    
    # Forecast accuracy on test set (compare t+1 forecast vs actual t+1 phase)
    test_preds = predictions_df[predictions_df['visit_occurrence_id'].isin(test_visits)]
    if len(test_preds) > 0:
        # For each visit, compare forecast at t with actual at t+1
        forecast_matches = []
        for visit_id in test_visits:
            visit_preds = test_preds[test_preds['visit_occurrence_id'] == visit_id].sort_values('t_hr')
            if len(visit_preds) < 2:
                continue
            for i in range(len(visit_preds) - 1):
                forecast = visit_preds.iloc[i]['phase_hat_tplus1']
                actual_next = visit_preds.iloc[i + 1]['phase_hat']
                forecast_matches.append(forecast == actual_next)
        
        if forecast_matches:
            accuracy = np.mean(forecast_matches)
            print(f"\n1-hour forecast accuracy (test set): {accuracy:.1%}")
    
    print("\nPipeline complete!")
    return model, predictions_df, df


if __name__ == "__main__":
    model, predictions, df_processed = main()
