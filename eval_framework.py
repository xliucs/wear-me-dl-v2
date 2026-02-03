#!/usr/bin/env python3
"""
Standardized Evaluation Framework for WEAR-ME-DL-V2.
All experiments MUST use this module for consistent train/test splits and metrics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(path='data.csv'):
    """Load and preprocess the WEAR-ME v2 dataset."""
    df = pd.read_csv(path)
    
    # Encode sex
    df['sex_num'] = (df['sex'] == 'Male').astype(int)
    
    # Target
    y = df['True_HOMA_IR'].values
    
    # Remove non-feature columns
    drop_cols = ['Participant_id', 'sex', 'True_HOMA_IR']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    
    # Handle NaN targets
    mask = ~np.isnan(y)
    X = X[mask].reset_index(drop=True)
    y = y[mask]
    
    return X, y, feature_cols

# Feature groups
DEMOGRAPHICS = ['age', 'bmi', 'sex_num']
WEARABLES = [
    'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
    'HRV (mean)', 'HRV (median)', 'HRV (std)',
    'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
    'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
    'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)',
]
BLOOD_BIOMARKERS = ['total cholesterol', 'hdl', 'triglycerides', 'ldl', 'chol/hdl', 'non hdl', 'glucose']

def get_feature_sets(X):
    """Return feature matrices for Model A (all) and Model B (no blood)."""
    all_cols = DEMOGRAPHICS + WEARABLES + BLOOD_BIOMARKERS
    dw_cols = DEMOGRAPHICS + WEARABLES
    
    # Ensure columns exist
    all_cols = [c for c in all_cols if c in X.columns]
    dw_cols = [c for c in dw_cols if c in X.columns]
    
    X_all = X[all_cols].values
    X_dw = X[dw_cols].values
    
    return X_all, X_dw, all_cols, dw_cols

# =============================================================================
# CROSS-VALIDATION
# =============================================================================
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5

def get_cv_splits(y, n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    """
    Get standardized CV splits. Uses stratified K-fold with binned target.
    Returns list of (train_idx, test_idx) tuples.
    ALL experiments must use these exact splits.
    """
    bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(cv.split(np.zeros(len(y)), bins))

def get_oof_predictions(X, y, model_fn, splits=None, scale=True):
    """
    Get out-of-fold predictions using standardized splits.
    
    Args:
        X: feature matrix (np.ndarray)
        y: target array
        model_fn: callable that returns a fitted model, signature: model_fn() -> model with fit/predict
        splits: CV splits (if None, uses get_cv_splits)
        scale: whether to StandardScale features
    
    Returns:
        oof_preds: array of OOF predictions (averaged across repeats)
        fold_scores: list of per-fold R² scores
    """
    if splits is None:
        splits = get_cv_splits(y)
    
    n = len(y)
    oof_sum = np.zeros(n)
    oof_count = np.zeros(n)
    fold_scores = []
    
    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        oof_sum[test_idx] += preds
        oof_count[test_idx] += 1
        fold_scores.append(r2_score(y_test, preds))
    
    # Average OOF predictions across repeats
    oof_preds = oof_sum / np.clip(oof_count, 1, None)
    
    return oof_preds, fold_scores

# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y_true, y_pred):
    """Compute standard metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r, p = pearsonr(y_true, y_pred)
    return {
        'R2': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Pearson_r': r,
        'Pearson_p': p,
    }

def print_metrics(name, y_true, y_pred, fold_scores=None):
    """Print metrics in a standardized format."""
    m = compute_metrics(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  R²:        {m['R2']:.4f}")
    print(f"  Pearson r: {m['Pearson_r']:.4f}")
    print(f"  MAE:       {m['MAE']:.4f}")
    print(f"  RMSE:      {m['RMSE']:.4f}")
    if fold_scores:
        print(f"  Fold R² mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        print(f"  Fold R² range: [{min(fold_scores):.4f}, {max(fold_scores):.4f}]")
    print(f"{'='*60}")
    return m

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def engineer_all_features(X_df, feature_names):
    """
    Engineer features for ALL model (demographics + wearables + blood biomarkers).
    Input X_df should be a DataFrame with original column names.
    """
    X = X_df.copy() if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df, columns=feature_names)
    
    # === Metabolic indices (key for HOMA-IR prediction) ===
    # TyG index: log(triglycerides * glucose / 2) — strongest proxy for IR
    X['tyg'] = np.log(X['triglycerides'].clip(lower=1) * X['glucose'].clip(lower=1) / 2)
    
    # METS-IR: log(2*glucose + triglycerides) * BMI
    X['mets_ir'] = np.log(2 * X['glucose'].clip(lower=1) + X['triglycerides'].clip(lower=1)) * X['bmi']
    
    # Triglyceride/HDL ratio — insulin resistance surrogate
    X['trig_hdl'] = X['triglycerides'] / X['hdl'].clip(lower=1)
    
    # Glucose × BMI interaction
    X['glucose_bmi'] = X['glucose'] * X['bmi']
    
    # Insulin resistance proxy: glucose × BMI × triglycerides / HDL
    X['ir_proxy'] = X['glucose'] * X['bmi'] * X['triglycerides'] / (X['hdl'].clip(lower=1) * 100)
    
    # Visceral adiposity proxy
    X['vat_proxy'] = X['bmi'] * X['triglycerides'] / X['hdl'].clip(lower=1)
    
    # Non-HDL cholesterol ratio
    X['non_hdl_ratio'] = X['non hdl'] / X['hdl'].clip(lower=1)
    
    # === Glucose interactions ===
    X['glucose_sq'] = X['glucose'] ** 2
    X['glucose_age'] = X['glucose'] * X['age']
    X['glucose_hdl'] = X['glucose'] / X['hdl'].clip(lower=1)
    X['glucose_trig'] = X['glucose'] * X['triglycerides']
    X['glucose_log'] = np.log1p(X['glucose'])
    
    # === BMI interactions ===
    X['bmi_sq'] = X['bmi'] ** 2
    X['bmi_cubed'] = X['bmi'] ** 3
    X['bmi_age'] = X['bmi'] * X['age']
    X['bmi_trig'] = X['bmi'] * X['triglycerides']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
    X['age_sq'] = X['age'] ** 2
    
    # === Wearable interactions ===
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'
    slp = 'SLEEP Duration (mean)'
    
    X['bmi_rhr'] = X['bmi'] * X[rhr]
    X['bmi_hrv_inv'] = X['bmi'] / X[hrv].clip(lower=1)
    X['bmi_stp_inv'] = X['bmi'] / X[stp].clip(lower=1) * 1000
    X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)
    X['cardio_fitness'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
    X['met_load'] = X['bmi'] * X[rhr] / X[stp].clip(lower=1) * 1000
    X['sed_risk'] = X['bmi'] ** 2 * X[rhr] / (X[stp].clip(lower=1) * X[hrv].clip(lower=1))
    
    # === Wearable distribution features ===
    for pfx, m, md, s in [
        ('rhr', 'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)'),
        ('hrv', 'HRV (mean)', 'HRV (median)', 'HRV (std)'),
        ('stp', 'STEPS (mean)', 'STEPS (median)', 'STEPS (std)'),
        ('slp', 'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)'),
        ('azm', 'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)'),
    ]:
        X[f'{pfx}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    
    # === Cross-modal: wearables × blood ===
    X['glucose_rhr'] = X['glucose'] * X[rhr]
    X['glucose_hrv_inv'] = X['glucose'] / X[hrv].clip(lower=1)
    X['tyg_bmi_rhr'] = X['tyg'] * X['bmi'] * X[rhr] / 1000
    
    return X.fillna(0)


def engineer_dw_features(X_df, feature_names):
    """
    Engineer features for DW model (demographics + wearables only).
    """
    X = X_df.copy() if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df, columns=feature_names)
    
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'
    slp = 'SLEEP Duration (mean)'
    azm = 'AZM Weekly (mean)'
    rhr_md = 'Resting Heart Rate (median)'
    hrv_md = 'HRV (median)'
    stp_md = 'STEPS (median)'
    slp_md = 'SLEEP Duration (median)'
    azm_md = 'AZM Weekly (median)'
    rhr_s = 'Resting Heart Rate (std)'
    hrv_s = 'HRV (std)'
    stp_s = 'STEPS (std)'
    slp_s = 'SLEEP Duration (std)'
    azm_s = 'AZM Weekly (std)'
    
    # === Distribution features ===
    for pfx, m, md, s in [('rhr', rhr, rhr_md, rhr_s), ('hrv', hrv, hrv_md, hrv_s),
                           ('stp', stp, stp_md, stp_s), ('slp', slp, slp_md, slp_s),
                           ('azm', azm, azm_md, azm_s)]:
        X[f'{pfx}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    
    # === Polynomial features ===
    for col, nm in [(X['bmi'], 'bmi'), (X['age'], 'age'), (X[rhr], 'rhr'), (X[hrv], 'hrv'), (X[stp], 'stp')]:
        X[f'{nm}_sq'] = col ** 2
        X[f'{nm}_log'] = np.log1p(col.clip(lower=0))
        X[f'{nm}_inv'] = 1 / (col.clip(lower=0.01))
    
    # === BMI interactions ===
    X['bmi_rhr'] = X['bmi'] * X[rhr]
    X['bmi_sq_rhr'] = X['bmi'] ** 2 * X[rhr]
    X['bmi_hrv'] = X['bmi'] * X[hrv]
    X['bmi_hrv_inv'] = X['bmi'] / X[hrv].clip(lower=1)
    X['bmi_stp'] = X['bmi'] * X[stp]
    X['bmi_stp_inv'] = X['bmi'] / X[stp].clip(lower=1) * 1000
    X['bmi_slp'] = X['bmi'] * X[slp]
    X['bmi_azm'] = X['bmi'] * X[azm]
    X['bmi_age'] = X['bmi'] * X['age']
    X['bmi_sq_age'] = X['bmi'] ** 2 * X['age']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
    X['bmi_rhr_hrv'] = X['bmi'] * X[rhr] / X[hrv].clip(lower=1)
    X['bmi_rhr_stp'] = X['bmi'] * X[rhr] / X[stp].clip(lower=1) * 1000
    X['bmi_cubed'] = X['bmi'] ** 3
    
    # === Age interactions ===
    X['age_rhr'] = X['age'] * X[rhr]
    X['age_hrv_inv'] = X['age'] / X[hrv].clip(lower=1)
    X['age_stp'] = X['age'] * X[stp]
    X['age_slp'] = X['age'] * X[slp]
    X['age_sex'] = X['age'] * X['sex_num']
    X['age_bmi_sex'] = X['age'] * X['bmi'] * X['sex_num']
    X['age_sq'] = X['age'] ** 2
    
    # === Wearable cross features ===
    X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)
    X['stp_hrv'] = X[stp] * X[hrv]
    X['stp_rhr'] = X[stp] / X[rhr].clip(lower=1)
    X['azm_stp'] = X[azm] / X[stp].clip(lower=1)
    X['slp_hrv'] = X[slp] * X[hrv]
    X['slp_rhr'] = X[slp] / X[rhr].clip(lower=1)
    
    # === Composite health indices ===
    X['cardio_fitness'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
    X['cardio_log'] = np.log1p(X['cardio_fitness'].clip(lower=0))
    X['met_load'] = X['bmi'] * X[rhr] / X[stp].clip(lower=1) * 1000
    X['met_load_log'] = np.log1p(X['met_load'].clip(lower=0))
    X['recovery'] = X[hrv] / X[rhr].clip(lower=1) * X[slp]
    X['activity_bmi'] = (X[stp] + X[azm]) / X['bmi']
    X['sed_risk'] = X['bmi'] ** 2 * X[rhr] / (X[stp].clip(lower=1) * X[hrv].clip(lower=1))
    X['sed_risk_log'] = np.log1p(X['sed_risk'].clip(lower=0))
    X['auto_health'] = X[hrv] / X[rhr].clip(lower=1)
    X['hr_reserve'] = (220 - X['age'] - X[rhr]) / X['bmi']
    X['fitness_age'] = X['age'] * X[rhr] / X[hrv].clip(lower=1)
    X['bmi_fitness'] = X['bmi'] * X[rhr] / (X[hrv].clip(lower=1) * X[stp].clip(lower=1)) * 10000
    
    # === Conditional features ===
    X['obese'] = (X['bmi'] >= 30).astype(float)
    X['older'] = (X['age'] >= 50).astype(float)
    X['obese_rhr'] = X['obese'] * X[rhr]
    X['obese_low_hrv'] = X['obese'] * (X[hrv] < X[hrv].median()).astype(float)
    X['older_bmi'] = X['older'] * X['bmi']
    X['older_rhr'] = X['older'] * X[rhr]
    X['rhr_cv_bmi'] = X['rhr_cv'] * X['bmi']
    X['hrv_cv_bmi'] = X['hrv_cv'] * X['bmi']
    X['rhr_cv_age'] = X['rhr_cv'] * X['age']
    
    # === Rank features ===
    for col in ['bmi', 'age', rhr, hrv, stp]:
        X[f'rank_{col[:3]}'] = X[col].rank(pct=True)
    
    return X.fillna(0)


if __name__ == '__main__':
    # Quick test
    X, y, feat_names = load_data()
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")
    
    X_all, X_dw, all_cols, dw_cols = get_feature_sets(X)
    print(f"\nModel A (ALL): {X_all.shape[1]} features")
    print(f"Model B (DW):  {X_dw.shape[1]} features")
    
    splits = get_cv_splits(y)
    print(f"\nCV splits: {len(splits)} total ({N_SPLITS}-fold × {N_REPEATS} repeats)")
    
    # Verify splits are deterministic
    splits2 = get_cv_splits(y)
    assert all(np.array_equal(s1[0], s2[0]) for s1, s2 in zip(splits, splits2)), "Splits not deterministic!"
    print("✓ Splits are deterministic")
