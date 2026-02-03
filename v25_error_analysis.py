#!/usr/bin/env python3
"""
V25: Deep Error Analysis — WHERE and WHY are we failing?

Before trying more models, understand:
1. Which samples have largest errors? What do they have in common?
2. Is the error systematic (bias) or random (irreducible noise)?
3. Are there subgroups where we're great vs terrible?
4. What does the residual distribution look like?
5. Feature importance vs residual correlation — what signal are we missing?
6. Upper bound estimation via leave-one-feature-out
"""
import numpy as np, pandas as pd, time, warnings, sys, json
from collections import defaultdict
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V25: DEEP ERROR ANALYSIS")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
y_log = np.log1p(y)

# V7 features
def eng_v7(X_df, cols):
    X = X_df[cols].copy()
    g=X['glucose'].clip(lower=1); t=X['triglycerides'].clip(lower=1)
    h=X['hdl'].clip(lower=1); b=X['bmi']; l=X['ldl']
    tc=X['total cholesterol']; nh=X['non hdl']; ch=X['chol/hdl']; a=X['age']
    rhr=X['Resting Heart Rate (mean)']; hrv=X['HRV (mean)'].clip(lower=1)
    stp=X['STEPS (mean)'].clip(lower=1); slp=X['SLEEP Duration (mean)']
    for nm,v in [('tyg',np.log(t*g/2)),('tyg_bmi',np.log(t*g/2)*b),
        ('mets_ir',np.log(2*g+t)*b/np.log(h)),('trig_hdl',t/h),('trig_hdl_log',np.log1p(t/h)),
        ('vat_proxy',b*t/h),('ir_proxy',g*b*t/(h*100)),('glucose_bmi',g*b),
        ('glucose_sq',g**2),('glucose_log',np.log(g)),('glucose_hdl',g/h),
        ('glucose_trig',g*t/1000),('glucose_non_hdl',g*nh/100),('glucose_chol_hdl',g*ch),
        ('bmi_sq',b**2),('bmi_cubed',b**3),('bmi_age',b*a),('age_sq',a**2),
        ('bmi_rhr',b*rhr),('bmi_hrv_inv',b/hrv),('bmi_stp_inv',b/stp*1000),
        ('rhr_hrv',rhr/hrv),('cardio_fitness',hrv*stp/rhr),
        ('met_load',b*rhr/stp*1000),('sed_risk',b**2*rhr/(stp*hrv)),
        ('glucose_rhr',g*rhr),('glucose_hrv_inv',g/hrv),('tyg_bmi_rhr',np.log(t*g/2)*b*rhr/1000),
        ('log_homa_proxy',np.log(g)+np.log(b)+np.log(t)-np.log(h)),
        ('ir_proxy_log',np.log1p(g*b*t/(h*100))),('ir_proxy_rhr',g*b*t/(h*100)*rhr/100),
        ('non_hdl_ratio',nh/h),('chol_glucose',tc*g/1000),('ldl_glucose',l*g/1000),
        ('bmi_trig',b*t/1000),('bmi_sex',b*X['sex_num']),('glucose_age',g*a/100),
        ('rhr_skew',(rhr-X['Resting Heart Rate (median)'])/X['Resting Heart Rate (std)'].clip(lower=0.01)),
        ('hrv_skew',(hrv-X['HRV (median)'])/X['HRV (std)'].clip(lower=0.01)),
        ('stp_skew',(stp-X['STEPS (median)'])/X['STEPS (std)'].clip(lower=0.01)),
        ('slp_skew',(slp-X['SLEEP Duration (median)'])/X['SLEEP Duration (std)'].clip(lower=0.01)),
        ('azm_skew',(X['AZM Weekly (mean)']-X['AZM Weekly (median)'])/X['AZM Weekly (std)'].clip(lower=0.01)),
        ('rhr_cv',X['Resting Heart Rate (std)'].clip(lower=0.01)/rhr),
        ('hrv_cv',X['HRV (std)'].clip(lower=0.01)/hrv),
        ('stp_cv',X['STEPS (std)'].clip(lower=0.01)/stp),
        ('slp_cv',X['SLEEP Duration (std)'].clip(lower=0.01)/slp),
        ('azm_cv',X['AZM Weekly (std)'].clip(lower=0.01)/X['AZM Weekly (mean)'].clip(lower=0.01)),
    ]:
        X[nm] = v
    return X.fillna(0)

X_v7_df = eng_v7(X_df, all_cols)
X_v7 = X_v7_df.values
feat_names = list(X_v7_df.columns)
print(f"Features: {len(feat_names)}")

# Get OOF predictions from best model (XGB Optuna wsqrt)
oof_sum = np.zeros(n)
oof_count = np.zeros(n)
for tr, te in splits:
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(X_v7[tr], y_log[tr], sample_weight=w_sqrt[tr])
    oof_sum[te] += m.predict(X_v7[te])
    oof_count[te] += 1

oof_log = oof_sum / np.clip(oof_count, 1, None)
oof = np.expm1(oof_log)
residuals = y - oof
abs_errors = np.abs(residuals)
rel_errors = abs_errors / np.clip(y, 0.1, None)

print(f"\nOverall: R² = {r2_score(y, oof):.4f}")
print(f"MAE = {mean_absolute_error(y, oof):.3f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, oof)):.3f}")

# ============================================================
# 1. ERROR BY HOMA RANGE
# ============================================================
print("\n" + "="*60)
print("  1. ERROR BY HOMA-IR RANGE")
print("="*60)

ranges = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 15), (15, 100)]
for lo, hi in ranges:
    mask = (y >= lo) & (y < hi)
    cnt = mask.sum()
    if cnt == 0: continue
    mae = mean_absolute_error(y[mask], oof[mask])
    bias = np.mean(residuals[mask])
    mrel = np.mean(rel_errors[mask])
    r2_sub = r2_score(y[mask], oof[mask]) if cnt > 1 and np.std(y[mask]) > 0 else float('nan')
    print(f"  [{lo:2d}-{hi:2d}): n={cnt:4d}, MAE={mae:.3f}, bias={bias:+.3f}, "
          f"rel_err={mrel:.1%}, R²={r2_sub:.3f}")
sys.stdout.flush()

# ============================================================
# 2. ERROR BY FEATURE SUBGROUPS
# ============================================================
print("\n" + "="*60)
print("  2. ERROR BY FEATURE SUBGROUPS")
print("="*60)

glucose = X_df['glucose'].values
bmi = X_df['bmi'].values
trig = X_df['triglycerides'].values
hdl = X_df['hdl'].values
age = X_df['age'].values
sex = X_df['sex_num'].values
rhr = X_df['Resting Heart Rate (mean)'].values
steps = X_df['STEPS (mean)'].values

for fname, vals, cuts in [
    ('glucose', glucose, [70, 90, 100, 110, 126, 200]),
    ('bmi', bmi, [18.5, 25, 30, 35, 40, 60]),
    ('triglycerides', trig, [50, 100, 150, 200, 300, 500]),
    ('hdl', hdl, [30, 40, 50, 60, 80, 120]),
    ('age', age, [20, 30, 40, 50, 60, 70, 90]),
    ('sex', sex, [0, 1, 2]),
    ('rhr', rhr, [40, 55, 65, 75, 85, 100]),
    ('steps', steps, [0, 3000, 6000, 9000, 12000, 30000]),
]:
    print(f"\n  --- {fname} ---")
    for i in range(len(cuts)-1):
        lo, hi = cuts[i], cuts[i+1]
        mask = (vals >= lo) & (vals < hi)
        cnt = mask.sum()
        if cnt < 10: continue
        mae = mean_absolute_error(y[mask], oof[mask])
        bias = np.mean(residuals[mask])
        r2_sub = r2_score(y[mask], oof[mask]) if cnt > 1 and np.std(y[mask]) > 0 else float('nan')
        print(f"    [{lo:6.0f}-{hi:6.0f}): n={cnt:4d}, MAE={mae:.3f}, bias={bias:+.3f}, R²={r2_sub:.3f}")
sys.stdout.flush()

# ============================================================
# 3. RESIDUAL ANALYSIS
# ============================================================
print("\n" + "="*60)
print("  3. RESIDUAL STATISTICS")
print("="*60)

print(f"  Mean residual: {np.mean(residuals):.4f}")
print(f"  Std residual: {np.std(residuals):.4f}")
print(f"  Skewness: {pd.Series(residuals).skew():.4f}")
print(f"  Kurtosis: {pd.Series(residuals).kurtosis():.4f}")
print(f"  Min/Max residual: {residuals.min():.3f} / {residuals.max():.3f}")

# Percentiles
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  P{pct:02d}: {np.percentile(residuals, pct):.3f}")

# ============================================================
# 4. RESIDUAL vs FEATURE CORRELATION
# ============================================================
print("\n" + "="*60)
print("  4. RESIDUAL CORRELATION WITH FEATURES")
print("="*60)

correlations = []
for i, fname in enumerate(feat_names):
    r = np.corrcoef(X_v7[:, i], residuals)[0, 1]
    correlations.append((fname, r, abs(r)))
correlations.sort(key=lambda x: -x[2])
print("  Top 20 features correlated with residuals:")
for fname, r, ar in correlations[:20]:
    print(f"    {fname:30s}: r={r:+.4f}")

# Also check raw features
print("\n  Raw feature correlations with residuals:")
for col in all_cols:
    r = np.corrcoef(X_df[col].values, residuals)[0, 1]
    print(f"    {col:30s}: r={r:+.4f}")
sys.stdout.flush()

# ============================================================
# 5. WORST PREDICTIONS — what makes them special?
# ============================================================
print("\n" + "="*60)
print("  5. WORST 20 PREDICTIONS")
print("="*60)

worst_idx = np.argsort(-abs_errors)[:20]
print(f"  {'idx':>5} {'y_true':>8} {'y_pred':>8} {'error':>8} {'glucose':>8} {'bmi':>8} {'trig':>8} {'hdl':>8} {'age':>5} {'sex':>4}")
for i in worst_idx:
    print(f"  {i:5d} {y[i]:8.2f} {oof[i]:8.2f} {residuals[i]:+8.2f} "
          f"{glucose[i]:8.1f} {bmi[i]:8.1f} {trig[i]:8.1f} {hdl[i]:8.1f} "
          f"{age[i]:5.0f} {sex[i]:4.0f}")
sys.stdout.flush()

# ============================================================
# 6. AUTOCORRELATION — are errors structured?
# ============================================================
print("\n" + "="*60)
print("  6. ERROR STRUCTURE")
print("="*60)

# Correlation of residuals with y
r_res_y = np.corrcoef(y, residuals)[0, 1]
print(f"  Correlation(residual, y_true): {r_res_y:.4f}")

# Correlation of residuals with prediction
r_res_pred = np.corrcoef(oof, residuals)[0, 1]
print(f"  Correlation(residual, y_pred): {r_res_pred:.4f}")

# Correlation of abs_error with y
r_ae_y = np.corrcoef(y, abs_errors)[0, 1]
print(f"  Correlation(|error|, y_true): {r_ae_y:.4f}")

# Heteroscedasticity: error variance by prediction quintile
for q in range(5):
    lo_pct = np.percentile(oof, q*20)
    hi_pct = np.percentile(oof, (q+1)*20)
    mask = (oof >= lo_pct) & (oof < hi_pct + 0.001)
    if mask.sum() > 0:
        print(f"  Pred Q{q+1} [{lo_pct:.2f}-{hi_pct:.2f}]: "
              f"n={mask.sum()}, err_std={np.std(residuals[mask]):.3f}, "
              f"err_mean={np.mean(residuals[mask]):+.3f}")
sys.stdout.flush()

# ============================================================
# 7. LEAVE-ONE-FEATURE-GROUP-OUT ANALYSIS  
# ============================================================
print("\n" + "="*60)
print("  7. FEATURE GROUP IMPORTANCE (drop-one R²)")
print("="*60)

# Group features
raw_groups = {
    'glucose_group': ['glucose'],
    'bmi_group': ['bmi'],
    'trig_group': ['triglycerides'],
    'hdl_group': ['hdl'],
    'ldl_group': ['ldl'],
    'total_chol_group': ['total cholesterol'],
    'non_hdl_group': ['non hdl'],
    'chol_hdl_group': ['chol/hdl'],
    'age_group': ['age'],
    'sex_group': ['sex_num'],
    'rhr_group': ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)'],
    'hrv_group': ['HRV (mean)', 'HRV (median)', 'HRV (std)'],
    'steps_group': ['STEPS (mean)', 'STEPS (median)', 'STEPS (std)'],
    'sleep_group': ['SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)'],
    'azm_group': ['AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)'],
}

# Quick 5-fold for speed
quick_splits = get_cv_splits(y, n_splits=5, n_repeats=1)

for grp_name, raw_feats in raw_groups.items():
    # Find all V7 feature indices that depend on these raw features
    drop_indices = []
    for i, fname in enumerate(feat_names):
        # Check if this feature is or depends on any of the raw features
        fname_lower = fname.lower()
        should_drop = False
        for rf in raw_feats:
            rf_lower = rf.lower().replace(' ', '_').replace('(', '').replace(')', '')
            rf_words = rf.lower().split()
            if fname_lower == rf.lower():
                should_drop = True
            elif rf_lower in fname_lower.replace(' ', '_'):
                should_drop = True
            elif all(w in fname_lower for w in rf_words):
                should_drop = True
        if should_drop:
            drop_indices.append(i)
    
    keep_indices = [i for i in range(len(feat_names)) if i not in drop_indices]
    X_dropped = X_v7[:, keep_indices]
    
    # Train with dropped features
    oof_sum_d = np.zeros(n)
    oof_count_d = np.zeros(n)
    for tr, te in quick_splits:
        m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
            subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
            reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
        m.fit(X_dropped[tr], y_log[tr], sample_weight=w_sqrt[tr])
        oof_sum_d[te] += m.predict(X_dropped[te])
        oof_count_d[te] += 1
    oof_d = np.expm1(oof_sum_d / np.clip(oof_count_d, 1, None))
    r2_d = r2_score(y, oof_d)
    print(f"  Drop {grp_name:20s} ({len(drop_indices):2d} feats): R² = {r2_d:.4f} (Δ = {r2_d - r2_score(y, oof):+.4f})")
sys.stdout.flush()

# ============================================================
# 8. INFORMATION CEILING ESTIMATION
# ============================================================
print("\n" + "="*60)
print("  8. INFORMATION CEILING ESTIMATION")
print("="*60)

# Method: Train on FULL data, predict on FULL data (overfit bound)
m_overfit = xgb.XGBRegressor(n_estimators=2000, max_depth=8, learning_rate=0.01,
    subsample=1.0, colsample_bytree=1.0, min_child_weight=1,
    random_state=42, verbosity=0)
m_overfit.fit(X_v7, y_log, sample_weight=w_sqrt)
oof_overfit = np.expm1(m_overfit.predict(X_v7))
print(f"  Overfit R² (train=test): {r2_score(y, oof_overfit):.4f}")
print(f"  This is the UPPER BOUND with these features")

# Noise estimation: how much of y's variance is unpredictable?
# If duplicate/near-duplicate samples exist with different y values, that's noise
# Check: for samples with similar features, how much does y vary?
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=6).fit(StandardScaler().fit_transform(X_v7))
_, indices = nn.kneighbors(StandardScaler().fit_transform(X_v7))

neighbor_var = []
for i in range(n):
    nbr_y = y[indices[i, 1:]]  # exclude self
    neighbor_var.append(np.var(nbr_y))

avg_neighbor_var = np.mean(neighbor_var)
total_var = np.var(y)
noise_ratio = avg_neighbor_var / total_var
theoretical_max = 1 - noise_ratio

print(f"\n  Average neighbor y-variance: {avg_neighbor_var:.4f}")
print(f"  Total y-variance: {total_var:.4f}")
print(f"  Estimated noise ratio: {noise_ratio:.4f}")
print(f"  Theoretical max R²: {theoretical_max:.4f}")
print(f"  Current R²: {r2_score(y, oof):.4f}")
print(f"  Gap to theoretical: {theoretical_max - r2_score(y, oof):.4f}")

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
print("\n" + "="*60)
print(f"  V25 ERROR ANALYSIS COMPLETE ({elapsed:.1f}s)")
print("="*60)

# Save key findings
results = {
    'overall_r2': float(r2_score(y, oof)),
    'mae': float(mean_absolute_error(y, oof)),
    'rmse': float(np.sqrt(mean_squared_error(y, oof))),
    'residual_corr_y': float(r_res_y),
    'residual_corr_pred': float(r_res_pred),
    'noise_ratio': float(noise_ratio),
    'theoretical_max_r2': float(theoretical_max),
    'top_residual_correlations': {fname: float(r) for fname, r, _ in correlations[:10]},
    'elapsed': elapsed,
}
with open('v25_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved v25_results.json")
