#!/usr/bin/env python3
"""
V5: Insulin proxy learning.

KEY INSIGHT: HOMA_IR = glucose × insulin / 405
- We HAVE glucose
- We DON'T have insulin
- Strategy: predict insulin_proxy = HOMA_IR / glucose
  (this isolates the unknown insulin component)
- Then: HOMA_IR_pred = insulin_proxy_pred × glucose

This removes the known glucose contribution, letting the model focus on
estimating insulin from BMI, lipids, wearables.
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits, compute_metrics,
                             DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

print("="*60)
print("  V5: INSULIN PROXY LEARNING")
print("="*60)

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

glucose = X_df['glucose'].values

# Create insulin proxy target
# HOMA_IR = glucose * insulin / 405
# insulin_proxy = HOMA_IR / glucose = insulin / 405
insulin_proxy = y / glucose.clip(min=1)

print(f"HOMA_IR:       mean={y.mean():.3f} std={y.std():.3f} range=[{y.min():.2f}, {y.max():.2f}]")
print(f"Insulin proxy: mean={insulin_proxy.mean():.4f} std={insulin_proxy.std():.4f} range=[{insulin_proxy.min():.4f}, {insulin_proxy.max():.4f}]")
print(f"Correlation glucose-HOMA_IR: r={np.corrcoef(glucose, y)[0,1]:.3f}")
print(f"Correlation glucose-insulin_proxy: r={np.corrcoef(glucose, insulin_proxy)[0,1]:.3f}")

# Features for insulin prediction: EXCLUDE glucose (it's used in reconstruction)
# Use: BMI, age, sex, lipids (trig, hdl, ldl, chol, non-hdl, chol/hdl), wearables
non_glucose_cols = [c for c in all_cols if c != 'glucose']
X_no_glucose = X_df[non_glucose_cols].values
print(f"\nFeatures (no glucose): {len(non_glucose_cols)}")

# Also engineer insulin-relevant features (without glucose!)
def engineer_insulin_features(X_df, cols):
    X = X_df[cols].copy()
    b = X['bmi']
    t = X['triglycerides'].clip(lower=1)
    h = X['hdl'].clip(lower=1)
    l = X['ldl']
    tc = X['total cholesterol']
    nh = X['non hdl']
    a = X['age']
    sex = X['sex_num']
    
    # Insulin proxies (without glucose)
    X['trig_hdl'] = t / h
    X['trig_hdl_sq'] = (t/h) ** 2
    X['trig_hdl_log'] = np.log1p(t/h)
    X['trig_hdl_bmi'] = t / h * b
    X['vat'] = b * t / h
    X['vat_sq'] = X['vat'] ** 2
    X['vat_log'] = np.log1p(X['vat'])
    X['bmi_sq'] = b ** 2
    X['bmi_cubed'] = b ** 3
    X['bmi_log'] = np.log(b.clip(lower=1))
    X['bmi_trig'] = b * t
    X['bmi_hdl_inv'] = b / h
    X['bmi_age'] = b * a
    X['bmi_sex'] = b * sex
    X['ldl_hdl'] = l / h
    X['non_hdl_ratio'] = nh / h
    X['tc_hdl_bmi'] = tc / h * b
    X['age_sq'] = a ** 2
    X['age_bmi_sq'] = a * b ** 2
    
    # Wearable interactions (important for insulin sensitivity)
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'
    slp = 'SLEEP Duration (mean)'
    
    if rhr in X.columns:
        X['bmi_rhr'] = b * X[rhr]
        X['bmi_hrv_inv'] = b / X[hrv].clip(lower=1)
        X['trig_hdl_rhr'] = X['trig_hdl'] * X[rhr]
        X['vat_rhr'] = X['vat'] * X[rhr] / 100
        X['cardio'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
        X['met_load'] = b * X[rhr] / X[stp].clip(lower=1) * 1000
        X['sed_risk'] = b**2 * X[rhr] / (X[stp].clip(lower=1) * X[hrv].clip(lower=1))
        
        for pfx, m, md, s in [
            ('rhr', rhr, 'Resting Heart Rate (median)', 'Resting Heart Rate (std)'),
            ('hrv', hrv, 'HRV (median)', 'HRV (std)'),
            ('stp', stp, 'STEPS (median)', 'STEPS (std)'),
            ('slp', slp, 'SLEEP Duration (median)', 'SLEEP Duration (std)'),
            ('azm', 'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)'),
        ]:
            X[f'{pfx}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
            X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    
    return X.fillna(0)

X_insulin_eng = engineer_insulin_features(X_df, non_glucose_cols)
print(f"Engineered insulin features: {X_insulin_eng.shape[1]}")

# ============================================================
# APPROACH 1: Direct HOMA_IR prediction (baseline)
# APPROACH 2: Predict insulin_proxy, reconstruct with glucose
# APPROACH 3: Predict log(insulin_proxy), reconstruct
# ============================================================

def get_oof(model_fn, X, y_arr, splits, scale=False, log_target=False):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_arr) if log_target else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale: sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

models = [
    ('xgb_d3', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), False),
    ('xgb_d4', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), False),
    ('xgb_d6_slow', lambda: xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.6, colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0, random_state=42, verbosity=0), False),
    ('hgbr', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), False),
    ('hgbr_log', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), False),
    ('lgb', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), False),
    ('enet', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), True),
    ('ridge_100', lambda: Ridge(alpha=100), True),
    ('bayesian', lambda: BayesianRidge(), True),
]

print("\n" + "="*60)
print("  APPROACH 1: Direct HOMA_IR prediction (baseline)")
print("="*60)
for mname, mfn, scale in models:
    log_t = mname == 'hgbr_log'
    oof = get_oof(mfn, X_all_raw, y, splits, scale, log_t)
    r2 = r2_score(y, oof)
    print(f"  {mname:20s} R²={r2:.4f}")

print("\n" + "="*60)
print("  APPROACH 2: Predict insulin_proxy → reconstruct HOMA_IR")
print("="*60)
print("  (features: no glucose, target: HOMA_IR/glucose)")
best_r2_a2 = -999
for mname, mfn, scale in models:
    log_t = mname == 'hgbr_log'
    # Predict insulin proxy using non-glucose features
    oof_proxy = get_oof(mfn, X_no_glucose, insulin_proxy, splits, scale, log_t)
    # Reconstruct HOMA_IR
    oof_homa = oof_proxy * glucose
    r2 = r2_score(y, oof_homa)
    print(f"  {mname:20s} R²={r2:.4f}")
    if r2 > best_r2_a2: best_r2_a2 = r2

print(f"\n  Also with engineered features:")
for mname, mfn, scale in models:
    log_t = mname == 'hgbr_log'
    oof_proxy = get_oof(mfn, X_insulin_eng.values, insulin_proxy, splits, scale, log_t)
    oof_homa = oof_proxy * glucose
    r2 = r2_score(y, oof_homa)
    print(f"  {mname+'_eng':20s} R²={r2:.4f}")
    if r2 > best_r2_a2: best_r2_a2 = r2

print("\n" + "="*60)
print("  APPROACH 3: Predict log(HOMA_IR) directly")
print("="*60)
print("  (full features, target: log(HOMA_IR))")
for mname, mfn, scale in models:
    if mname == 'hgbr_log': continue
    oof_log = get_oof(mfn, X_all_raw, y, splits, scale, log_target=True)
    r2 = r2_score(y, oof_log)
    print(f"  {mname:20s} R²={r2:.4f}")

print("\n" + "="*60)
print("  APPROACH 4: Two-stage model")
print("="*60)
print("  Stage 1: predict insulin_proxy from non-glucose features")
print("  Stage 2: predict HOMA_IR from ALL features + insulin_proxy_pred")
best_r2_a4 = -999
for mname, mfn, scale in models[:4]:  # Just top models for speed
    log_t = mname == 'hgbr_log'
    # Stage 1: get OOF insulin proxy predictions
    oof_proxy = get_oof(mfn, X_no_glucose, insulin_proxy, splits, scale, log_t)
    
    # Stage 2: augment features with proxy prediction
    X_augmented = np.column_stack([X_all_raw, oof_proxy.reshape(-1,1)])
    
    for m2name, m2fn, sc2 in [
        ('xgb_d4', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), False),
        ('hgbr', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), False),
        ('enet', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), True),
    ]:
        oof_final = get_oof(m2fn, X_augmented, y, splits, sc2)
        r2 = r2_score(y, oof_final)
        print(f"  {mname}→{m2name}: R²={r2:.4f}")
        if r2 > best_r2_a4: best_r2_a4 = r2

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("  V5 SUMMARY")
print("="*60)
print(f"  Best insulin proxy approach: R²={best_r2_a2:.4f}")
print(f"  Best two-stage approach: R²={best_r2_a4:.4f}")
print(f"  Target: R²=0.65 | Gap: {0.65 - max(best_r2_a2, best_r2_a4):.4f}")
