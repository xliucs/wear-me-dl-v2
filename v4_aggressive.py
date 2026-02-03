#!/usr/bin/env python3
"""
V4: Aggressive push toward R²=0.65.

Research-informed strategy:
1. HOMA_IR = glucose × insulin / 405 → need better insulin proxies
2. Paper (arxiv:2505.03784) shows:
   - Glucose alone doubles R² (0.212→0.435)
   - BMI is #1 demographic predictor
   - Triglycerides/HDL is a strong IR proxy
   - Time-series embeddings help significantly
3. Without insulin or time-series, we must:
   a) Maximize nonlinear interactions between glucose, lipid panel, BMI
   b) Use target-aware feature engineering
   c) Try quantile/robust regression for heavy-tailed HOMA_IR
   d) Optimally tune XGBoost/LightGBM hyperparameters
   e) Build many diverse models and blend optimally
"""
import numpy as np, pandas as pd, time, warnings, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits, compute_metrics,
                             engineer_all_features, engineer_dw_features,
                             DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor, 
                               RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import pearsonr
from itertools import combinations

print("="*60)
print("  V4: AGGRESSIVE PUSH")
print("="*60)

# Load
X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# ============================================================
# 1. ENHANCED FEATURE ENGINEERING
# ============================================================
print("\n--- Enhanced Feature Engineering ---")

def engineer_v4(X_df, all_cols):
    """V4: Aggressive feature engineering with deep insulin resistance proxies."""
    X = X_df.copy() if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df, columns=all_cols)
    
    # --- Core metabolic indices (proven IR proxies from literature) ---
    g = X['glucose'].clip(lower=1)
    t = X['triglycerides'].clip(lower=1)
    h = X['hdl'].clip(lower=1)
    b = X['bmi']
    l = X['ldl']
    tc = X['total cholesterol']
    nh = X['non hdl']
    ch = X['chol/hdl']
    a = X['age']
    sex = X['sex_num']
    
    # TyG index variants
    X['tyg'] = np.log(t * g / 2)
    X['tyg_bmi'] = X['tyg'] * b
    X['tyg_sq'] = X['tyg'] ** 2
    X['tyg_bmi_sq'] = X['tyg'] * b ** 2
    
    # METS-IR
    X['mets_ir'] = np.log(2*g + t) * b
    X['mets_ir_sq'] = X['mets_ir'] ** 2
    
    # Trig/HDL ratio (strong IR surrogate)
    X['trig_hdl'] = t / h
    X['trig_hdl_sq'] = X['trig_hdl'] ** 2
    X['trig_hdl_log'] = np.log1p(X['trig_hdl'])
    X['trig_hdl_bmi'] = X['trig_hdl'] * b
    
    # Visceral adiposity proxies
    X['vat_proxy'] = b * t / h
    X['vat_proxy_sq'] = X['vat_proxy'] ** 2
    X['vat_proxy_log'] = np.log1p(X['vat_proxy'])
    
    # IR proxy (glucose × BMI × trig / HDL)
    X['ir_proxy'] = g * b * t / (h * 100)
    X['ir_proxy_sq'] = X['ir_proxy'] ** 2
    X['ir_proxy_log'] = np.log1p(X['ir_proxy'])
    X['ir_proxy_sqrt'] = np.sqrt(X['ir_proxy'].clip(lower=0))
    
    # Glucose interactions
    X['glucose_bmi'] = g * b
    X['glucose_bmi_sq'] = g * b ** 2
    X['glucose_sq'] = g ** 2
    X['glucose_cubed'] = g ** 3
    X['glucose_log'] = np.log(g)
    X['glucose_hdl'] = g / h
    X['glucose_trig'] = g * t
    X['glucose_trig_log'] = np.log(g * t)
    X['glucose_age'] = g * a
    X['glucose_non_hdl'] = g * nh
    X['glucose_chol_hdl'] = g * ch
    
    # BMI interactions
    X['bmi_sq'] = b ** 2
    X['bmi_cubed'] = b ** 3
    X['bmi_log'] = np.log(b.clip(lower=1))
    X['bmi_age'] = b * a
    X['bmi_sex'] = b * sex
    X['bmi_trig'] = b * t
    X['bmi_hdl_inv'] = b / h
    X['bmi_trig_hdl'] = b * t / h
    
    # Lipid interactions
    X['ldl_hdl'] = l / h
    X['non_hdl_ratio'] = nh / h
    X['tc_hdl_bmi'] = tc / h * b
    X['trig_tc'] = t / tc.clip(lower=1)
    X['ldl_trig'] = l * t / 1000
    
    # Age interactions
    X['age_sq'] = a ** 2
    X['age_bmi_sq'] = a * b ** 2
    X['age_glucose'] = a * g
    X['age_trig_hdl'] = a * t / h
    
    # --- Wearable features ---
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'
    slp = 'SLEEP Duration (mean)'
    azm = 'AZM Weekly (mean)'
    
    if rhr in X.columns:
        # Wearable × metabolic cross-features (key for capturing hidden signal)
        X['bmi_rhr'] = b * X[rhr]
        X['bmi_hrv_inv'] = b / X[hrv].clip(lower=1)
        X['bmi_stp_inv'] = b / X[stp].clip(lower=1) * 1000
        X['glucose_rhr'] = g * X[rhr]
        X['glucose_hrv_inv'] = g / X[hrv].clip(lower=1)
        X['trig_hdl_rhr'] = X['trig_hdl'] * X[rhr]
        X['ir_proxy_rhr'] = X['ir_proxy'] * X[rhr] / 100
        X['tyg_rhr'] = X['tyg'] * X[rhr]
        X['vat_rhr'] = X['vat_proxy'] * X[rhr] / 100
        
        # Composite health
        X['cardio_fitness'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
        X['met_load'] = b * X[rhr] / X[stp].clip(lower=1) * 1000
        X['sed_risk'] = b ** 2 * X[rhr] / (X[stp].clip(lower=1) * X[hrv].clip(lower=1))
        X['recovery'] = X[hrv] / X[rhr].clip(lower=1) * X[slp]
        
        # Wearable variability
        for pfx, m, md, s in [
            ('rhr', rhr, 'Resting Heart Rate (median)', 'Resting Heart Rate (std)'),
            ('hrv', hrv, 'HRV (median)', 'HRV (std)'),
            ('stp', stp, 'STEPS (median)', 'STEPS (std)'),
            ('slp', slp, 'SLEEP Duration (median)', 'SLEEP Duration (std)'),
            ('azm', azm, 'AZM Weekly (median)', 'AZM Weekly (std)'),
        ]:
            X[f'{pfx}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
            X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
        
        # Rank features
        for col in ['bmi', 'age', rhr, hrv, stp, 'glucose', 'triglycerides', 'hdl']:
            if col in X.columns:
                X[f'rank_{col[:4]}'] = X[col].rank(pct=True)
    
    return X.fillna(0)

X_v4 = engineer_v4(X_df[all_cols], all_cols)
print(f"V4 features: {X_v4.shape[1]}")

# Feature selection
mi = mutual_info_regression(X_v4.values, y, random_state=42)
mi_order = np.argsort(mi)[::-1]

# Multiple feature set sizes
for k in [20, 30, 40, 50]:
    top_k_names = [X_v4.columns[i] for i in mi_order[:k]]
    print(f"  Top {k} MI: {top_k_names[:5]}...")

X_v4_all = X_v4.values
X_v4_mi40 = X_v4.values[:, mi_order[:40]]
X_v4_mi30 = X_v4.values[:, mi_order[:30]]

# ============================================================
# 2. OPTIMIZED XGBoost with HYPERPARAMETER SEARCH
# ============================================================
print("\n--- Hyperparameter Optimization ---")

def get_oof(model_fn, X, y_arr, splits, scale=False, log_target=False):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_arr) if log_target else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

xgb_configs = [
    ('xgb_d3_lr01', dict(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0)),
    ('xgb_d3_lr03', dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05)),
    ('xgb_d4_lr03', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, min_child_weight=5)),
    ('xgb_d5_lr02', dict(n_estimators=500, max_depth=5, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6, min_child_weight=10)),
    ('xgb_d6_lr01', dict(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.6, colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0)),
    ('xgb_d3_huber', dict(n_estimators=400, max_depth=3, learning_rate=0.03, objective='reg:pseudohubererror', subsample=0.8, colsample_bytree=0.8)),
    ('xgb_d4_gamma', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, gamma=0.5)),
]

lgb_configs = [
    ('lgb_d4_lr03', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0)),
    ('lgb_d6_lr01', dict(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.7, colsample_bytree=0.6, min_child_samples=20, reg_alpha=0.1)),
    ('lgb_dart', dict(n_estimators=300, max_depth=4, learning_rate=0.03, boosting_type='dart', subsample=0.8)),
    ('lgb_d3_leaves63', dict(n_estimators=500, num_leaves=63, learning_rate=0.02, subsample=0.7, colsample_bytree=0.7)),
]

hgbr_configs = [
    ('hgbr_d4_lr03', dict(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=10)),
    ('hgbr_d6_lr02', dict(max_iter=500, max_depth=6, learning_rate=0.02, min_samples_leaf=15)),
    ('hgbr_d8_lr01', dict(max_iter=800, max_depth=8, learning_rate=0.01, min_samples_leaf=20)),
]

oof_pool = {}
base_scores = {}

# Test on multiple feature sets
feature_sets = [
    ('v4_all', X_v4_all, False),
    ('v4_mi40', X_v4_mi40, False),
    ('v4_mi30', X_v4_mi30, False),
    ('raw', X_all_raw, False),
]

for fs_name, X_fs, scale_needed in feature_sets:
    print(f"\n  Feature set: {fs_name} ({X_fs.shape[1]} features)")
    
    # XGBoost configs
    for name, params in xgb_configs:
        full_name = f'{name}_{fs_name}'
        t0 = time.time()
        oof = get_oof(lambda p=params: xgb.XGBRegressor(**p, random_state=42, verbosity=0), X_fs, y, splits)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        base_scores[full_name] = r2
        print(f"    {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    
    # HGBR configs (including log target)
    for name, params in hgbr_configs:
        for log_t in [False, True]:
            full_name = f'{name}{"_log" if log_t else ""}_{fs_name}'
            t0 = time.time()
            oof = get_oof(lambda p=params: HistGradientBoostingRegressor(**p, random_state=42), X_fs, y, splits, log_target=log_t)
            r2 = r2_score(y, oof)
            oof_pool[full_name] = oof
            base_scores[full_name] = r2
            print(f"    {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")

# LightGBM (only on key feature sets)
for fs_name, X_fs in [('v4_mi40', X_v4_mi40), ('raw', X_all_raw)]:
    for name, params in lgb_configs:
        full_name = f'{name}_{fs_name}'
        t0 = time.time()
        try:
            oof = get_oof(lambda p=params: lgb.LGBMRegressor(**p, random_state=42, verbose=-1), X_fs, y, splits)
            r2 = r2_score(y, oof)
            oof_pool[full_name] = oof
            base_scores[full_name] = r2
            print(f"    {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    {full_name:40s} FAILED: {e}")

# Linear models on v4 engineered features (with scaling)
for name, factory in [
    ('enet_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ('enet_001', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
    ('lasso_01', lambda: Lasso(alpha=0.1, max_iter=5000)),
    ('lasso_001', lambda: Lasso(alpha=0.01, max_iter=5000)),
    ('ridge_100', lambda: Ridge(alpha=100)),
    ('ridge_1000', lambda: Ridge(alpha=1000)),
    ('bayesian', lambda: BayesianRidge()),
]:
    for fs_name, X_fs in [('v4_mi40', X_v4_mi40), ('v4_all', X_v4_all)]:
        full_name = f'{name}_{fs_name}'
        t0 = time.time()
        oof = get_oof(factory, X_fs, y, splits, scale=True)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        base_scores[full_name] = r2
        print(f"    {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")

# ET and RF on key sets
for fname, Xfs in [('v4_mi40', X_v4_mi40), ('raw', X_all_raw)]:
    for mname, mfactory in [
        ('et200', lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1)),
    ]:
        full_name = f'{mname}_{fname}'
        t0 = time.time()
        oof = get_oof(mfactory, Xfs, y, splits)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        base_scores[full_name] = r2
        print(f"    {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")

print(f"\n  Total models: {len(oof_pool)}")

# ============================================================
# 3. OPTIMAL BLENDING
# ============================================================
print("\n--- Optimal Blending ---")

sorted_models = sorted(base_scores, key=base_scores.get, reverse=True)
print(f"  Top 5: {[(n, f'{base_scores[n]:.4f}') for n in sorted_models[:5]]}")

# Blend top models
for top_k in [5, 8, 10, 15]:
    top_names = sorted_models[:top_k]
    top_oofs = np.column_stack([oof_pool[k] for k in top_names])
    
    best_r2 = -999
    rng = np.random.RandomState(42)
    for _ in range(500000):
        w = rng.dirichlet(np.ones(top_k))
        pred = top_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > best_r2: best_r2 = r2; best_w = w
    
    print(f"  Top-{top_k} blend: R²={best_r2:.4f}")

# ============================================================
# 4. STACKING WITH DIVERSE STACKERS
# ============================================================
print("\n--- Stacking ---")

oof_matrix = np.column_stack([oof_pool[k] for k in sorted(oof_pool.keys())])
print(f"  Stack matrix: {oof_matrix.shape}")

for sname, sfactory in [
    ('ridge_01', lambda: Ridge(alpha=0.1)),
    ('ridge_1', lambda: Ridge(alpha=1)),
    ('ridge_10', lambda: Ridge(alpha=10)),
    ('enet_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ('bayesian', lambda: BayesianRidge()),
    ('hgbr_s', lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=2, learning_rate=0.05, random_state=42)),
]:
    oof = get_oof(sfactory, oof_matrix, y, splits, scale=(not sname.startswith('hgbr')))
    r2 = r2_score(y, oof)
    print(f"  Stack {sname:15s} R²={r2:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("  V4 SUMMARY — TOP 15")
print("="*60)

sorted_all = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
for i, (name, r2) in enumerate(sorted_all[:15], 1):
    print(f"  {i:2d}. {name:45s} R²={r2:.4f}")

best_name, best_r2 = sorted_all[0]
print(f"\n  ★ BEST SINGLE: {best_name} R²={best_r2:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_r2:.4f}")
