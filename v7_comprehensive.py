#!/usr/bin/env python3
"""
V7: Comprehensive approach informed by research.

Key insights:
1. Reference repo: R²=0.5948 with 46 blood biomarkers. We only have 7.
2. Paper (arxiv:2503.05119): TabKANet + CatBoost work well for IR prediction.
   Features: age, gender, BMI, pulse, BP, waist, glucose. AUC=0.86 for classification.
3. METS-IR = ln(2*glucose + triglycerides) * BMI / ln(HDL) — proven IR surrogate.
4. Our data: heavy right tail (skewness=2.62). Log transform helps.
5. All models converge to ~0.51 → need DIVERSITY in predictions.

Strategy for V7:
- PART 1: Comprehensive model pool with V4 features (fast, <3 min)
- PART 2: PyTorch neural network with feature gating (from reference)
- PART 3: Mega-blend all approaches
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V7: COMPREHENSIVE (research-informed)")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# ============================================================
# ENHANCED FEATURE ENGINEERING (V7)
# ============================================================
def engineer_v7(X_df, cols):
    """V7 features — research-informed, includes METS-IR."""
    X = X_df[cols].copy() if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df, columns=cols)
    
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
    
    # === PROVEN IR SURROGATES FROM LITERATURE ===
    # TyG index (Simental-Mendia 2008)
    X['tyg'] = np.log(t * g / 2)
    X['tyg_bmi'] = X['tyg'] * b
    
    # METS-IR (Bello-Chavolla 2018) — CORRECT FORMULA
    X['mets_ir'] = np.log(2*g + t) * b / np.log(h)
    
    # TG/HDL ratio (strong IR proxy)
    X['trig_hdl'] = t / h
    X['trig_hdl_log'] = np.log1p(t/h)
    
    # VAT proxy
    X['vat_proxy'] = b * t / h
    
    # IR proxy
    X['ir_proxy'] = g * b * t / (h * 100)
    
    # === GLUCOSE INTERACTIONS ===
    X['glucose_bmi'] = g * b
    X['glucose_sq'] = g ** 2
    X['glucose_log'] = np.log(g)
    X['glucose_hdl'] = g / h
    X['glucose_trig'] = g * t / 1000
    X['glucose_non_hdl'] = g * nh / 100
    X['glucose_chol_hdl'] = g * ch
    
    # === BMI INTERACTIONS ===
    X['bmi_sq'] = b ** 2
    X['bmi_log'] = np.log(b.clip(lower=1))
    X['bmi_trig'] = b * t / 100
    X['bmi_hdl_inv'] = b / h
    X['bmi_age'] = b * a
    
    # === LIPID RATIOS ===
    X['ldl_hdl'] = l / h
    X['non_hdl_ratio'] = nh / h
    X['tc_hdl_bmi'] = tc / h * b
    X['trig_tc'] = t / tc.clip(lower=1)
    
    # === SQUARED/POWER TERMS for nonlinearity ===
    X['tyg_sq'] = X['tyg'] ** 2
    X['mets_ir_sq'] = X['mets_ir'] ** 2
    X['trig_hdl_sq'] = X['trig_hdl'] ** 2
    X['vat_sq'] = X['vat_proxy'] ** 2
    X['ir_proxy_sq'] = X['ir_proxy'] ** 2
    X['ir_proxy_log'] = np.log1p(X['ir_proxy'])
    
    # === WEARABLE INTERACTIONS ===
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'
    slp = 'SLEEP Duration (mean)'
    
    if rhr in X.columns:
        X['bmi_rhr'] = b * X[rhr]
        X['glucose_rhr'] = g * X[rhr]
        X['trig_hdl_rhr'] = X['trig_hdl'] * X[rhr]
        X['ir_proxy_rhr'] = X['ir_proxy'] * X[rhr] / 100
        X['tyg_rhr'] = X['tyg'] * X[rhr]
        X['mets_rhr'] = X['mets_ir'] * X[rhr]
        X['bmi_hrv_inv'] = b / X[hrv].clip(lower=1)
        X['cardio_fitness'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
        X['met_load'] = b * X[rhr] / X[stp].clip(lower=1) * 1000
        
        # Wearable variability (CV = std/mean)
        for pfx, m, s in [
            ('rhr', rhr, 'Resting Heart Rate (std)'),
            ('hrv', hrv, 'HRV (std)'),
            ('stp', stp, 'STEPS (std)'),
            ('slp', slp, 'SLEEP Duration (std)'),
        ]:
            if s in X.columns:
                X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    
    # === LOG TRANSFORMS of key features ===
    X['log_glucose'] = np.log(g)
    X['log_trig'] = np.log(t)
    X['log_bmi'] = np.log(b.clip(lower=1))
    X['log_hdl'] = np.log(h)
    X['log_homa_proxy'] = np.log(g) + np.log(b.clip(lower=1)) + np.log(t) - np.log(h)
    
    return X.fillna(0)

X_v7 = engineer_v7(X_df[all_cols], all_cols)
print(f"V7 features: {X_v7.shape[1]}")

# MI-based feature selection
mi = mutual_info_regression(X_v7.values, y, random_state=42)
mi_order = np.argsort(mi)[::-1]
top_names = [X_v7.columns[i] for i in mi_order[:10]]
print(f"Top 10 MI: {top_names}")
sys.stdout.flush()

X_v7_all = X_v7.values
X_v7_mi35 = X_v7.values[:, mi_order[:35]]
X_v7_mi50 = X_v7.values[:, mi_order[:50]]
X_eng = engineer_all_features(X_df[all_cols], all_cols).values

# ============================================================
# PART 1: FAST MODEL POOL 
# ============================================================
print("\n--- Part 1: Fast Model Pool ---")
sys.stdout.flush()

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

oof_pool = {}
scores = {}

# Key XGBoost configs (informed by Optuna V6b: depth 3, lr ~0.03, log target helps)
xgb_configs = [
    ('xgb_d3_log', dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55, 
                         colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01), True),
    ('xgb_d4_log', dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
                         colsample_bytree=0.7, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0), True),
    ('xgb_d6_log', dict(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.6,
                         colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0), True),
    ('xgb_d3', dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.7,
                     colsample_bytree=0.7, min_child_weight=5), False),
    ('xgb_d4', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                     colsample_bytree=0.7, min_child_weight=10, reg_alpha=0.1), False),
]

# Test on different feature sets
feature_configs = [
    ('v7_all', X_v7_all, False),
    ('v7_mi35', X_v7_mi35, False),
    ('raw', X_all_raw, False),
    ('eng', X_eng, False),
]

for fs_name, X_fs, scale_needed in feature_configs:
    for xname, xparams, log_t in xgb_configs:
        full_name = f'{xname}_{fs_name}'
        t0 = time.time()
        oof = get_oof(lambda p=xparams: xgb.XGBRegressor(**p, random_state=42, verbosity=0),
                      X_fs, y, splits, log_target=log_t)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        scores[full_name] = r2
        print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# HGBR configs
for fs_name, X_fs in [('v7_all', X_v7_all), ('raw', X_all_raw), ('eng', X_eng)]:
    for name, params, log_t in [
        ('hgbr_d4_log', dict(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=10), True),
        ('hgbr_d6_log', dict(max_iter=500, max_depth=6, learning_rate=0.02, min_samples_leaf=15), True),
    ]:
        full_name = f'{name}_{fs_name}'
        t0 = time.time()
        oof = get_oof(lambda p=params: HistGradientBoostingRegressor(**p, random_state=42),
                      X_fs, y, splits, log_target=log_t)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        scores[full_name] = r2
        print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# Linear models (scaled)
for mname, mfn in [
    ('enet_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ('enet_001', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
    ('ridge_100', lambda: Ridge(alpha=100)),
    ('ridge_1k', lambda: Ridge(alpha=1000)),
    ('bayesian', lambda: BayesianRidge()),
    ('huber', lambda: HuberRegressor(max_iter=1000)),
]:
    for fs_name, X_fs in [('v7_all', X_v7_all), ('v7_mi35', X_v7_mi35), ('eng', X_eng)]:
        full_name = f'{mname}_{fs_name}'
        t0 = time.time()
        oof = get_oof(mfn, X_fs, y, splits, scale=True)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        scores[full_name] = r2
        print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# LightGBM
for fs_name, X_fs in [('v7_all', X_v7_all), ('raw', X_all_raw)]:
    for name, params, log_t in [
        ('lgb_d4', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, 
                        colsample_bytree=0.7, min_child_samples=15, verbose=-1), False),
        ('lgb_d4_log', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.7, min_child_samples=15, verbose=-1), True),
    ]:
        full_name = f'{name}_{fs_name}'
        t0 = time.time()
        oof = get_oof(lambda p=params: lgb.LGBMRegressor(**p, random_state=42),
                      X_fs, y, splits, log_target=log_t)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        scores[full_name] = r2
        print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# ET
for fs_name, X_fs in [('v7_mi35', X_v7_mi35), ('raw', X_all_raw)]:
    full_name = f'et200_{fs_name}'
    t0 = time.time()
    oof = get_oof(lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
                  X_fs, y, splits)
    r2 = r2_score(y, oof)
    oof_pool[full_name] = oof
    scores[full_name] = r2
    print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

print(f"\n  Total models: {len(oof_pool)} ({time.time()-t_start:.0f}s elapsed)")

# ============================================================
# PART 2: OPTIMAL BLENDING
# ============================================================
print("\n--- Part 2: Optimal Blending ---")
sys.stdout.flush()

sorted_models = sorted(scores, key=scores.get, reverse=True)
print(f"  Top 10:")
for i, name in enumerate(sorted_models[:10], 1):
    print(f"    {i:2d}. {name:40s} R²={scores[name]:.4f}")

# Dirichlet blend for top 5, 8, 10, 15
for top_k in [5, 8, 10, 15, 20]:
    top_names = sorted_models[:top_k]
    top_oofs = np.column_stack([oof_pool[k] for k in top_names])
    
    best_r2 = -999
    rng = np.random.RandomState(42)
    n_trials = 500000
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(top_k))
        pred = top_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > best_r2: best_r2 = r2; best_w = w
    
    print(f"  Top-{top_k:2d} blend ({n_trials//1000}K): R²={best_r2:.4f}")
    if top_k == 10:
        blend_weights = dict(zip(top_names, best_w))
sys.stdout.flush()

# Stacking
print("\n  Stacking:")
oof_matrix = np.column_stack([oof_pool[k] for k in sorted_models[:20]])
for sname, sfn in [
    ('ridge_1', lambda: Ridge(alpha=1)),
    ('ridge_10', lambda: Ridge(alpha=10)),
    ('enet', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
]:
    oof = get_oof(sfn, oof_matrix, y, splits, scale=True)
    r2 = r2_score(y, oof)
    print(f"    Stack {sname:10s}: R²={r2:.4f}")
sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  V7 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"\n  Top 15 single models:")
for i, name in enumerate(sorted_models[:15], 1):
    print(f"    {i:2d}. {name:40s} R²={scores[name]:.4f}")

best_name = sorted_models[0]
best_r2 = scores[best_name]
print(f"\n  ★ BEST SINGLE: {best_name} R²={best_r2:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_r2:.4f}")

# Save results
results = {
    'best_single': {'name': best_name, 'r2': float(best_r2)},
    'top_10': {n: float(scores[n]) for n in sorted_models[:10]},
    'elapsed_seconds': elapsed,
}
with open('v7_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to v7_results.json")
sys.stdout.flush()
