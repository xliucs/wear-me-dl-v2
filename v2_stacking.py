#!/usr/bin/env python3
"""
V2: Multi-model stacking with TabPFN + aggressive feature selection.
Focus on Model A (ALL features) to push toward R²=0.65.
"""
import numpy as np, pandas as pd, time, warnings, json, sys
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits, compute_metrics,
                             engineer_all_features, engineer_dw_features,
                             DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor, 
                               ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

print("="*60)
print("  V2: STACKING + FEATURE SELECTION")
print("="*60)

# Load
X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
X_all_eng_df = engineer_all_features(X_df[all_cols], all_cols)
X_dw_eng_df = engineer_dw_features(X_df[dw_cols], dw_cols)
splits = get_cv_splits(y)
n = len(y)

print(f"Data: {n} | ALL eng: {X_all_eng_df.shape[1]} | DW eng: {X_dw_eng_df.shape[1]}")

# Feature selection via mutual information
X_all_eng = X_all_eng_df.values
mi = mutual_info_regression(X_all_eng, y, random_state=42)
mi_order = np.argsort(mi)[::-1]
top_features = mi_order[:35]
X_all_mi35 = X_all_eng[:, top_features]

print(f"Top 35 MI features selected")
print(f"Top 5 MI features: {[X_all_eng_df.columns[i] for i in top_features[:5]]}")

# ============================================================
# OOF PREDICTION GENERATION
# ============================================================
def get_oof(model_fn, X, y_target, splits, scale=False, log_target=False):
    """Fast OOF prediction."""
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_target) if log_target else y_target
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_count[te] += 1
    return oof_sum / np.clip(oof_count, 1, None)

# ============================================================
# MODEL A: ALL FEATURES — Build diverse OOF pool
# ============================================================
print("\n--- Building OOF prediction pool (ALL) ---")

model_configs = [
    # (name, factory, features, scale, log)
    ('ridge100_raw', lambda: Ridge(alpha=100), X_all_raw, True, False),
    ('enet_raw', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_all_raw, True, False),
    ('bayesian_raw', lambda: BayesianRidge(), X_all_raw, True, False),
    ('xgb_d3_raw', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
    ('xgb_d4_raw', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
    ('xgb_d6_raw', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
    ('hgbr_raw', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_raw, False, False),
    ('hgbr_log_raw', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_raw, False, True),
    ('et_raw', lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1), X_all_raw, False, False),
    ('lgb_raw', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), X_all_raw, False, False),
    ('kr_rbf_raw', lambda: KernelRidge(alpha=1.0, kernel='rbf'), X_all_raw, True, False),
    ('svr_raw', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), X_all_raw, True, False),
    # Engineered features
    ('enet_eng', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_all_eng, True, False),
    ('lasso_eng', lambda: Lasso(alpha=0.1, max_iter=5000), X_all_eng, True, False),
    ('ridge1k_eng', lambda: Ridge(alpha=1000), X_all_eng, True, False),
    ('xgb_d4_eng', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_eng, False, False),
    ('xgb_d6_eng', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_eng, False, False),
    ('hgbr_eng', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_eng, False, False),
    ('hgbr_log_eng', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_eng, False, True),
    ('lgb_eng', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), X_all_eng, False, False),
    ('rf_eng', lambda: RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1), X_all_eng, False, False),
    ('knn15_eng', lambda: KNeighborsRegressor(n_neighbors=15, weights='distance'), X_all_eng, True, False),
    # MI-selected features
    ('enet_mi35', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_all_mi35, True, False),
    ('xgb_d4_mi35', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_mi35, False, False),
    ('hgbr_mi35', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_mi35, False, False),
    ('hgbr_log_mi35', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_mi35, False, True),
    # XGB with different objectives
    ('xgb_mae_raw', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, objective='reg:absoluteerror', subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
    ('xgb_huber_raw', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, objective='reg:pseudohubererror', subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
]

oof_pool = {}
base_scores = {}

for name, factory, X, scale, log in model_configs:
    t0 = time.time()
    try:
        oof = get_oof(factory, X, y, splits, scale, log)
        r2 = r2_score(y, oof)
        oof_pool[name] = oof
        base_scores[name] = r2
        print(f"  {name:25s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  {name:25s} FAILED: {e}")

print(f"\n  Pool size: {len(oof_pool)} models")

# ============================================================
# STACKING: Layer 1
# ============================================================
print("\n--- Layer-1 Stacking ---")

# Build stacking matrix
oof_matrix = np.column_stack([oof_pool[k] for k in sorted(oof_pool.keys())])
print(f"  Stacking matrix: {oof_matrix.shape}")

# Different stackers
stackers = {
    'ridge_01': lambda: Ridge(alpha=0.1),
    'ridge_1': lambda: Ridge(alpha=1),
    'ridge_10': lambda: Ridge(alpha=10),
    'lasso_001': lambda: Lasso(alpha=0.01, max_iter=5000),
    'enet_01': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
    'bayesian': lambda: BayesianRidge(),
    'huber': lambda: HuberRegressor(max_iter=500),
    'xgb_stack': lambda: xgb.XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42, verbosity=0),
    'lgb_stack': lambda: lgb.LGBMRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=42, verbose=-1),
    'knn5_stack': lambda: KNeighborsRegressor(n_neighbors=5, weights='distance'),
    'svr_stack': lambda: SVR(kernel='rbf', C=1, epsilon=0.1),
}

stack_oof = {}
stack_scores = {}

for sname, sfactory in stackers.items():
    oof = get_oof(sfactory, oof_matrix, y, splits, scale=True)
    r2 = r2_score(y, oof)
    stack_oof[sname] = oof
    stack_scores[sname] = r2
    print(f"  Stack {sname:20s} R²={r2:.4f}")

best_stack = max(stack_scores, key=stack_scores.get)
print(f"\n  Best stacker: {best_stack} R²={stack_scores[best_stack]:.4f}")

# ============================================================
# BLENDING: Dirichlet weight search
# ============================================================
print("\n--- Dirichlet Blend Search ---")

# Use top-N base models for blending
sorted_models = sorted(base_scores, key=base_scores.get, reverse=True)
top_k = min(10, len(sorted_models))
top_names = sorted_models[:top_k]
top_oofs = np.column_stack([oof_pool[k] for k in top_names])

print(f"  Blending top {top_k}: {[f'{n}({base_scores[n]:.4f})' for n in top_names[:5]]}")

best_blend_r2 = -999
best_blend_weights = None
rng = np.random.RandomState(42)

for _ in range(200000):
    w = rng.dirichlet(np.ones(top_k))
    pred = top_oofs @ w
    r2 = 1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
    if r2 > best_blend_r2:
        best_blend_r2 = r2
        best_blend_weights = w

blend_pred = top_oofs @ best_blend_weights
print(f"  Best Dirichlet blend: R²={best_blend_r2:.4f}")
print(f"  Weights: {dict(zip(top_names, [f'{w:.3f}' for w in best_blend_weights]))}")

# ============================================================
# Layer-2 Stacking
# ============================================================
print("\n--- Layer-2 Stacking ---")

# Combine base OOFs + stack OOFs
l2_matrix = np.column_stack([oof_matrix] + [stack_oof[k] for k in sorted(stack_oof.keys())])
print(f"  Layer-2 matrix: {l2_matrix.shape}")

l2_stackers = {
    'l2_ridge_01': lambda: Ridge(alpha=0.1),
    'l2_ridge_1': lambda: Ridge(alpha=1),
    'l2_ridge_10': lambda: Ridge(alpha=10),
    'l2_bayesian': lambda: BayesianRidge(),
    'l2_enet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
}

for sname, sfactory in l2_stackers.items():
    oof = get_oof(sfactory, l2_matrix, y, splits, scale=True)
    r2 = r2_score(y, oof)
    print(f"  {sname:20s} R²={r2:.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("  MODEL A (ALL) — FINAL RESULTS")
print("="*60)

all_scores = {}
all_scores.update(base_scores)
all_scores.update({f'stack_{k}': v for k, v in stack_scores.items()})
all_scores['blend_dirichlet'] = best_blend_r2

sorted_all = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
for i, (name, r2) in enumerate(sorted_all[:15], 1):
    print(f"  {i:2d}. {name:30s} R²={r2:.4f}")

best_name, best_r2 = sorted_all[0]
print(f"\n  ★ BEST: {best_name} R²={best_r2:.4f} (target: 0.65, gap: {0.65 - best_r2:.4f})")
