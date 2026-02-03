#!/usr/bin/env python3
"""
V1: Baseline — Comprehensive single-model evaluation.
Tests 15+ model types on both ALL and DW feature sets.
Goal: Establish strong baselines before stacking/ensembling.
"""
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from eval_framework import (
    load_data, get_feature_sets, get_cv_splits, 
    compute_metrics, print_metrics,
    engineer_all_features, engineer_dw_features,
    DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor, GradientBoostingRegressor,
    RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

print("="*70)
print("  WEAR-ME-DL V2 — V1 BASELINE")
print("="*70)

# Load data
X_df, y, feat_names = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)

# Engineer features
X_all_eng = engineer_all_features(X_df[all_cols], all_cols)
X_dw_eng = engineer_dw_features(X_df[dw_cols], dw_cols)

print(f"\nData: {len(y)} samples")
print(f"Target: mean={y.mean():.2f}, std={y.std():.2f}")
print(f"ALL raw features: {X_all_raw.shape[1]}")
print(f"ALL engineered features: {X_all_eng.shape[1]}")
print(f"DW raw features: {X_dw_raw.shape[1]}")
print(f"DW engineered features: {X_dw_eng.shape[1]}")

# Get standardized splits
splits = get_cv_splits(y)
print(f"CV: 5-fold × 5 repeats = {len(splits)} splits\n")

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def get_models():
    """Return dict of model_name -> model_factory."""
    models = {
        # Linear
        'ridge_1': lambda: Ridge(alpha=1),
        'ridge_10': lambda: Ridge(alpha=10),
        'ridge_100': lambda: Ridge(alpha=100),
        'ridge_1000': lambda: Ridge(alpha=1000),
        'lasso_01': lambda: Lasso(alpha=0.1, max_iter=5000),
        'elasticnet': lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
        'bayesian_ridge': lambda: BayesianRidge(),
        'huber': lambda: HuberRegressor(max_iter=500),
        
        # Tree-based
        'xgb_d3': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, 
                                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        'xgb_d4': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        'xgb_d6': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03,
                                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),
        'xgb_mae': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                              objective='reg:absoluteerror', subsample=0.8, 
                                              colsample_bytree=0.8, random_state=42, verbosity=0),
        'hgbr': lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42),
        'hgbr_deep': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=6, learning_rate=0.03, random_state=42),
        'rf_300': lambda: RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=1),
        'et_300': lambda: ExtraTreesRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=1),
        'gbr': lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
        
        # Kernel
        'kr_rbf': lambda: KernelRidge(alpha=1.0, kernel='rbf', gamma=None),
        'kr_poly2': lambda: KernelRidge(alpha=1.0, kernel='poly', degree=2),
        'svr_rbf': lambda: SVR(kernel='rbf', C=10, epsilon=0.1),
        
        # Neighbor
        'knn_10': lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'),
        'knn_20': lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'),
    }
    
    if HAS_LGB:
        models['lgb_gbdt'] = lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, 
                                                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        models['lgb_dart'] = lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                                         boosting_type='dart', subsample=0.8, random_state=42, verbose=-1)
    
    return models

# =============================================================================
# EVALUATION LOOP
# =============================================================================
def evaluate_all_models(X_raw, X_eng, y, splits, feature_set_name, use_log_target=False):
    """Evaluate all models on raw and engineered features."""
    models = get_models()
    results = []
    
    for feat_name, X in [('raw', X_raw), ('eng', X_eng.values if hasattr(X_eng, 'values') else X_eng)]:
        for model_name, model_fn in models.items():
            t0 = time.time()
            n = len(y)
            oof_sum = np.zeros(n)
            oof_count = np.zeros(n)
            
            y_train_target = np.log1p(y) if use_log_target else y
            
            needs_scale = model_name.startswith(('ridge', 'lasso', 'elastic', 'bayesian', 'huber', 
                                                  'kr_', 'svr', 'knn', 'nusvr'))
            
            try:
                for train_idx, test_idx in splits:
                    X_tr, X_te = X[train_idx], X[test_idx]
                    y_tr = y_train_target[train_idx]
                    
                    if needs_scale:
                        scaler = StandardScaler()
                        X_tr = scaler.fit_transform(X_tr)
                        X_te = scaler.transform(X_te)
                    
                    model = model_fn()
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    
                    if use_log_target:
                        preds = np.expm1(preds)
                    
                    oof_sum[test_idx] += preds
                    oof_count[test_idx] += 1
                
                oof_preds = oof_sum / np.clip(oof_count, 1, None)
                r2 = r2_score(y, oof_preds)
                elapsed = time.time() - t0
                
                results.append({
                    'feature_set': feature_set_name,
                    'features': feat_name,
                    'model': model_name,
                    'log_target': use_log_target,
                    'R2': r2,
                    'time': elapsed,
                    'oof_preds': oof_preds,
                })
                
                tag = f"{'log_' if use_log_target else ''}{feat_name}"
                print(f"  {model_name:20s} [{tag:8s}] R²={r2:.4f}  ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"  {model_name:20s} [{feat_name:8s}] FAILED: {e}")
    
    return results


# =============================================================================
# RUN: MODEL A (ALL FEATURES)
# =============================================================================
print("\n" + "="*70)
print("  MODEL A: ALL FEATURES (demographics + wearables + blood biomarkers)")
print("="*70)

results_all = []
print("\n--- Standard target ---")
results_all.extend(evaluate_all_models(X_all_raw, X_all_eng, y, splits, 'ALL'))
print("\n--- Log-transformed target ---")
results_all.extend(evaluate_all_models(X_all_raw, X_all_eng, y, splits, 'ALL', use_log_target=True))

# Best ALL
best_all = max(results_all, key=lambda x: x['R2'])
print(f"\n{'*'*60}")
print(f"  BEST ALL: {best_all['model']} [{best_all['features']}] "
      f"{'(log)' if best_all['log_target'] else ''} R²={best_all['R2']:.4f}")
print(f"{'*'*60}")

m = compute_metrics(y, best_all['oof_preds'])
print(f"  Pearson r: {m['Pearson_r']:.4f}")
print(f"  MAE: {m['MAE']:.4f}")
print(f"  RMSE: {m['RMSE']:.4f}")

# =============================================================================
# RUN: MODEL B (DW FEATURES)
# =============================================================================
print("\n" + "="*70)
print("  MODEL B: DW FEATURES (demographics + wearables only)")
print("="*70)

results_dw = []
print("\n--- Standard target ---")
results_dw.extend(evaluate_all_models(X_dw_raw, X_dw_eng, y, splits, 'DW'))
print("\n--- Log-transformed target ---")
results_dw.extend(evaluate_all_models(X_dw_raw, X_dw_eng, y, splits, 'DW', use_log_target=True))

# Best DW
best_dw = max(results_dw, key=lambda x: x['R2'])
print(f"\n{'*'*60}")
print(f"  BEST DW: {best_dw['model']} [{best_dw['features']}] "
      f"{'(log)' if best_dw['log_target'] else ''} R²={best_dw['R2']:.4f}")
print(f"{'*'*60}")

m = compute_metrics(y, best_dw['oof_preds'])
print(f"  Pearson r: {m['Pearson_r']:.4f}")
print(f"  MAE: {m['MAE']:.4f}")
print(f"  RMSE: {m['RMSE']:.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("  SUMMARY — TOP 10 MODELS")
print("="*70)

all_results = results_all + results_dw
all_results.sort(key=lambda x: x['R2'], reverse=True)

print(f"\n{'Rank':>4} {'Set':>4} {'Features':>8} {'Model':>20} {'Log':>4} {'R²':>8}")
print("-" * 60)
for i, r in enumerate(all_results[:20], 1):
    print(f"{i:4d} {r['feature_set']:>4} {r['features']:>8} {r['model']:>20} "
          f"{'Y' if r['log_target'] else 'N':>4} {r['R2']:8.4f}")

# Save results
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'oof_preds'} for r in all_results])
results_df.to_csv('v1_results.csv', index=False)
print(f"\nResults saved to v1_results.csv")
