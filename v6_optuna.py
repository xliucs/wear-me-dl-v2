#!/usr/bin/env python3
"""
V6: Optuna hyperparameter optimization + right-tail analysis.

Analysis: Why are we stuck at R²=0.51?
- HOMA_IR has heavy right tail: mean=2.43, max=14.82
- High HOMA_IR cases (>5) contribute disproportionately to MSE
- If we can better predict these outliers, R² jumps significantly
- Strategy: use Optuna to find optimal XGB/LGB hyperparams
"""
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits, compute_metrics,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False

print("="*60)
print("  V6: OPTUNA + RIGHT-TAIL ANALYSIS")
print("="*60)

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
X_all_eng_df = engineer_all_features(X_df[all_cols], all_cols)
X_all_eng = X_all_eng_df.values
splits = get_cv_splits(y)
n = len(y)

# ============================================================
# 1. DATA ANALYSIS - Understanding the error distribution
# ============================================================
print("\n--- Data Analysis ---")
print(f"Target distribution:")
for q in [25, 50, 75, 90, 95, 99]:
    print(f"  P{q}: {np.percentile(y, q):.2f}")
print(f"  Skewness: {pd.Series(y).skew():.2f}")
print(f"  Kurtosis: {pd.Series(y).kurtosis():.2f}")

# Correlation analysis
print(f"\nFeature correlations with HOMA_IR (|r| > 0.2):")
for col in all_cols:
    if col == 'sex_num':
        vals = X_df[col].values
    else:
        vals = X_df[col].values
    r = np.corrcoef(vals, y)[0,1]
    if abs(r) > 0.2:
        print(f"  {col:40s} r={r:.3f}")

# ============================================================
# 2. OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================
if HAS_OPTUNA:
    print("\n--- Optuna XGBoost Optimization (100 trials) ---")
    
    def objective_xgb(trial, X, y_arr, splits):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 2),
            'random_state': 42,
            'verbosity': 0,
        }
        
        # Use log target?
        use_log = trial.suggest_categorical('use_log', [True, False])
        
        oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
        yt = np.log1p(y_arr) if use_log else y_arr
        for tr, te in splits:
            m = xgb.XGBRegressor(**params)
            m.fit(X[tr], yt[tr])
            p = m.predict(X[te])
            if use_log: p = np.expm1(p)
            oof_sum[te] += p; oof_cnt[te] += 1
        oof = oof_sum / np.clip(oof_cnt, 1, None)
        return r2_score(y_arr, oof)
    
    # Optimize on raw features
    study_raw = optuna.create_study(direction='maximize')
    study_raw.optimize(lambda t: objective_xgb(t, X_all_raw, y, splits), n_trials=100, show_progress_bar=False)
    print(f"  Best XGB (raw): R²={study_raw.best_value:.4f}")
    print(f"  Params: {study_raw.best_params}")
    
    # Optimize on engineered features
    study_eng = optuna.create_study(direction='maximize')
    study_eng.optimize(lambda t: objective_xgb(t, X_all_eng, y, splits), n_trials=100, show_progress_bar=False)
    print(f"\n  Best XGB (eng): R²={study_eng.best_value:.4f}")
    print(f"  Params: {study_eng.best_params}")
    
    # Optimize LightGBM
    print("\n--- Optuna LightGBM Optimization (100 trials) ---")
    
    def objective_lgb(trial, X, y_arr, splits):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'random_state': 42,
            'verbose': -1,
        }
        
        use_log = trial.suggest_categorical('use_log', [True, False])
        
        oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
        yt = np.log1p(y_arr) if use_log else y_arr
        for tr, te in splits:
            m = lgb.LGBMRegressor(**params)
            m.fit(X[tr], yt[tr])
            p = m.predict(X[te])
            if use_log: p = np.expm1(p)
            oof_sum[te] += p; oof_cnt[te] += 1
        oof = oof_sum / np.clip(oof_cnt, 1, None)
        return r2_score(y_arr, oof)
    
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(lambda t: objective_lgb(t, X_all_raw, y, splits), n_trials=100, show_progress_bar=False)
    print(f"  Best LGB (raw): R²={study_lgb.best_value:.4f}")
    print(f"  Params: {study_lgb.best_params}")
    
    # ============================================================
    # 3. BLEND OPTUNA BEST MODELS
    # ============================================================
    print("\n--- Blending Optuna Best Models ---")
    
    def get_oof_with_params(params_dict, model_class, X, y_arr, splits, use_log=False):
        oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
        yt = np.log1p(y_arr) if use_log else y_arr
        for tr, te in splits:
            m = model_class(**{k:v for k,v in params_dict.items() if k not in ['use_log']})
            m.fit(X[tr], yt[tr])
            p = m.predict(X[te])
            if use_log: p = np.expm1(p)
            oof_sum[te] += p; oof_cnt[te] += 1
        return oof_sum / np.clip(oof_cnt, 1, None)
    
    # Get OOFs from best models
    xgb_raw_params = {k: v for k, v in study_raw.best_params.items() if k != 'use_log'}
    xgb_raw_params.update({'random_state': 42, 'verbosity': 0})
    xgb_raw_log = study_raw.best_params.get('use_log', False)
    oof_xgb_raw = get_oof_with_params(xgb_raw_params, xgb.XGBRegressor, X_all_raw, y, splits, xgb_raw_log)
    
    xgb_eng_params = {k: v for k, v in study_eng.best_params.items() if k != 'use_log'}
    xgb_eng_params.update({'random_state': 42, 'verbosity': 0})
    xgb_eng_log = study_eng.best_params.get('use_log', False)
    oof_xgb_eng = get_oof_with_params(xgb_eng_params, xgb.XGBRegressor, X_all_eng, y, splits, xgb_eng_log)
    
    lgb_raw_params = {k: v for k, v in study_lgb.best_params.items() if k != 'use_log'}
    lgb_raw_params.update({'random_state': 42, 'verbose': -1})
    lgb_raw_log = study_lgb.best_params.get('use_log', False)
    oof_lgb_raw = get_oof_with_params(lgb_raw_params, lgb.LGBMRegressor, X_all_raw, y, splits, lgb_raw_log)
    
    # Also add our previous best: ElasticNet on engineered
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        sc = StandardScaler(); Xtr = sc.fit_transform(X_all_eng[tr]); Xte = sc.transform(X_all_eng[te])
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000); m.fit(Xtr, y[tr])
        oof_sum[te] += m.predict(Xte); oof_cnt[te] += 1
    oof_enet = oof_sum / np.clip(oof_cnt, 1, None)
    
    # Blend
    oofs = [oof_xgb_raw, oof_xgb_eng, oof_lgb_raw, oof_enet]
    names = ['xgb_raw_opt', 'xgb_eng_opt', 'lgb_raw_opt', 'enet_eng']
    for i, (name, oof) in enumerate(zip(names, oofs)):
        print(f"  {name}: R²={r2_score(y, oof):.4f}")
    
    # Dirichlet blend
    top_oofs = np.column_stack(oofs)
    best_blend = -999
    rng = np.random.RandomState(42)
    for _ in range(500000):
        w = rng.dirichlet(np.ones(len(oofs)))
        pred = top_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > best_blend: best_blend = r2; best_w = w
    
    print(f"\n  Blend: R²={best_blend:.4f}")
    print(f"  Weights: {dict(zip(names, [f'{w:.3f}' for w in best_w]))}")
    
    print(f"\n  ★ BEST V6: R²={best_blend:.4f}")
    print(f"  Target: 0.65 | Gap: {0.65 - best_blend:.4f}")
    
else:
    print("Optuna not installed. pip install optuna")
    
    # Fallback: manual grid
    print("\n--- Manual Grid Search ---")
    best_r2 = -999
    configs = [
        dict(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5),
        dict(n_estimators=800, max_depth=4, learning_rate=0.01, subsample=0.6, colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10),
        dict(n_estimators=1000, max_depth=5, learning_rate=0.005, subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=3.0, min_child_weight=15),
    ]
    for i, cfg in enumerate(configs):
        oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
        for tr, te in splits:
            m = xgb.XGBRegressor(**cfg, random_state=42, verbosity=0)
            m.fit(X_all_raw[tr], y[tr])
            oof_sum[te] += m.predict(X_all_raw[te]); oof_cnt[te] += 1
        oof = oof_sum / np.clip(oof_cnt, 1, None)
        r2 = r2_score(y, oof)
        print(f"  Config {i+1}: R²={r2:.4f}")
        if r2 > best_r2: best_r2 = r2
    print(f"\n  Best grid: R²={best_r2:.4f}")
