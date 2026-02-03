#!/usr/bin/env python3
"""
V6b: Fast Optuna — 30 trials only, raw features, XGBoost + LightGBM.
Designed to complete in <5 min.
"""
import numpy as np, pandas as pd, time, warnings, sys
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*60)
print("  V6b: FAST OPTUNA (30 trials)")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
X_all_eng = engineer_all_features(X_df[all_cols], all_cols).values
splits = get_cv_splits(y)
n = len(y)

def evaluate(model_fn, X, y_arr, splits, log_target=False):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_arr) if log_target else y_arr
    for tr, te in splits:
        m = model_fn(); m.fit(X[tr], yt[tr])
        p = m.predict(X[te])
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

# --- XGBoost Optuna (30 trials, raw features) ---
print("\n--- XGB Optuna (30 trials, raw) ---"); sys.stdout.flush()
t0 = time.time()

def obj_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5, log=True),
        'random_state': 42, 'verbosity': 0,
    }
    log_t = trial.suggest_categorical('log_target', [True, False])
    oof = evaluate(lambda: xgb.XGBRegressor(**params), X_all_raw, y, splits, log_t)
    return r2_score(y, oof)

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(obj_xgb, n_trials=30)
print(f"  Best XGB raw: R²={study_xgb.best_value:.4f} ({time.time()-t0:.0f}s)")
print(f"  Params: {study_xgb.best_params}")
sys.stdout.flush()

# --- XGBoost Optuna (30 trials, eng features) ---
print("\n--- XGB Optuna (30 trials, eng) ---"); sys.stdout.flush()
t0 = time.time()

def obj_xgb_eng(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5, log=True),
        'random_state': 42, 'verbosity': 0,
    }
    log_t = trial.suggest_categorical('log_target', [True, False])
    oof = evaluate(lambda: xgb.XGBRegressor(**params), X_all_eng, y, splits, log_t)
    return r2_score(y, oof)

study_xgb_eng = optuna.create_study(direction='maximize')
study_xgb_eng.optimize(obj_xgb_eng, n_trials=30)
print(f"  Best XGB eng: R²={study_xgb_eng.best_value:.4f} ({time.time()-t0:.0f}s)")
print(f"  Params: {study_xgb_eng.best_params}")
sys.stdout.flush()

# --- LightGBM Optuna (30 trials, raw) ---
print("\n--- LGB Optuna (30 trials, raw) ---"); sys.stdout.flush()
t0 = time.time()

def obj_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 5, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 5, log=True),
        'random_state': 42, 'verbose': -1,
    }
    log_t = trial.suggest_categorical('log_target', [True, False])
    oof = evaluate(lambda: lgb.LGBMRegressor(**params), X_all_raw, y, splits, log_t)
    return r2_score(y, oof)

study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(obj_lgb, n_trials=30)
print(f"  Best LGB raw: R²={study_lgb.best_value:.4f} ({time.time()-t0:.0f}s)")
print(f"  Params: {study_lgb.best_params}")
sys.stdout.flush()

# --- Blend best models ---
print("\n--- Blending Optuna Best + ElasticNet ---"); sys.stdout.flush()

# Get OOFs
bp_xgb = {k:v for k,v in study_xgb.best_params.items() if k != 'log_target'}
bp_xgb.update({'random_state': 42, 'verbosity': 0})
oof_xgb = evaluate(lambda: xgb.XGBRegressor(**bp_xgb), X_all_raw, y, splits, study_xgb.best_params['log_target'])

bp_xgb_eng = {k:v for k,v in study_xgb_eng.best_params.items() if k != 'log_target'}
bp_xgb_eng.update({'random_state': 42, 'verbosity': 0})
oof_xgb_eng = evaluate(lambda: xgb.XGBRegressor(**bp_xgb_eng), X_all_eng, y, splits, study_xgb_eng.best_params['log_target'])

bp_lgb = {k:v for k,v in study_lgb.best_params.items() if k != 'log_target'}
bp_lgb.update({'random_state': 42, 'verbose': -1})
oof_lgb = evaluate(lambda: lgb.LGBMRegressor(**bp_lgb), X_all_raw, y, splits, study_lgb.best_params['log_target'])

# ElasticNet baseline
def oof_scaled(mfn, X):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = mfn(); m.fit(Xtr, y[tr])
        oof_sum[te] += m.predict(Xte); oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_enet = oof_scaled(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_all_eng)

oofs = [oof_xgb, oof_xgb_eng, oof_lgb, oof_enet]
names = ['xgb_opt', 'xgb_eng_opt', 'lgb_opt', 'enet_eng']
for nm, oof in zip(names, oofs):
    print(f"  {nm}: R²={r2_score(y, oof):.4f}")

# Dirichlet blend
stack = np.column_stack(oofs)
best_r2 = -999
rng = np.random.RandomState(42)
for _ in range(500000):
    w = rng.dirichlet(np.ones(4))
    pred = stack @ w
    r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
    if r2 > best_r2: best_r2 = r2; best_w = w

print(f"\n  Dirichlet blend: R²={best_r2:.4f}")
print(f"  Weights: {dict(zip(names, [f'{w:.3f}' for w in best_w]))}")

print(f"\n★ BEST V6b: R²={best_r2:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_r2:.4f}")
sys.stdout.flush()
