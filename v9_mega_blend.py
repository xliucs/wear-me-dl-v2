#!/usr/bin/env python3
"""
V9: Ultimate mega-blend.

Key insight from V8b: blending similar models (XGB d3 x5 seeds) doesn't help.
V7's 0.5368 came from mixing XGB + LGB + HGBR + ElasticNet (diverse types).

Strategy: Build the most DIVERSE pool possible, then blend optimally.
- Different model families (XGB, LGB, HGBR, GBR, ET, Ridge, ElasticNet)
- Different feature sets (raw, eng, v7)
- Different targets (raw, log, sqrt)
- Different hyperparams (shallow/deep, fast/slow learning)
- 2M Dirichlet trials for optimal weights
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V9: ULTIMATE MEGA-BLEND")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# V7 features
def engineer_v7(X_df, cols):
    X = X_df[cols].copy() if isinstance(X_df, pd.DataFrame) else pd.DataFrame(X_df, columns=cols)
    g = X['glucose'].clip(lower=1); t = X['triglycerides'].clip(lower=1)
    h = X['hdl'].clip(lower=1); b = X['bmi']; l = X['ldl']
    tc = X['total cholesterol']; nh = X['non hdl']; ch = X['chol/hdl']
    a = X['age']; sex = X['sex_num']
    X['tyg'] = np.log(t * g / 2); X['tyg_bmi'] = X['tyg'] * b
    X['mets_ir'] = np.log(2*g + t) * b / np.log(h)
    X['trig_hdl'] = t / h; X['trig_hdl_log'] = np.log1p(t/h)
    X['vat_proxy'] = b * t / h; X['ir_proxy'] = g * b * t / (h * 100)
    X['glucose_bmi'] = g * b; X['glucose_sq'] = g**2; X['glucose_log'] = np.log(g)
    X['glucose_hdl'] = g / h; X['glucose_trig'] = g * t / 1000
    X['glucose_non_hdl'] = g * nh / 100; X['glucose_chol_hdl'] = g * ch
    X['bmi_sq'] = b**2; X['bmi_log'] = np.log(b.clip(lower=1))
    X['bmi_trig'] = b * t / 100; X['bmi_hdl_inv'] = b / h; X['bmi_age'] = b * a
    X['ldl_hdl'] = l / h; X['non_hdl_ratio'] = nh / h
    X['tc_hdl_bmi'] = tc / h * b; X['trig_tc'] = t / tc.clip(lower=1)
    X['tyg_sq'] = X['tyg']**2; X['mets_ir_sq'] = X['mets_ir']**2
    X['trig_hdl_sq'] = X['trig_hdl']**2; X['vat_sq'] = X['vat_proxy']**2
    X['ir_proxy_sq'] = X['ir_proxy']**2; X['ir_proxy_log'] = np.log1p(X['ir_proxy'])
    rhr = 'Resting Heart Rate (mean)'; hrv = 'HRV (mean)'
    stp = 'STEPS (mean)'; slp = 'SLEEP Duration (mean)'
    if rhr in X.columns:
        X['bmi_rhr'] = b * X[rhr]; X['glucose_rhr'] = g * X[rhr]
        X['trig_hdl_rhr'] = X['trig_hdl'] * X[rhr]
        X['ir_proxy_rhr'] = X['ir_proxy'] * X[rhr] / 100
        X['tyg_rhr'] = X['tyg'] * X[rhr]; X['mets_rhr'] = X['mets_ir'] * X[rhr]
        X['bmi_hrv_inv'] = b / X[hrv].clip(lower=1)
        X['cardio_fitness'] = X[hrv] * X[stp] / X[rhr].clip(lower=1)
        X['met_load'] = b * X[rhr] / X[stp].clip(lower=1) * 1000
        for pfx, m, s in [('rhr', rhr, 'Resting Heart Rate (std)'),
                           ('hrv', hrv, 'HRV (std)'), ('stp', stp, 'STEPS (std)'),
                           ('slp', slp, 'SLEEP Duration (std)')]:
            if s in X.columns: X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    X['log_glucose'] = np.log(g); X['log_trig'] = np.log(t)
    X['log_bmi'] = np.log(b.clip(lower=1)); X['log_hdl'] = np.log(h)
    X['log_homa_proxy'] = np.log(g) + np.log(b.clip(lower=1)) + np.log(t) - np.log(h)
    return X.fillna(0)

X_v7 = engineer_v7(X_df[all_cols], all_cols).values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values
print(f"Features: raw={X_all_raw.shape[1]}, eng={X_eng.shape[1]}, v7={X_v7.shape[1]}")
sys.stdout.flush()

def get_oof(model_fn, X, y_arr, splits, scale=False, target_fn=None, inv_fn=None):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = target_fn(y_arr) if target_fn else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale: sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if inv_fn: p = inv_fn(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

log_fn = np.log1p
inv_log = np.expm1
sqrt_fn = lambda y: np.sqrt(y.clip(min=0))
inv_sqrt = lambda p: p ** 2

oof_pool = {}
scores = {}
model_count = 0

def add_model(name, oof):
    global model_count
    r2 = r2_score(y, oof)
    oof_pool[name] = oof; scores[name] = r2; model_count += 1
    print(f"  [{model_count:2d}] {name:50s} R²={r2:.4f}")
    sys.stdout.flush()

# ============================================================
# DIVERSE MODEL POOL — maximize prediction diversity!
# ============================================================

# --- XGBoost family (log target) ---
for seed in [42, 789, 2024]:
    xgb_d3 = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
                   colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01)
    add_model(f'xgb_d3_log_s{seed}_v7',
              get_oof(lambda s=seed: xgb.XGBRegressor(**xgb_d3, random_state=s, verbosity=0),
                      X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
    add_model(f'xgb_d3_log_s{seed}_eng',
              get_oof(lambda s=seed: xgb.XGBRegressor(**xgb_d3, random_state=s, verbosity=0),
                      X_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))

# Deeper XGB
xgb_d6 = dict(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.6,
               colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0)
add_model('xgb_d6_log_v7',
          get_oof(lambda: xgb.XGBRegressor(**xgb_d6, random_state=42, verbosity=0),
                  X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))

# XGB sqrt target
add_model('xgb_d3_sqrt_v7',
          get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0),
                  X_v7, y, splits, target_fn=sqrt_fn, inv_fn=inv_sqrt))

# XGB raw target (different perspective)
add_model('xgb_d3_raw_v7',
          get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0),
                  X_v7, y, splits))

# --- LightGBM family ---
lgb_d4 = dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
              colsample_bytree=0.7, min_child_samples=15, verbose=-1)
lgb_d3 = dict(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.7,
              colsample_bytree=0.7, min_child_samples=20, verbose=-1)
add_model('lgb_d4_log_v7', get_oof(lambda: lgb.LGBMRegressor(**lgb_d4, random_state=42),
                                     X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
add_model('lgb_d3_log_v7', get_oof(lambda: lgb.LGBMRegressor(**lgb_d3, random_state=42),
                                     X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
add_model('lgb_d4_log_eng', get_oof(lambda: lgb.LGBMRegressor(**lgb_d4, random_state=42),
                                      X_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))
add_model('lgb_d4_log_raw', get_oof(lambda: lgb.LGBMRegressor(**lgb_d4, random_state=42),
                                      X_all_raw, y, splits, target_fn=log_fn, inv_fn=inv_log))

# --- HGBR family ---
hgbr_d4 = dict(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=10)
add_model('hgbr_d4_log_v7', get_oof(lambda: HistGradientBoostingRegressor(**hgbr_d4, random_state=42),
                                      X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
add_model('hgbr_d4_log_eng', get_oof(lambda: HistGradientBoostingRegressor(**hgbr_d4, random_state=42),
                                       X_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))
add_model('hgbr_d4_log_raw', get_oof(lambda: HistGradientBoostingRegressor(**hgbr_d4, random_state=42),
                                       X_all_raw, y, splits, target_fn=log_fn, inv_fn=inv_log))

# --- ExtraTrees (very different from boosting) ---
add_model('et200_v7', get_oof(lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
                                X_v7, y, splits))
add_model('et200_eng', get_oof(lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
                                 X_eng, y, splits))

# --- Linear family (scaled) ---
add_model('enet_01_v7', get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
                                  X_v7, y, splits, scale=True))
add_model('enet_01_eng', get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
                                   X_eng, y, splits, scale=True))
add_model('ridge_500_v7', get_oof(lambda: Ridge(alpha=500), X_v7, y, splits, scale=True))
add_model('ridge_100_eng', get_oof(lambda: Ridge(alpha=100), X_eng, y, splits, scale=True))
add_model('bayesian_v7', get_oof(lambda: BayesianRidge(), X_v7, y, splits, scale=True))

print(f"\n  Total: {model_count} diverse models ({time.time()-t_start:.0f}s)")
sys.stdout.flush()

# ============================================================
# MEGA-BLEND with strategic model selection
# ============================================================
print("\n--- Mega-Blend ---")
sys.stdout.flush()

sorted_models = sorted(scores, key=scores.get, reverse=True)
print(f"  Top 15:")
for i, name in enumerate(sorted_models[:15], 1):
    print(f"    {i:2d}. {name:50s} R²={scores[name]:.4f}")

# Strategic blend: pick 1 best from each family
families = {
    'xgb_log': [k for k in oof_pool if k.startswith('xgb') and 'log' in k],
    'xgb_other': [k for k in oof_pool if k.startswith('xgb') and 'log' not in k],
    'lgb': [k for k in oof_pool if k.startswith('lgb')],
    'hgbr': [k for k in oof_pool if k.startswith('hgbr')],
    'et': [k for k in oof_pool if k.startswith('et')],
    'linear': [k for k in oof_pool if any(k.startswith(p) for p in ['enet', 'ridge', 'bayesian'])],
}

print("\n  Family bests:")
family_best = {}
for fname, members in families.items():
    if members:
        best = max(members, key=lambda k: scores[k])
        family_best[fname] = best
        print(f"    {fname:12s}: {best:40s} R²={scores[best]:.4f}")

# Blend top from each family
family_names = list(family_best.values())
family_oofs = np.column_stack([oof_pool[k] for k in family_names])
best_r2 = -999
rng = np.random.RandomState(42)
for _ in range(2000000):
    w = rng.dirichlet(np.ones(len(family_names)))
    pred = family_oofs @ w
    r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
    if r2 > best_r2: best_r2 = r2; best_w = w
print(f"\n  ★ Family blend (2M): R²={best_r2:.4f}")
print(f"  Weights:")
for nm, w in zip(family_names, best_w):
    if w > 0.01: print(f"    {nm:40s} w={w:.3f}")

# Also try all models Dirichlet
for top_k in [10, 15, 20, len(sorted_models)]:
    if top_k > len(sorted_models): top_k = len(sorted_models)
    top_names = sorted_models[:top_k]
    top_oofs = np.column_stack([oof_pool[k] for k in top_names])
    bst = -999
    for _ in range(1000000):
        w = rng.dirichlet(np.ones(top_k))
        pred = top_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > bst: bst = r2
    print(f"  Top-{top_k:2d} Dirichlet (1M): R²={bst:.4f}")
    if bst > best_r2: best_r2 = bst
sys.stdout.flush()

# Stacking
print("\n  Stacking:")
all_oofs = np.column_stack([oof_pool[k] for k in sorted_models])
for sname, sfn in [
    ('ridge_10', lambda: Ridge(alpha=10)),
    ('enet_05', lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
]:
    oof = get_oof(sfn, all_oofs, y, splits, scale=True)
    r2 = r2_score(y, oof)
    print(f"    Stack {sname:10s}: R²={r2:.4f}")
    if r2 > best_r2: best_r2 = r2
sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  V9 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ BEST OVERALL: R²={best_r2:.4f}")
print(f"  Best single: {sorted_models[0]} R²={scores[sorted_models[0]]:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_r2:.4f}")

results = {
    'best_r2': float(best_r2),
    'best_single': {'name': sorted_models[0], 'r2': float(scores[sorted_models[0]])},
    'all_scores': {k: float(scores[k]) for k in sorted_models},
    'elapsed': elapsed
}
with open('v9_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v9_results.json")
sys.stdout.flush()
