#!/usr/bin/env python3
"""
V10: Residual analysis + greedy blend + quantile-specific models.

The key question: WHERE do our models fail?
If we can identify and fix systematic errors, R² jumps.

Strategy:
1. Analyze residuals of best model — find patterns in errors
2. Build "residual corrector" models (meta-learning on errors)
3. Greedy forward selection for blend (optimal, not random)
4. Quantile-specific models: separate for low/medium/high HOMA_IR
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V10: RESIDUAL ANALYSIS + GREEDY BLEND")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# V7 features (compact)
def engineer_v7(X_df, cols):
    X = X_df[cols].copy()
    g = X['glucose'].clip(lower=1); t = X['triglycerides'].clip(lower=1)
    h = X['hdl'].clip(lower=1); b = X['bmi']; l = X['ldl']
    tc = X['total cholesterol']; nh = X['non hdl']; ch = X['chol/hdl']
    a = X['age']; sex = X['sex_num']
    for nm, val in [('tyg', np.log(t*g/2)), ('tyg_bmi', np.log(t*g/2)*b),
        ('mets_ir', np.log(2*g+t)*b/np.log(h)), ('trig_hdl', t/h), ('trig_hdl_log', np.log1p(t/h)),
        ('vat_proxy', b*t/h), ('ir_proxy', g*b*t/(h*100)), ('glucose_bmi', g*b),
        ('glucose_sq', g**2), ('glucose_log', np.log(g)), ('glucose_hdl', g/h),
        ('glucose_trig', g*t/1000), ('glucose_non_hdl', g*nh/100), ('glucose_chol_hdl', g*ch),
        ('bmi_sq', b**2), ('bmi_log', np.log(b.clip(lower=1))), ('bmi_trig', b*t/100),
        ('bmi_hdl_inv', b/h), ('bmi_age', b*a), ('ldl_hdl', l/h), ('non_hdl_ratio', nh/h),
        ('tc_hdl_bmi', tc/h*b), ('trig_tc', t/tc.clip(lower=1))]:
        X[nm] = val
    X['tyg_sq'] = X['tyg']**2; X['mets_ir_sq'] = X['mets_ir']**2
    X['trig_hdl_sq'] = X['trig_hdl']**2; X['vat_sq'] = X['vat_proxy']**2
    X['ir_proxy_sq'] = X['ir_proxy']**2; X['ir_proxy_log'] = np.log1p(X['ir_proxy'])
    rhr='Resting Heart Rate (mean)'; hrv='HRV (mean)'; stp='STEPS (mean)'; slp='SLEEP Duration (mean)'
    if rhr in X.columns:
        for nm, val in [('bmi_rhr', b*X[rhr]), ('glucose_rhr', g*X[rhr]),
            ('trig_hdl_rhr', X['trig_hdl']*X[rhr]), ('ir_proxy_rhr', X['ir_proxy']*X[rhr]/100),
            ('tyg_rhr', X['tyg']*X[rhr]), ('mets_rhr', X['mets_ir']*X[rhr]),
            ('bmi_hrv_inv', b/X[hrv].clip(lower=1)),
            ('cardio_fitness', X[hrv]*X[stp]/X[rhr].clip(lower=1)),
            ('met_load', b*X[rhr]/X[stp].clip(lower=1)*1000)]:
            X[nm] = val
        for pfx,m,s in [('rhr',rhr,'Resting Heart Rate (std)'),('hrv',hrv,'HRV (std)'),
                         ('stp',stp,'STEPS (std)'),('slp',slp,'SLEEP Duration (std)')]:
            if s in X.columns: X[f'{pfx}_cv'] = X[s]/X[m].clip(lower=0.01)
    X['log_glucose']=np.log(g); X['log_trig']=np.log(t)
    X['log_bmi']=np.log(b.clip(lower=1)); X['log_hdl']=np.log(h)
    X['log_homa_proxy']=np.log(g)+np.log(b.clip(lower=1))+np.log(t)-np.log(h)
    return X.fillna(0)

X_v7 = engineer_v7(X_df[all_cols], all_cols).values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values

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

log_fn = np.log1p; inv_log = np.expm1
sqrt_fn = lambda y: np.sqrt(y.clip(min=0)); inv_sqrt = lambda p: p**2

# ============================================================
# PART 1: RESIDUAL ANALYSIS
# ============================================================
print("\n--- Part 1: Residual Analysis ---")
sys.stdout.flush()

# Get OOF from best single model
xgb_d3 = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
               colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01)
oof_best = get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
                    X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log)
residuals = y - oof_best
abs_residuals = np.abs(residuals)

print(f"  Best model OOF R²={r2_score(y, oof_best):.4f}")
print(f"  Residuals: mean={residuals.mean():.4f}, std={residuals.std():.4f}")
print(f"  MAE: {mean_absolute_error(y, oof_best):.4f}")
print(f"  MSE: {np.mean(residuals**2):.4f}")

# Analyze error by HOMA_IR quantile
print(f"\n  Error by HOMA_IR range:")
for lo, hi, label in [(0, 1, 'Low [0-1)'), (1, 2, 'Normal [1-2)'), (2, 3, 'Elevated [2-3)'),
                        (3, 5, 'High [3-5)'), (5, 8, 'Very High [5-8)'), (8, 100, 'Extreme [8+)')]:
    mask = (y >= lo) & (y < hi)
    if mask.sum() > 0:
        r = residuals[mask]
        print(f"    {label:20s} n={mask.sum():4d}  MAE={np.abs(r).mean():.3f}  "
              f"bias={r.mean():+.3f}  RMSE={np.sqrt(np.mean(r**2)):.3f}")
sys.stdout.flush()

# Correlation of errors with features
print(f"\n  Feature correlations with |residual|:")
for col in all_cols[:10]:  # Top raw features
    r = np.corrcoef(X_df[col].values, abs_residuals)[0,1]
    if abs(r) > 0.1:
        print(f"    {col:40s} r={r:.3f}")
sys.stdout.flush()

# ============================================================
# PART 2: RESIDUAL CORRECTION (meta-learning)
# ============================================================
print("\n--- Part 2: Residual Correction ---")
sys.stdout.flush()

# Idea: train a model on residuals, then add correction
# OOF residuals for the best model, then predict residuals with a new model
for res_model_name, res_model_fn, res_fs_name, res_X in [
    ('xgb_d3_resid_v7', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                    random_state=42, verbosity=0), 'v7', X_v7),
    ('hgbr_d3_resid_v7', lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=3,
                                                                  learning_rate=0.05, random_state=42), 'v7', X_v7),
    ('ridge_resid_eng', lambda: Ridge(alpha=100), 'eng', X_eng),
]:
    oof_resid = get_oof(res_model_fn, res_X, residuals, splits, scale=(res_model_name.startswith('ridge')))
    corrected = oof_best + oof_resid
    r2 = r2_score(y, corrected)
    print(f"  {res_model_name:40s} corrected R²={r2:.4f} (residual R²={r2_score(residuals, oof_resid):.4f})")
    sys.stdout.flush()

# ============================================================
# PART 3: BUILD ALL DIVERSE MODELS (for greedy blend)
# ============================================================
print("\n--- Part 3: Building Diverse Pool ---")
sys.stdout.flush()

oof_pool = {}
scores = {}

models_to_build = [
    # XGB log targets
    ('xgb_d3_log_s42_v7', lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0), X_v7, log_fn, inv_log),
    ('xgb_d3_log_s789_v7', lambda: xgb.XGBRegressor(**xgb_d3, random_state=789, verbosity=0), X_v7, log_fn, inv_log),
    ('xgb_d3_log_s2024_v7', lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0), X_v7, log_fn, inv_log),
    ('xgb_d3_log_s42_eng', lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0), X_eng, log_fn, inv_log),
    ('xgb_d6_log_v7', lambda: xgb.XGBRegressor(n_estimators=800, max_depth=6, learning_rate=0.01,
        subsample=0.6, colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0,
        random_state=42, verbosity=0), X_v7, log_fn, inv_log),
    # XGB sqrt target
    ('xgb_d3_sqrt_v7', lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0), X_v7, sqrt_fn, inv_sqrt),
    # XGB raw target
    ('xgb_d3_v7', lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0), X_v7, None, None),
    # LGB
    ('lgb_d3_log_v7', lambda: lgb.LGBMRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=20, verbose=-1, random_state=42),
     X_v7, log_fn, inv_log),
    ('lgb_d4_log_v7', lambda: lgb.LGBMRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=15, verbose=-1, random_state=42),
     X_v7, log_fn, inv_log),
    # HGBR
    ('hgbr_d4_log_v7', lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4,
        learning_rate=0.03, min_samples_leaf=10, random_state=42), X_v7, log_fn, inv_log),
    # ET
    ('et200_v7', lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1), X_v7, None, None),
    ('et200_eng', lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1), X_eng, None, None),
    # Linear
    ('enet_01_eng', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_eng, None, None),
    ('enet_01_v7', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_v7, None, None),
    ('ridge_500_v7', lambda: Ridge(alpha=500), X_v7, None, None),
    ('bayesian_v7', lambda: BayesianRidge(), X_v7, None, None),
]

for name, mfn, X_fs, tfn, ifn in models_to_build:
    t0 = time.time()
    scale = name.startswith(('enet', 'ridge', 'bayesian'))
    oof = get_oof(mfn, X_fs, y, splits, scale=scale, target_fn=tfn, inv_fn=ifn)
    r2 = r2_score(y, oof)
    oof_pool[name] = oof; scores[name] = r2
    print(f"  {name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

# ============================================================
# PART 4: GREEDY FORWARD SELECTION BLEND
# ============================================================
print("\n--- Part 4: Greedy Forward Selection ---")
sys.stdout.flush()

sorted_pool = sorted(scores, key=scores.get, reverse=True)

# Start with best single model
selected = [sorted_pool[0]]
remaining = set(sorted_pool[1:])
current_blend = oof_pool[selected[0]].copy()
current_r2 = scores[selected[0]]
print(f"  Step 1: {selected[0]:40s} R²={current_r2:.4f}")

# Greedily add models that maximize blend R²
for step in range(2, min(len(oof_pool) + 1, 12)):
    best_add = None; best_add_r2 = current_r2
    for name in remaining:
        # Try adding this model with optimized weight
        for alpha in np.arange(0.02, 0.50, 0.02):
            blend = (1 - alpha) * current_blend + alpha * oof_pool[name]
            r2 = r2_score(y, blend)
            if r2 > best_add_r2:
                best_add_r2 = r2; best_add = name; best_alpha = alpha
    
    if best_add is None or best_add_r2 <= current_r2 + 0.0001:
        print(f"  Step {step}: No improvement, stopping")
        break
    
    selected.append(best_add)
    remaining.discard(best_add)
    current_blend = (1 - best_alpha) * current_blend + best_alpha * oof_pool[best_add]
    current_r2 = best_add_r2
    print(f"  Step {step}: +{best_add:35s} α={best_alpha:.2f} → R²={current_r2:.4f}")
    sys.stdout.flush()

print(f"\n  ★ Greedy blend: R²={current_r2:.4f} ({len(selected)} models)")

# Also try Dirichlet on the greedy-selected set
if len(selected) > 1:
    sel_oofs = np.column_stack([oof_pool[k] for k in selected])
    best_dir = -999
    rng = np.random.RandomState(42)
    for _ in range(2000000):
        w = rng.dirichlet(np.ones(len(selected)))
        pred = sel_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > best_dir: best_dir = r2; best_dir_w = w
    print(f"  ★ Dirichlet on greedy set (2M): R²={best_dir:.4f}")
    print(f"  Weights:")
    for nm, w in zip(selected, best_dir_w):
        if w > 0.01: print(f"    {nm:40s} w={w:.3f}")

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  V10 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
best_overall = max(current_r2, best_dir) if len(selected) > 1 else current_r2
print(f"  ★ BEST: R²={best_overall:.4f}")
print(f"  Best single: {sorted_pool[0]} R²={scores[sorted_pool[0]]:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_overall:.4f}")

results = {
    'best_r2': float(best_overall),
    'greedy_r2': float(current_r2),
    'greedy_models': selected,
    'best_single': {'name': sorted_pool[0], 'r2': float(scores[sorted_pool[0]])},
    'elapsed': elapsed
}
with open('v10_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v10_results.json")
sys.stdout.flush()
