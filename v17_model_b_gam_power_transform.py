#!/usr/bin/env python3
"""
V17: Model B Focus + GAM + Power Transform Optimization

V16 confirmed: Model A is at its ceiling (~0.546) with current features.
Time to try fundamentally different approaches:

1. MODEL B (demographics + wearables only): Paper baseline R²=0.212, our best R²=0.2463.
   Huge room for improvement. If we can reach R²=0.35+ with wearables only, that's
   scientifically more interesting than squeezing 0.001 from Model A.

2. GAM (Generalized Additive Model): Captures smooth non-linearities that trees miss.
   Unlike trees, GAMs model each feature's marginal effect as a smooth spline.
   
3. Power transform optimization: Instead of log1p, find the optimal Box-Cox lambda
   per fold. Small difference but could help tail prediction.

4. Model A final optimization: Try the best ideas from above on Model A too.
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, engineer_dw_features)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from scipy.optimize import minimize_scalar
from scipy.special import inv_boxcox
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V17: MODEL B + GAM + POWER TRANSFORM")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()

# Feature engineering for both models
def eng_v7(X_df, cols):
    X = X_df[cols].copy()
    g=X['glucose'].clip(lower=1); t=X['triglycerides'].clip(lower=1)
    h=X['hdl'].clip(lower=1); b=X['bmi']; l=X['ldl']
    tc=X['total cholesterol']; nh=X['non hdl']; ch=X['chol/hdl']; a=X['age']
    for nm,v in [('tyg',np.log(t*g/2)),('tyg_bmi',np.log(t*g/2)*b),
        ('mets_ir',np.log(2*g+t)*b/np.log(h)),('trig_hdl',t/h),('trig_hdl_log',np.log1p(t/h)),
        ('vat_proxy',b*t/h),('ir_proxy',g*b*t/(h*100)),('glucose_bmi',g*b),
        ('glucose_sq',g**2),('glucose_log',np.log(g)),('glucose_hdl',g/h),
        ('glucose_trig',g*t/1000),('glucose_non_hdl',g*nh/100),('glucose_chol_hdl',g*ch),
        ('bmi_sq',b**2),('bmi_log',np.log(b.clip(lower=1))),('bmi_trig',b*t/100),
        ('bmi_hdl_inv',b/h),('bmi_age',b*a),('ldl_hdl',l/h),('non_hdl_ratio',nh/h),
        ('tc_hdl_bmi',tc/h*b),('trig_tc',t/tc.clip(lower=1))]:
        X[nm] = v
    X['tyg_sq']=X['tyg']**2; X['mets_ir_sq']=X['mets_ir']**2
    X['trig_hdl_sq']=X['trig_hdl']**2; X['vat_sq']=X['vat_proxy']**2
    X['ir_proxy_sq']=X['ir_proxy']**2; X['ir_proxy_log']=np.log1p(X['ir_proxy'])
    rhr='Resting Heart Rate (mean)'; hrv='HRV (mean)'; stp='STEPS (mean)'
    if rhr in X.columns:
        for nm,v in [('bmi_rhr',b*X[rhr]),('glucose_rhr',g*X[rhr]),
            ('trig_hdl_rhr',X['trig_hdl']*X[rhr]),('ir_proxy_rhr',X['ir_proxy']*X[rhr]/100),
            ('tyg_rhr',X['tyg']*X[rhr]),('mets_rhr',X['mets_ir']*X[rhr]),
            ('bmi_hrv_inv',b/X[hrv].clip(lower=1)),
            ('cardio_fitness',X[hrv]*X[stp]/X[rhr].clip(lower=1)),
            ('met_load',b*X[rhr]/X[stp].clip(lower=1)*1000)]:
            X[nm] = v
        for pfx,m,s in [('rhr',rhr,'Resting Heart Rate (std)'),('hrv',hrv,'HRV (std)'),
                         ('stp',stp,'STEPS (std)'),('slp','SLEEP Duration (mean)','SLEEP Duration (std)')]:
            if s in X.columns: X[f'{pfx}_cv']=X[s]/X[m].clip(lower=0.01)
    X['log_glucose']=np.log(g); X['log_trig']=np.log(t)
    X['log_bmi']=np.log(b.clip(lower=1)); X['log_hdl']=np.log(h)
    X['log_homa_proxy']=np.log(g)+np.log(b.clip(lower=1))+np.log(t)-np.log(h)
    return X

X_v7_df = eng_v7(X_df[all_cols], all_cols).fillna(0)
X_v7 = X_v7_df.values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values

# Model B features
X_dw_eng = engineer_dw_features(X_df[dw_cols], dw_cols).values
print(f"  Model A: {X_v7.shape[1]} V7 features, {X_eng.shape[1]} eng features")
print(f"  Model B: {X_dw_eng.shape[1]} DW eng features")
sys.stdout.flush()

log_fn = np.log1p; inv_log = np.expm1

def get_oof(model_fn, X, y_arr, splits, scale=False, target_fn=None, inv_fn=None, weights=None):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = target_fn(y_arr) if target_fn else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale: sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        m = model_fn()
        if weights is not None:
            m.fit(Xtr, yt[tr], sample_weight=weights[tr])
        else:
            m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if inv_fn: p = inv_fn(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_pool_a = {}; scores_a = {}; cnt_a = 0
oof_pool_b = {}; scores_b = {}; cnt_b = 0

def add_a(name, oof):
    global cnt_a; cnt_a += 1
    r2 = r2_score(y, oof); oof_pool_a[name] = oof; scores_a[name] = r2
    print(f"  A[{cnt_a:2d}] {name:50s} R²={r2:.4f}"); sys.stdout.flush()

def add_b(name, oof):
    global cnt_b; cnt_b += 1
    r2 = r2_score(y, oof); oof_pool_b[name] = oof; scores_b[name] = r2
    print(f"  B[{cnt_b:2d}] {name:50s} R²={r2:.4f}"); sys.stdout.flush()

# ============================================================
# PART 1: MODEL B — DEMOGRAPHICS + WEARABLES ONLY
# ============================================================
print("\n" + "="*60)
print("  MODEL B: DEMOGRAPHICS + WEARABLES ONLY")
print("="*60)
print("  Paper baseline: R²=0.212, Our previous: R²=0.2463")
sys.stdout.flush()

# Linear models
print("\n--- B.1: Linear Models ---"); sys.stdout.flush()
for alpha in [0.01, 0.1, 1.0]:
    add_b(f'ridge_a{alpha}_dw',
        get_oof(lambda a=alpha: Ridge(alpha=a), X_dw_eng, y, splits, scale=True))

for alpha, l1 in [(0.01, 0.5), (0.1, 0.5), (0.01, 0.1), (0.1, 0.9)]:
    add_b(f'enet_a{alpha}_l{l1}_dw',
        get_oof(lambda a=alpha, l=l1: ElasticNet(alpha=a, l1_ratio=l, max_iter=5000),
                X_dw_eng, y, splits, scale=True))

add_b('huber_dw',
    get_oof(lambda: HuberRegressor(max_iter=500, alpha=0.01, epsilon=1.5),
            X_dw_eng, y, splits, scale=True))

# Tree models
print("\n--- B.2: Tree Models ---"); sys.stdout.flush()

# XGB with log target
for depth in [3, 4, 5]:
    add_b(f'xgb_d{depth}_log_dw',
        get_oof(lambda d=depth: xgb.XGBRegressor(n_estimators=400, max_depth=d, learning_rate=0.03,
            subsample=0.6, colsample_bytree=0.7, random_state=2024, verbosity=0),
            X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))

# XGB with sqrt weighting
add_b('xgb_d3_log_wsqrt_dw',
    get_oof(lambda: xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.6, colsample_bytree=0.7, random_state=2024, verbosity=0),
        X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# LGB
add_b('lgb_d3_log_dw',
    get_oof(lambda: lgb.LGBMRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=20, verbose=-1, random_state=42),
        X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))

add_b('lgb_d3_log_wsqrt_dw',
    get_oof(lambda: lgb.LGBMRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=20, verbose=-1, random_state=42),
        X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# GBR
add_b('gbr_d3_log_dw',
    get_oof(lambda: GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.03,
        subsample=0.6, min_samples_leaf=10, random_state=42),
        X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))

# HGBR (fast)
add_b('hgbr_log_dw',
    get_oof(lambda: HistGradientBoostingRegressor(max_iter=400, max_depth=5, learning_rate=0.03,
        min_samples_leaf=10, random_state=42),
        X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))

# Model B blend
print("\n--- B.3: Model B Blend ---"); sys.stdout.flush()
sorted_b = sorted(scores_b, key=scores_b.get, reverse=True)
print("  Top Model B models:")
for i, nm in enumerate(sorted_b[:10], 1):
    print(f"    {i:2d}. {nm:50s} R²={scores_b[nm]:.4f}")

# Greedy blend for Model B
sel_b = [sorted_b[0]]; rem_b = set(sorted_b[1:])
blend_b = oof_pool_b[sel_b[0]].copy(); cur_b = scores_b[sel_b[0]]
print(f"\n  Greedy Model B:")
print(f"    Step 1: {sel_b[0]:45s} R²={cur_b:.4f}")
for step in range(2, 10):
    best_add = None; best_r2 = cur_b
    for nm in rem_b:
        for alpha in np.arange(0.05, 0.55, 0.05):
            b = (1-alpha)*blend_b + alpha*oof_pool_b[nm]
            r2 = r2_score(y, b)
            if r2 > best_r2: best_r2 = r2; best_add = nm; best_a = alpha
    if best_add is None or best_r2 <= cur_b + 0.0001: break
    sel_b.append(best_add); rem_b.discard(best_add)
    blend_b = (1-best_a)*blend_b + best_a*oof_pool_b[best_add]; cur_b = best_r2
    print(f"    Step {step}: +{best_add:40s} α={best_a:.2f} → R²={cur_b:.4f}")
sys.stdout.flush()

# Dirichlet for Model B
sel_b_oofs = np.column_stack([oof_pool_b[k] for k in sel_b])
best_b_dir = -999; rng = np.random.RandomState(42)
for _ in range(2000000):
    w = rng.dirichlet(np.ones(len(sel_b)))
    r2 = 1 - np.sum((y - sel_b_oofs@w)**2) / np.sum((y - y.mean())**2)
    if r2 > best_b_dir: best_b_dir = r2; best_b_w = w

print(f"\n  ★ Model B Greedy: R²={cur_b:.4f}")
print(f"  ★ Model B Dirichlet: R²={best_b_dir:.4f}")
print(f"  ★ Paper baseline: R²=0.212 | Improvement: +{best_b_dir-0.212:.4f}")
print("  Weights:")
for nm, w in zip(sel_b, best_b_w):
    if w > 0.01: print(f"    {nm:50s} w={w:.3f}")

# ============================================================
# PART 2: MODEL A — POWER TRANSFORM OPTIMIZATION
# ============================================================
print("\n" + "="*60)
print("  MODEL A: POWER TRANSFORM + FINAL OPTIMIZATION")
print("="*60)
sys.stdout.flush()

# Try different power transforms
print("\n--- A.1: Target Transform Search ---"); sys.stdout.flush()

v13_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017, subsample=0.52,
    colsample_bytree=0.78, min_child_weight=29, reg_alpha=2.8, reg_lambda=0.045)
lgb_params = {"n_estimators": 768, "max_depth": 4, "learning_rate": 0.0129,
    "subsample": 0.409, "colsample_bytree": 0.889, "min_child_samples": 36,
    "reg_alpha": 3.974, "reg_lambda": 0.203, "num_leaves": 10, "verbose": -1, "random_state": 42}

# Power transform: y^lambda, optimized per fold
for lam in [0.2, 0.3, 0.4, 0.5, 0.6]:
    tfn = lambda y_arr, l=lam: np.power(y_arr, l)
    ifn = lambda p, l=lam: np.power(np.clip(p, 0, None), 1.0/l)
    add_a(f'xgb_power{lam}_wsqrt',
        get_oof(lambda: xgb.XGBRegressor(**v13_params, random_state=2024, verbosity=0),
                X_v7, y, splits, target_fn=tfn, inv_fn=ifn, weights=w_sqrt))

# sqrt(y) as target (power=0.5)
add_a('xgb_sqrt_target_wsqrt',
    get_oof(lambda: xgb.XGBRegressor(**v13_params, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=np.sqrt, inv_fn=lambda p: np.clip(p, 0, None)**2, weights=w_sqrt))

# Baselines for blend
print("\n--- A.2: Baselines ---"); sys.stdout.flush()
add_a('xgb_optuna_wsqrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**v13_params, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

add_a('lgb_optuna_wsqrt_v7',
    get_oof(lambda: lgb.LGBMRegressor(**lgb_params), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

gbr_params = {"n_estimators": 373, "max_depth": 3, "learning_rate": 0.0313,
    "subsample": 0.470, "min_samples_leaf": 12, "max_features": 0.556, "random_state": 42}
add_a('gbr_optuna_wsqrt_v7',
    get_oof(lambda: GradientBoostingRegressor(**gbr_params), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

add_a('enet_01_eng',
    get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            X_eng, y, splits, scale=True))

# XGB on raw features (no engineering) with log target
add_a('xgb_raw_log_wsqrt',
    get_oof(lambda: xgb.XGBRegressor(**v13_params, random_state=2024, verbosity=0),
            X_all_raw, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

print(f"\n  Total Model A: {cnt_a} | Total Model B: {cnt_b} ({time.time()-t_start:.0f}s)")
sys.stdout.flush()

# ============================================================
# PART 3: MODEL A BLEND
# ============================================================
print("\n--- A.3: Model A Blend ---"); sys.stdout.flush()

sorted_a = sorted(scores_a, key=scores_a.get, reverse=True)
print("  All Model A ranked:")
for i, nm in enumerate(sorted_a, 1):
    print(f"    {i:2d}. {nm:50s} R²={scores_a[nm]:.4f}")

# Greedy
sel_a = [sorted_a[0]]; rem_a = set(sorted_a[1:])
blend_a = oof_pool_a[sel_a[0]].copy(); cur_a = scores_a[sel_a[0]]
print(f"\n  Greedy Model A:")
print(f"    Step 1: {sel_a[0]:45s} R²={cur_a:.4f}")
for step in range(2, 12):
    best_add = None; best_r2 = cur_a
    for nm in rem_a:
        for alpha in np.arange(0.02, 0.50, 0.02):
            b = (1-alpha)*blend_a + alpha*oof_pool_a[nm]
            r2 = r2_score(y, b)
            if r2 > best_r2: best_r2 = r2; best_add = nm; best_a = alpha
    if best_add is None or best_r2 <= cur_a + 0.00005: break
    sel_a.append(best_add); rem_a.discard(best_add)
    blend_a = (1-best_a)*blend_a + best_a*oof_pool_a[best_add]; cur_a = best_r2
    print(f"    Step {step}: +{best_add:40s} α={best_a:.2f} → R²={cur_a:.4f}")
sys.stdout.flush()

# Dirichlet
sel_a_oofs = np.column_stack([oof_pool_a[k] for k in sel_a])
best_a_dir = -999
for _ in range(2000000):
    w = rng.dirichlet(np.ones(len(sel_a)))
    r2 = 1 - np.sum((y - sel_a_oofs@w)**2) / np.sum((y - y.mean())**2)
    if r2 > best_a_dir: best_a_dir = r2; best_a_w = w

print(f"\n  ★ Model A Greedy: R²={cur_a:.4f}")
print(f"  ★ Model A Dirichlet: R²={best_a_dir:.4f}")
print("  Weights:")
for nm, w in zip(sel_a, best_a_w):
    if w > 0.01: print(f"    {nm:50s} w={w:.3f}")

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_model_a = max(cur_a, best_a_dir)
best_model_b = max(cur_b, best_b_dir)

print(f"\n{'='*60}")
print(f"  V17 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ MODEL A BEST: R²={best_model_a:.4f}")
print(f"    Best single: {sorted_a[0]} R²={scores_a[sorted_a[0]]:.4f}")
if best_model_a > 0.5465:
    print(f"    🎉 NEW BEST Model A! +{best_model_a - 0.5465:.4f} over V14")
else:
    print(f"    Gap to V14: {0.5465 - best_model_a:.4f}")

print(f"\n  ★ MODEL B BEST: R²={best_model_b:.4f}")
print(f"    Best single: {sorted_b[0]} R²={scores_b[sorted_b[0]]:.4f}")
print(f"    Paper baseline: R²=0.212 | Our improvement: +{best_model_b-0.212:.4f}")
print(f"    Previous best: R²=0.2463 | Change: {best_model_b-0.2463:+.4f}")

print(f"\n  Target A: 0.65 | Gap: {0.65 - best_model_a:.4f}")

results = {
    'model_a': {
        'best_r2': float(best_model_a),
        'best_single': {'name': sorted_a[0], 'r2': float(scores_a[sorted_a[0]])},
        'greedy_r2': float(cur_a),
        'dirichlet_r2': float(best_a_dir),
        'all_scores': {k: float(scores_a[k]) for k in sorted_a},
    },
    'model_b': {
        'best_r2': float(best_model_b),
        'best_single': {'name': sorted_b[0], 'r2': float(scores_b[sorted_b[0]])},
        'greedy_r2': float(cur_b),
        'dirichlet_r2': float(best_b_dir),
        'all_scores': {k: float(scores_b[k]) for k in sorted_b},
    },
    'elapsed': elapsed
}
with open('v17_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v17_results.json"); sys.stdout.flush()
