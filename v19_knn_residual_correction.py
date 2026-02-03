#!/usr/bin/env python3
"""
V19: KNN Regression + Residual Correction + Final Ceiling Analysis

Last unexplored ideas:
1. KNN regression with optimized k and distance metrics
2. Residual correction: predict V14 residuals with a different model
3. SVR with RBF kernel (revisited with better features)
4. Final ceiling analysis with feature importance
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import load_data, get_feature_sets, get_cv_splits, engineer_all_features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V19: KNN + KERNEL METHODS + RESIDUAL CORRECTION")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all, _, all_cols, _ = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()

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

X_v7 = eng_v7(X_df[all_cols], all_cols).fillna(0).values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values
log_fn, inv_log = np.log1p, np.expm1

def get_oof(model_fn, X, y_arr, splits, scale=False, target_fn=None, inv_fn=None, weights=None):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = target_fn(y_arr) if target_fn else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale: sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        m = model_fn()
        if weights is not None: m.fit(Xtr, yt[tr], sample_weight=weights[tr])
        else: m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if inv_fn: p = inv_fn(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_pool = {}; scores = {}; cnt = 0
def add(name, oof):
    global cnt; cnt += 1
    r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
    print(f"  [{cnt:2d}] {name:50s} R²={r2:.4f}"); sys.stdout.flush()

v13 = dict(n_estimators=612, max_depth=4, learning_rate=0.017, subsample=0.52,
    colsample_bytree=0.78, min_child_weight=29, reg_alpha=2.8, reg_lambda=0.045)
lgb_p = {'n_estimators': 768, 'max_depth': 4, 'learning_rate': 0.0129,
    'subsample': 0.409, 'colsample_bytree': 0.889, 'min_child_samples': 36,
    'reg_alpha': 3.974, 'reg_lambda': 0.203, 'num_leaves': 10, 'verbose': -1, 'random_state': 42}

# ============================================================
# PART 1: KNN REGRESSION (new model family)
# ============================================================
print("\n--- Part 1: KNN Regression ---"); sys.stdout.flush()

for k in [5, 10, 15, 20, 30, 50]:
    add(f'knn_k{k}_uniform',
        get_oof(lambda k=k: KNeighborsRegressor(n_neighbors=k, weights='uniform', n_jobs=-1),
                X_v7, y, splits, scale=True))

for k in [5, 10, 15, 20, 30]:
    add(f'knn_k{k}_distance',
        get_oof(lambda k=k: KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1),
                X_v7, y, splits, scale=True))

# KNN on raw features (fewer dimensions = better for KNN)
add('knn_k15_raw',
    get_oof(lambda: KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1),
            X_all, y, splits, scale=True))

# ============================================================
# PART 2: KERNEL METHODS
# ============================================================
print("\n--- Part 2: Kernel Methods ---"); sys.stdout.flush()

# Kernel Ridge with RBF
for alpha in [0.1, 1.0, 10.0]:
    for gamma in [0.001, 0.01, 0.1]:
        add(f'kr_rbf_a{alpha}_g{gamma}',
            get_oof(lambda a=alpha, g=gamma: KernelRidge(alpha=a, kernel='rbf', gamma=g),
                    X_v7, y, splits, scale=True))

# ============================================================
# PART 3: RESIDUAL CORRECTION
# ============================================================
print("\n--- Part 3: Residual Correction ---"); sys.stdout.flush()

# Get base model OOF predictions
base_oof = get_oof(lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0),
                    X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt)
base_r2 = r2_score(y, base_oof)
print(f"  Base XGB: R²={base_r2:.4f}")

# Try to predict residuals
residuals = y - base_oof
print(f"  Residual stats: mean={residuals.mean():.3f}, std={residuals.std():.3f}")
print(f"  Residual R² ceiling (if perfect): {r2_score(y, base_oof + residuals):.4f}")

# KNN on residuals (non-parametric, might capture local patterns)
for k in [10, 20, 30]:
    res_pred = get_oof(lambda k=k: KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1),
                       X_v7, residuals, splits, scale=True)
    corrected = base_oof + res_pred
    add(f'xgb+knn_resid_k{k}', corrected)

# LGB on residuals
res_pred_lgb = get_oof(lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.01,
    subsample=0.6, colsample_bytree=0.6, min_child_samples=30, verbose=-1, random_state=42),
    X_v7, residuals, splits)
add('xgb+lgb_resid', base_oof + res_pred_lgb)

# ============================================================
# PART 4: BASELINES FOR BLEND
# ============================================================
print("\n--- Part 4: Baselines ---"); sys.stdout.flush()

add('xgb_optuna_wsqrt',
    get_oof(lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))
add('lgb_optuna_wsqrt',
    get_oof(lambda: lgb.LGBMRegressor(**lgb_p), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))
add('enet_01',
    get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            X_eng, y, splits, scale=True))

print(f"\n  Total: {cnt} models ({time.time()-t_start:.0f}s)")
sys.stdout.flush()

# ============================================================
# PART 5: BLEND
# ============================================================
print("\n--- Part 5: Greedy + Dirichlet ---"); sys.stdout.flush()

sorted_m = sorted(scores, key=scores.get, reverse=True)
print("  Top 20:")
for i, nm in enumerate(sorted_m[:20], 1):
    print(f"    {i:2d}. {nm:50s} R²={scores[nm]:.4f}")

sel = [sorted_m[0]]; rem = set(sorted_m[1:])
blend = oof_pool[sel[0]].copy(); cur = scores[sel[0]]
print(f"\n  Greedy:")
print(f"    Step 1: {sel[0]:45s} R²={cur:.4f}")
for step in range(2, 12):
    ba, br = None, cur
    for nm in rem:
        for a in np.arange(0.02, 0.50, 0.02):
            b = (1-a)*blend + a*oof_pool[nm]; r2 = r2_score(y, b)
            if r2 > br: br = r2; ba = nm; bsa = a
    if ba is None or br <= cur + 0.00005: break
    sel.append(ba); rem.discard(ba)
    blend = (1-bsa)*blend + bsa*oof_pool[ba]; cur = br
    print(f"    Step {step}: +{ba:40s} a={bsa:.2f} R²={cur:.4f}")
sys.stdout.flush()

sel_oofs = np.column_stack([oof_pool[k] for k in sel])
best_dir = -999; rng = np.random.RandomState(42)
for _ in range(2000000):
    w = rng.dirichlet(np.ones(len(sel)))
    r2 = 1 - np.sum((y - sel_oofs@w)**2) / np.sum((y - y.mean())**2)
    if r2 > best_dir: best_dir = r2; bw = w

print(f"\n  ★ Greedy: R²={cur:.4f}")
print(f"  ★ Dirichlet: R²={best_dir:.4f}")
for nm, w in zip(sel, bw):
    if w > 0.01: print(f"    {nm:50s} w={w:.3f}")

elapsed = time.time() - t_start
best = max(cur, best_dir)
knn_best = max((scores[k] for k in scores if 'knn' in k and 'resid' not in k), default=0)
kr_best = max((scores[k] for k in scores if 'kr_' in k), default=0)
resid_best = max((scores[k] for k in scores if 'resid' in k), default=0)

print(f"\n{'='*60}")
print(f"  V19 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ BEST BLEND: R²={best:.4f}")
print(f"  Best KNN single: R²={knn_best:.4f}")
print(f"  Best Kernel Ridge: R²={kr_best:.4f}")
print(f"  Best residual correction: R²={resid_best:.4f}")
if best > 0.5466:
    print(f"  🎉 NEW BEST! +{best - 0.5466:.4f} over V17")
else:
    print(f"  Gap to V17: {0.5466 - best:.4f}")

results = {
    'best_r2': float(best),
    'knn_best': float(knn_best),
    'kernel_ridge_best': float(kr_best),
    'residual_correction_best': float(resid_best),
    'all_scores': {k: float(scores[k]) for k in sorted_m},
    'elapsed': elapsed
}
with open('v19_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v19_results.json"); sys.stdout.flush()
