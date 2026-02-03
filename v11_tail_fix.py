#!/usr/bin/env python3
"""
V11: Fix the tail problem.

V10 key insight: extreme HOMA_IR [8+] has bias +4.94.
39 samples (3.6%) with HOMA>8 dominate MSE.
The model systematically underpredicts high values.

Approaches:
1. Winsorize target: cap at percentiles to reduce outlier MSE
2. Log target already helps (compresses tail) — try stronger transforms
3. Huber loss in XGBoost (robust to outliers)
4. Sample weighting: upweight tails so model tries harder there
5. Separate models for low vs high HOMA ranges
6. Post-hoc calibration: stretch predictions for high-HOMA samples
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V11: TAIL FIX")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, _, all_cols, _ = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# Compact V7 features
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
    return X.fillna(0)

X_v7 = eng_v7(X_df[all_cols], all_cols).values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values

xgb_d3 = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
               colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01)

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

log_fn=np.log1p; inv_log=np.expm1

oof_pool = {}; scores = {}; cnt = 0
def add(name, oof):
    global cnt; cnt += 1
    r2=r2_score(y,oof); oof_pool[name]=oof; scores[name]=r2
    print(f"  [{cnt:2d}] {name:55s} R²={r2:.4f}"); sys.stdout.flush()

# ============================================================
# 1. BASELINE (reproduce V10 best)
# ============================================================
print("\n--- 1. Baseline ---"); sys.stdout.flush()
add('xgb_d3_log_s2024_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
add('enet_01_eng',
    get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            X_eng, y, splits, scale=True))

# ============================================================
# 2. WINSORIZED TARGETS
# ============================================================
print("\n--- 2. Winsorized Targets ---"); sys.stdout.flush()
for cap_pct in [95, 97, 99]:
    cap_val = np.percentile(y, cap_pct)
    y_win = np.clip(y, None, cap_val)
    oof = get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
                  X_v7, y_win, splits, target_fn=log_fn, inv_fn=inv_log)
    # Evaluate against ORIGINAL y
    add(f'xgb_d3_log_winsor{cap_pct}_v7', oof)

# ============================================================
# 3. SAMPLE WEIGHTING (upweight tails)
# ============================================================
print("\n--- 3. Sample Weighting ---"); sys.stdout.flush()

# Weight proportional to y (upweight high HOMA)
w_prop = y / y.mean()
add('xgb_d3_log_wprop_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_prop))

# Weight proportional to sqrt(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
add('xgb_d3_log_wsqrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# Weight proportional to y² (aggressive upweight)
w_sq = y**2 / (y**2).mean()
add('xgb_d3_log_wsq_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sq))

# ============================================================
# 4. HUBER LOSS
# ============================================================
print("\n--- 4. Huber Loss ---"); sys.stdout.flush()
for delta in [1.0, 2.0, 3.0]:
    add(f'xgb_d3_huber{delta}_v7',
        get_oof(lambda d=delta: xgb.XGBRegressor(**{**xgb_d3, 'objective': 'reg:pseudohubererror',
                'huber_slope': d}, random_state=2024, verbosity=0),
                X_v7, y, splits))

# ============================================================
# 5. STRONGER TARGET TRANSFORMS
# ============================================================
print("\n--- 5. Stronger Transforms ---"); sys.stdout.flush()

# log(log(y+1)+1) — double log (even more compression)
dlog_fn = lambda y: np.log1p(np.log1p(y))
inv_dlog = lambda p: np.expm1(np.expm1(p))
add('xgb_d3_dlog_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=dlog_fn, inv_fn=inv_dlog))

# y^0.25 (quarter root — between sqrt and log)
qrt_fn = lambda y: y.clip(min=0) ** 0.25
inv_qrt = lambda p: p.clip(min=0) ** 4
add('xgb_d3_qrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=qrt_fn, inv_fn=inv_qrt))

# Quantile transform to uniform then predict
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=42)
y_qt = qt.fit_transform(y.reshape(-1,1)).ravel()
def qt_inv(p):
    return qt.inverse_transform(p.reshape(-1,1)).ravel()

oof_qt_sum, oof_qt_cnt = np.zeros(n), np.zeros(n)
for tr, te in splits:
    qt_fold = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=42)
    yt_fold = qt_fold.fit_transform(y[tr].reshape(-1,1)).ravel()
    m = xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0)
    m.fit(X_v7[tr], yt_fold)
    p_qt = m.predict(X_v7[te])
    p_orig = qt_fold.inverse_transform(p_qt.reshape(-1,1)).ravel()
    oof_qt_sum[te] += p_orig; oof_qt_cnt[te] += 1
oof_qt = oof_qt_sum / np.clip(oof_qt_cnt, 1, None)
add('xgb_d3_quantile_v7', oof_qt)

# ============================================================
# 6. ADD MORE DIVERSITY FOR BLEND
# ============================================================
print("\n--- 6. More Diversity ---"); sys.stdout.flush()
add('xgb_d3_log_s42_eng',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=42, verbosity=0),
            X_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))
add('lgb_d3_log_v7',
    get_oof(lambda: lgb.LGBMRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=20, verbose=-1, random_state=42),
     X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
add('hgbr_d4_log_v7',
    get_oof(lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=4,
        learning_rate=0.03, min_samples_leaf=10, random_state=42),
     X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))
add('ridge_500_v7',
    get_oof(lambda: Ridge(alpha=500), X_v7, y, splits, scale=True))

print(f"\n  Total: {cnt} models ({time.time()-t_start:.0f}s)")

# ============================================================
# 7. GREEDY BLEND + DIRICHLET
# ============================================================
print("\n--- 7. Greedy Blend + Dirichlet ---"); sys.stdout.flush()

sorted_m = sorted(scores, key=scores.get, reverse=True)
print(f"  Top 10:")
for i, nm in enumerate(sorted_m[:10], 1):
    print(f"    {i:2d}. {nm:55s} R²={scores[nm]:.4f}")

# Greedy forward selection
selected = [sorted_m[0]]; remaining = set(sorted_m[1:])
blend = oof_pool[selected[0]].copy()
cur_r2 = scores[selected[0]]
print(f"\n  Greedy:")
print(f"    Step 1: {selected[0]:50s} R²={cur_r2:.4f}")
for step in range(2, 12):
    best_add=None; best_r2=cur_r2
    for nm in remaining:
        for alpha in np.arange(0.02, 0.50, 0.02):
            b = (1-alpha)*blend + alpha*oof_pool[nm]
            r2 = r2_score(y, b)
            if r2 > best_r2: best_r2=r2; best_add=nm; best_a=alpha
    if best_add is None or best_r2 <= cur_r2 + 0.0001: break
    selected.append(best_add); remaining.discard(best_add)
    blend = (1-best_a)*blend + best_a*oof_pool[best_add]
    cur_r2 = best_r2
    print(f"    Step {step}: +{best_add:45s} α={best_a:.2f} → R²={cur_r2:.4f}")
    sys.stdout.flush()

# Dirichlet on greedy set
sel_oofs = np.column_stack([oof_pool[k] for k in selected])
best_dir = -999; rng = np.random.RandomState(42)
for _ in range(2000000):
    w = rng.dirichlet(np.ones(len(selected)))
    r2 = 1 - np.sum((y - sel_oofs @ w)**2) / np.sum((y-y.mean())**2)
    if r2 > best_dir: best_dir = r2; best_w = w

print(f"\n  ★ Greedy: R²={cur_r2:.4f} ({len(selected)} models)")
print(f"  ★ Dirichlet on greedy: R²={best_dir:.4f}")
print(f"  Weights:")
for nm, w in zip(selected, best_w):
    if w > 0.01: print(f"    {nm:50s} w={w:.3f}")

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_overall = max(cur_r2, best_dir)
print(f"\n{'='*60}")
print(f"  V11 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ BEST: R²={best_overall:.4f}")
print(f"  Best single: {sorted_m[0]} R²={scores[sorted_m[0]]:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_overall:.4f}")

results = {
    'best_r2': float(best_overall),
    'greedy_r2': float(cur_r2),
    'greedy_models': selected,
    'all_scores': {k: float(scores[k]) for k in sorted_m},
    'elapsed': elapsed
}
with open('v11_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v11_results.json")
sys.stdout.flush()
