#!/usr/bin/env python3
"""
V20: Quantile regression + Non-linear input transforms + Model B Optuna.

After 6 versions stuck at 0.546, we need fundamentally different approaches:

1. QuantileTransformer on INPUT features (not just target):
   - Maps each feature to uniform/normal distribution
   - Makes trees see different decision boundaries
   - Different from target transform — this changes feature space

2. Quantile regression: Train models at different quantiles (25th, 50th, 75th)
   then combine — captures asymmetric error structure

3. Model B Optuna: Seriously optimize wearables-only (currently 0.2449)

4. Try combining Model A + Model B predictions for better Model A
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, engineer_dw_features)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V20: QUANTILE + NON-LINEAR INPUT + MODEL B OPTUNA")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
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
    return X.fillna(0)

X_v7 = eng_v7(X_df[all_cols], all_cols).values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values
X_dw_eng = engineer_dw_features(X_df[dw_cols], dw_cols).values

log_fn=np.log1p; inv_log=np.expm1

# V13/V14 best params
xgb_opt = dict(n_estimators=612, max_depth=4, learning_rate=0.017, subsample=0.52,
    colsample_bytree=0.78, min_child_weight=29, reg_alpha=2.8, reg_lambda=0.045)
lgb_opt = dict(n_estimators=768, max_depth=4, learning_rate=0.013, subsample=0.41,
    colsample_bytree=0.89, min_child_samples=36, reg_alpha=3.97, reg_lambda=0.20,
    num_leaves=10, verbose=-1)

def get_oof(model_fn, X, y_arr, splits, scale=False, target_fn=None, inv_fn=None, weights=None, qt_input=False):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = target_fn(y_arr) if target_fn else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if qt_input:
            qt = QuantileTransformer(n_quantiles=min(200, len(tr)), output_distribution='normal', random_state=42)
            Xtr = qt.fit_transform(Xtr); Xte = qt.transform(Xte)
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn()
        if weights is not None: m.fit(Xtr, yt[tr], sample_weight=weights[tr])
        else: m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if inv_fn: p = inv_fn(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_pool={}; scores={}; cnt=0
def add(name, oof):
    global cnt; cnt+=1
    r2=r2_score(y,oof); oof_pool[name]=oof; scores[name]=r2
    print(f"  [{cnt:2d}] {name:55s} R²={r2:.4f}"); sys.stdout.flush()

# ============================================================
# PART 1: QUANTILE TRANSFORMER ON INPUT FEATURES
# ============================================================
print("\n--- Part 1: QuantileTransformer on Inputs ---"); sys.stdout.flush()

# XGB with QT input
add('xgb_qt_log_wsqrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_opt, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt, qt_input=True))

# LGB with QT input
add('lgb_qt_log_wsqrt_v7',
    get_oof(lambda: lgb.LGBMRegressor(**{**lgb_opt, 'random_state': 42}),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt, qt_input=True))

# XGB with QT on raw features (maybe V7 eng already saturated)
add('xgb_qt_log_wsqrt_raw',
    get_oof(lambda: xgb.XGBRegressor(**xgb_opt, random_state=2024, verbosity=0),
            X_all_raw, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt, qt_input=True))

# Without QT baselines for comparison
add('xgb_log_wsqrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_opt, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))
add('lgb_log_wsqrt_v7',
    get_oof(lambda: lgb.LGBMRegressor(**{**lgb_opt, 'random_state': 42}),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# ============================================================
# PART 2: QUANTILE REGRESSION (predict different quantiles)
# ============================================================
print("\n--- Part 2: Quantile Regression ---"); sys.stdout.flush()

for alpha in [0.25, 0.5, 0.75]:
    add(f'xgb_quantile{alpha}_wsqrt_v7',
        get_oof(lambda a=alpha: xgb.XGBRegressor(
            **{**xgb_opt, 'objective': 'reg:quantileerror', 'quantile_alpha': a},
            random_state=2024, verbosity=0),
            X_v7, y, splits, weights=w_sqrt))

# Average of quantile predictions
q25 = oof_pool.get('xgb_quantile0.25_wsqrt_v7')
q50 = oof_pool.get('xgb_quantile0.5_wsqrt_v7')
q75 = oof_pool.get('xgb_quantile0.75_wsqrt_v7')
if q25 is not None and q50 is not None and q75 is not None:
    avg_q = (q25 + q50 + q75) / 3
    add('quantile_avg_025_050_075', avg_q)
    # Weighted average favoring median
    wavg_q = 0.25*q25 + 0.5*q50 + 0.25*q75
    add('quantile_wavg', wavg_q)

# ============================================================
# PART 3: MODEL B OPTUNA (XGB + LGB)
# ============================================================
print("\n--- Part 3: Model B Optuna ---"); sys.stdout.flush()
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

b_pool={}; b_scores={}; b_cnt=0
def add_b(name, oof):
    global b_cnt; b_cnt+=1
    r2=r2_score(y,oof); b_pool[name]=oof; b_scores[name]=r2
    print(f"  B[{b_cnt:2d}] {name:50s} R²={r2:.4f}"); sys.stdout.flush()

def xgb_b_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 5.0, log=True),
        'random_state': 2024, 'verbosity': 0
    }
    oof = get_oof(lambda: xgb.XGBRegressor(**params), X_dw_eng, y, splits,
                  target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt)
    r2 = r2_score(y, oof)
    print(f"    B-XGB {trial.number+1:2d}: R²={r2:.4f} (d={params['max_depth']}, lr={params['learning_rate']:.3f})")
    sys.stdout.flush()
    return r2

study_b = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_b.optimize(xgb_b_objective, n_trials=30, timeout=180)
print(f"\n  ★ Model B XGB Optuna: R²={study_b.best_value:.4f}")
sys.stdout.flush()

xgb_b_params = {**study_b.best_params, 'random_state': 2024, 'verbosity': 0}
add_b('xgb_optuna_wsqrt_dw',
      get_oof(lambda: xgb.XGBRegressor(**xgb_b_params), X_dw_eng, y, splits,
              target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# Model B baselines
add_b('enet_09_dw', get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=5000),
                              X_dw_eng, y, splits, scale=True))
add_b('lgb_d3_log_dw',
      get_oof(lambda: lgb.LGBMRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
          subsample=0.7, colsample_bytree=0.7, min_child_samples=20, verbose=-1, random_state=42),
       X_dw_eng, y, splits, target_fn=log_fn, inv_fn=inv_log))
add_b('ridge_1000_dw', get_oof(lambda: Ridge(alpha=1000), X_dw_eng, y, splits, scale=True))

# Model B blend
b_sorted = sorted(b_scores, key=b_scores.get, reverse=True)
print(f"\n  Model B top:")
for i, nm in enumerate(b_sorted[:5], 1):
    print(f"    {i}. {nm:45s} R²={b_scores[nm]:.4f}")

if len(b_sorted) >= 2:
    sel_b=[b_sorted[0]]; rem_b=set(b_sorted[1:])
    blend_b=b_pool[sel_b[0]].copy(); cur_b=b_scores[sel_b[0]]
    for step in range(2, 6):
        best_add=None; best_r2=cur_b
        for nm in rem_b:
            for alpha in np.arange(0.05, 0.50, 0.05):
                b=(1-alpha)*blend_b+alpha*b_pool[nm]
                r2=r2_score(y,b)
                if r2>best_r2: best_r2=r2; best_add=nm; best_a=alpha
        if best_add is None or best_r2<=cur_b+0.0001: break
        sel_b.append(best_add); rem_b.discard(best_add)
        blend_b=(1-best_a)*blend_b+best_a*b_pool[best_add]; cur_b=best_r2
        print(f"    B Step {step}: +{best_add:35s} α={best_a:.2f} → R²={cur_b:.4f}")
    print(f"  ★ Model B blend: R²={cur_b:.4f}")

# ============================================================
# PART 4: MODEL A MEGA-BLEND
# ============================================================
print("\n--- Part 4: Model A Blend ---"); sys.stdout.flush()

# Add baselines
add('enet_01_eng', get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
                             X_eng, y, splits, scale=True))

sorted_m = sorted(scores, key=scores.get, reverse=True)
print(f"  Model A top 10:")
for i, nm in enumerate(sorted_m[:10], 1):
    print(f"    {i:2d}. {nm:55s} R²={scores[nm]:.4f}")

# Greedy
selected=[sorted_m[0]]; remaining=set(sorted_m[1:])
blend=oof_pool[selected[0]].copy(); cur_r2=scores[selected[0]]
print(f"\n  Greedy:")
print(f"    Step 1: {selected[0]:50s} R²={cur_r2:.4f}")
for step in range(2, 12):
    best_add=None; best_r2=cur_r2
    for nm in remaining:
        for alpha in np.arange(0.02, 0.50, 0.02):
            b=(1-alpha)*blend+alpha*oof_pool[nm]
            r2=r2_score(y,b)
            if r2>best_r2: best_r2=r2; best_add=nm; best_a=alpha
    if best_add is None or best_r2<=cur_r2+0.0001: break
    selected.append(best_add); remaining.discard(best_add)
    blend=(1-best_a)*blend+best_a*oof_pool[best_add]; cur_r2=best_r2
    print(f"    Step {step}: +{best_add:45s} α={best_a:.2f} → R²={cur_r2:.4f}")
    sys.stdout.flush()

# Dirichlet
sel_oofs=np.column_stack([oof_pool[k] for k in selected])
best_dir=-999; rng=np.random.RandomState(42)
for _ in range(2000000):
    w=rng.dirichlet(np.ones(len(selected)))
    r2=1-np.sum((y-sel_oofs@w)**2)/np.sum((y-y.mean())**2)
    if r2>best_dir: best_dir=r2; best_w=w
print(f"\n  ★ Model A Greedy: R²={cur_r2:.4f}")
print(f"  ★ Model A Dirichlet: R²={best_dir:.4f}")
print("  Weights:")
for nm,w in zip(selected, best_w):
    if w>0.01: print(f"    {nm:55s} w={w:.3f}")

# ============================================================
# SUMMARY
# ============================================================
elapsed=time.time()-t_start
best_a = max(cur_r2, best_dir)
best_b = cur_b if 'cur_b' in dir() else 0
print(f"\n{'='*60}")
print(f"  V20 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ MODEL A: R²={best_a:.4f}")
print(f"  ★ MODEL B: R²={best_b:.4f}")
print(f"  Target: 0.65 | Gap: {0.65-best_a:.4f}")

results = {
    'best_r2_a': float(best_a),
    'best_r2_b': float(best_b),
    'best_single_a': {'name': sorted_m[0], 'r2': float(scores[sorted_m[0]])},
    'all_scores_a': {k: float(scores[k]) for k in sorted_m},
    'all_scores_b': {k: float(b_scores[k]) for k in b_sorted} if b_sorted else {},
    'elapsed': elapsed
}
with open('v20_results.json','w') as f:
    json.dump(results,f,indent=2)
print(f"  Saved to v20_results.json"); sys.stdout.flush()
