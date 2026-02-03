#!/usr/bin/env python3
"""
V15: Optuna HGBR + Cross-model mega-blend.

We now have 3 strong families at ~0.54: XGB, LGB, GBR.
Add HGBR (4th family) and build the ultimate mega-blend.

Also: try combining models trained with DIFFERENT weight exponents
for maximum diversity (0.3, 0.5, 0.7 weighting).
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

t_start = time.time()
print("="*60)
print("  V15: OPTUNA HGBR + CROSS-MODEL MEGA-BLEND")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, _, all_cols, _ = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

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
log_fn=np.log1p; inv_log=np.expm1

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

oof_pool={}; scores={}; cnt=0
def add(name, oof):
    global cnt; cnt+=1
    r2=r2_score(y,oof); oof_pool[name]=oof; scores[name]=r2
    print(f"  [{cnt:2d}] {name:55s} R²={r2:.4f}"); sys.stdout.flush()

# ============================================================
# PART 1: OPTUNA HGBR WITH WEIGHTS
# ============================================================
print("\n--- Part 1: Optuna HGBR Weighted ---"); sys.stdout.flush()
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()

def hgbr_objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 8, 64),
        'l2_regularization': trial.suggest_float('l2_regularization', 0.01, 10.0, log=True),
        'random_state': 42
    }
    oof = get_oof(lambda: HistGradientBoostingRegressor(**params), X_v7, y, splits,
                  target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt)
    r2 = r2_score(y, oof)
    print(f"    HGBR Trial {trial.number+1:3d}: R²={r2:.4f} (d={params['max_depth']}, lr={params['learning_rate']:.4f}, "
          f"n={params['max_iter']}, leaves={params['max_leaf_nodes']})")
    sys.stdout.flush()
    return r2

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(hgbr_objective, n_trials=40, timeout=250)
hgbr_best = study.best_params
hgbr_best_r2 = study.best_value
print(f"\n  ★ HGBR Optuna best: R²={hgbr_best_r2:.4f}")
sys.stdout.flush()

add('hgbr_optuna_wsqrt_v7',
    get_oof(lambda: HistGradientBoostingRegressor(**{**hgbr_best, 'random_state': 42}),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# ============================================================
# PART 2: ALL BEST MODELS (different families + weight exponents)
# ============================================================
print("\n--- Part 2: All Best Models ---"); sys.stdout.flush()

# V13 Optuna XGB params
xgb_opt = dict(n_estimators=612, max_depth=4, learning_rate=0.017, subsample=0.52,
    colsample_bytree=0.78, min_child_weight=29, reg_alpha=2.8, reg_lambda=0.045)

# V14 Optuna LGB params  
lgb_opt = dict(n_estimators=768, max_depth=4, learning_rate=0.013, subsample=0.41,
    colsample_bytree=0.89, min_child_samples=36, reg_alpha=3.97, reg_lambda=0.20,
    num_leaves=10, verbose=-1)

# V14 Optuna GBR params
gbr_opt = dict(n_estimators=373, max_depth=3, learning_rate=0.031, subsample=0.47,
    min_samples_leaf=12, max_features=0.56)

# Build with different weight exponents for diversity
for exp in [0.3, 0.5, 0.7]:
    w = y**exp / (y**exp).mean()
    
    # XGB
    add(f'xgb_opt_w{exp}_v7',
        get_oof(lambda: xgb.XGBRegressor(**xgb_opt, random_state=2024, verbosity=0),
                X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w))
    # LGB
    add(f'lgb_opt_w{exp}_v7',
        get_oof(lambda: lgb.LGBMRegressor(**{**lgb_opt, 'random_state': 42}),
                X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w))

# XGB unweighted (diversity)
add('xgb_opt_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_opt, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))

# XGB d3 (original good params) weighted
xgb_d3 = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
               colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01)
add('xgb_d3_log_wsqrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# GBR weighted
add('gbr_opt_wsqrt_v7',
    get_oof(lambda: GradientBoostingRegressor(**{**gbr_opt, 'random_state': 42}),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# Linear baselines
add('enet_01_eng',
    get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            X_eng, y, splits, scale=True))
add('ridge_500_v7',
    get_oof(lambda: Ridge(alpha=500), X_v7, y, splits, scale=True))

print(f"\n  Total: {cnt} models ({time.time()-t_start:.0f}s)")

# ============================================================
# PART 3: MEGA GREEDY BLEND + DIRICHLET
# ============================================================
print("\n--- Part 3: Mega Blend ---"); sys.stdout.flush()

sorted_m = sorted(scores, key=scores.get, reverse=True)
print("  Top 15:")
for i, nm in enumerate(sorted_m[:15], 1):
    print(f"    {i:2d}. {nm:55s} R²={scores[nm]:.4f}")

# Greedy forward
selected=[sorted_m[0]]; remaining=set(sorted_m[1:])
blend=oof_pool[selected[0]].copy(); cur_r2=scores[selected[0]]
print(f"\n  Greedy:")
print(f"    Step 1: {selected[0]:50s} R²={cur_r2:.4f}")
for step in range(2, 15):
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

# Dirichlet on greedy set
sel_oofs=np.column_stack([oof_pool[k] for k in selected])
best_dir=-999; rng=np.random.RandomState(42)
for _ in range(3000000):
    w=rng.dirichlet(np.ones(len(selected)))
    r2=1-np.sum((y-sel_oofs@w)**2)/np.sum((y-y.mean())**2)
    if r2>best_dir: best_dir=r2; best_w=w
print(f"\n  ★ Greedy: R²={cur_r2:.4f}")
print(f"  ★ Dirichlet (3M): R²={best_dir:.4f}")
print("  Weights:")
for nm,w in zip(selected, best_w):
    if w>0.01: print(f"    {nm:55s} w={w:.3f}")

# Also Dirichlet on top-10 overall
top10=sorted_m[:min(10,len(sorted_m))]
top10_oofs=np.column_stack([oof_pool[k] for k in top10])
best10=-999
for _ in range(2000000):
    w=rng.dirichlet(np.ones(len(top10)))
    r2=1-np.sum((y-top10_oofs@w)**2)/np.sum((y-y.mean())**2)
    if r2>best10: best10=r2
print(f"  ★ Top-10 Dirichlet (2M): R²={best10:.4f}")

elapsed=time.time()-t_start
best_overall=max(cur_r2, best_dir, best10)
print(f"\n{'='*60}")
print(f"  V15 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ BEST: R²={best_overall:.4f}")
print(f"  Best single: {sorted_m[0]} R²={scores[sorted_m[0]]:.4f}")
print(f"  HGBR Optuna: R²={hgbr_best_r2:.4f}")
print(f"  Target: 0.65 | Gap: {0.65-best_overall:.4f}")

results = {
    'best_r2': float(best_overall),
    'hgbr_optuna': float(hgbr_best_r2), 'hgbr_params': hgbr_best,
    'best_single': {'name': sorted_m[0], 'r2': float(scores[sorted_m[0]])},
    'greedy_r2': float(cur_r2),
    'greedy_models': selected,
    'all_scores': {k: float(scores[k]) for k in sorted_m},
    'elapsed': elapsed
}
with open('v15_results.json','w') as f:
    json.dump(results,f,indent=2)
print(f"  Saved to v15_results.json"); sys.stdout.flush()
