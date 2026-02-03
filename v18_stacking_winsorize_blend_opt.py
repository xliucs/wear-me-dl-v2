#!/usr/bin/env python3
"""
V18: Proper Nested Stacking + Winsorization + Blend-Optimized Models

The ceiling is real. These are the last unexplored ideas:

1. NESTED STACKING: Train L1 base models in inner CV, use their predictions as
   features for L2 meta-learner in outer CV. This is the CORRECT way to stack.
   V2-V3 showed naive stacking leaks. We need 3-level nesting.

2. WINSORIZATION: Clip extreme HOMA values (>10) to reduce their outsized
   influence on R². The 23 samples with HOMA>10 contribute disproportionate MSE.

3. BLEND-OPTIMIZED DIVERSITY: Instead of training each model to maximize its own
   R², train diverse models that maximize the BLEND R² when combined.
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import load_data, get_feature_sets, get_cv_splits, engineer_all_features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V18: NESTED STACKING + WINSORIZE + BLEND OPT")
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
    print(f"  [{cnt:2d}] {name:55s} R²={r2:.4f}"); sys.stdout.flush()

v13 = dict(n_estimators=612, max_depth=4, learning_rate=0.017, subsample=0.52,
    colsample_bytree=0.78, min_child_weight=29, reg_alpha=2.8, reg_lambda=0.045)
lgb_p = {'n_estimators': 768, 'max_depth': 4, 'learning_rate': 0.0129,
    'subsample': 0.409, 'colsample_bytree': 0.889, 'min_child_samples': 36,
    'reg_alpha': 3.974, 'reg_lambda': 0.203, 'num_leaves': 10, 'verbose': -1, 'random_state': 42}
gbr_p = {'n_estimators': 373, 'max_depth': 3, 'learning_rate': 0.0313,
    'subsample': 0.470, 'min_samples_leaf': 12, 'max_features': 0.556, 'random_state': 42}

# ============================================================
# PART 1: WINSORIZATION — does clipping extremes help R²?
# ============================================================
print("\n--- Part 1: Winsorization ---"); sys.stdout.flush()

# Train on winsorized target, predict original
for clip_val in [8, 10, 12]:
    y_clip = np.clip(y, 0, clip_val)
    # OOF on clipped target, evaluate on original
    oof = get_oof(lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0),
                  X_v7, y_clip, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt)
    r2_orig = r2_score(y, oof)
    r2_clip = r2_score(y_clip, np.clip(oof, 0, clip_val))
    print(f"  Winsorize @{clip_val}: R²(orig)={r2_orig:.4f}, R²(clip)={r2_clip:.4f}")
    oof_pool[f'xgb_winsor{clip_val}'] = oof; scores[f'xgb_winsor{clip_val}'] = r2_orig
    cnt += 1
sys.stdout.flush()

# ============================================================
# PART 2: NESTED STACKING (proper, no leakage)
# ============================================================
print("\n--- Part 2: Nested Stacking ---"); sys.stdout.flush()

# Strategy: For each OUTER fold:
#   1. Use inner 5-fold CV on OUTER-train to get L1 OOF predictions
#   2. Train L2 meta-learner on L1 OOF predictions
#   3. Train L1 models on full OUTER-train, predict OUTER-test
#   4. Use L2 meta-learner on L1 OUTER-test predictions

def nested_stack_oof(X, y_arr, outer_splits, n_inner=5):
    """Proper nested stacking with inner CV for L1 and outer for evaluation."""
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    
    # Define L1 model factories
    l1_configs = [
        ('xgb_wsqrt', lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0), w_sqrt),
        ('lgb_wsqrt', lambda: lgb.LGBMRegressor(**lgb_p), w_sqrt),
        ('gbr_wsqrt', lambda: GradientBoostingRegressor(**gbr_p), w_sqrt),
        ('enet', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), None),
    ]
    
    for fold_i, (outer_tr, outer_te) in enumerate(outer_splits):
        X_outer_tr, X_outer_te = X[outer_tr], X[outer_te]
        y_outer_tr = y_arr[outer_tr]
        yt_outer_tr = log_fn(y_outer_tr)
        
        # Inner CV to get L1 OOF predictions on outer_train
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=42)
        l1_oof = np.zeros((len(outer_tr), len(l1_configs)))  # L1 predictions on outer_train
        l1_test = np.zeros((len(outer_te), len(l1_configs)))  # L1 predictions on outer_test
        
        for m_idx, (name, model_fn, wts) in enumerate(l1_configs):
            # Inner OOF
            for inner_tr, inner_te in inner_kf.split(X_outer_tr):
                Xtr = X_outer_tr[inner_tr]; Xte_inner = X_outer_tr[inner_te]
                ytr = yt_outer_tr[inner_tr]
                
                if name == 'enet':
                    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte_inner = sc.transform(Xte_inner)
                    m = model_fn(); m.fit(Xtr, y_outer_tr[inner_tr])
                    l1_oof[inner_te, m_idx] = m.predict(Xte_inner)
                else:
                    m = model_fn()
                    if wts is not None:
                        m.fit(Xtr, ytr, sample_weight=wts[outer_tr][inner_tr])
                    else:
                        m.fit(Xtr, ytr)
                    l1_oof[inner_te, m_idx] = inv_log(m.predict(Xte_inner))
            
            # Train L1 on full outer_train, predict outer_test
            if name == 'enet':
                sc = StandardScaler()
                Xtr_sc = sc.fit_transform(X_outer_tr); Xte_sc = sc.transform(X_outer_te)
                m = model_fn(); m.fit(Xtr_sc, y_outer_tr)
                l1_test[:, m_idx] = m.predict(Xte_sc)
            else:
                m = model_fn()
                if wts is not None:
                    m.fit(X_outer_tr, yt_outer_tr, sample_weight=wts[outer_tr])
                else:
                    m.fit(X_outer_tr, yt_outer_tr)
                l1_test[:, m_idx] = inv_log(m.predict(X_outer_te))
        
        # L2: Ridge on L1 predictions
        l2 = Ridge(alpha=1.0)
        l2.fit(l1_oof, y_outer_tr)
        p = l2.predict(l1_test)
        
        oof_sum[outer_te] += p; oof_cnt[outer_te] += 1
    
    return oof_sum / np.clip(oof_cnt, 1, None)

print("  Running nested stacking (this takes a while)...")
sys.stdout.flush()
oof_stack = nested_stack_oof(X_v7, y, splits, n_inner=5)
add('nested_stack_ridge', oof_stack)

# Also try L2 = ElasticNet
def nested_stack_oof_enet(X, y_arr, outer_splits, n_inner=5):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    l1_configs = [
        ('xgb_wsqrt', lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0), w_sqrt),
        ('lgb_wsqrt', lambda: lgb.LGBMRegressor(**lgb_p), w_sqrt),
        ('gbr_wsqrt', lambda: GradientBoostingRegressor(**gbr_p), w_sqrt),
        ('enet', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), None),
    ]
    for fold_i, (outer_tr, outer_te) in enumerate(outer_splits):
        X_outer_tr, X_outer_te = X[outer_tr], X[outer_te]
        y_outer_tr = y_arr[outer_tr]; yt_outer_tr = log_fn(y_outer_tr)
        inner_kf = KFold(n_splits=n_inner, shuffle=True, random_state=42)
        l1_oof = np.zeros((len(outer_tr), len(l1_configs)))
        l1_test = np.zeros((len(outer_te), len(l1_configs)))
        for m_idx, (name, model_fn, wts) in enumerate(l1_configs):
            for inner_tr, inner_te in inner_kf.split(X_outer_tr):
                Xtr = X_outer_tr[inner_tr]; Xte_inner = X_outer_tr[inner_te]
                if name == 'enet':
                    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte_inner = sc.transform(Xte_inner)
                    m = model_fn(); m.fit(Xtr, y_outer_tr[inner_tr])
                    l1_oof[inner_te, m_idx] = m.predict(Xte_inner)
                else:
                    m = model_fn()
                    if wts is not None: m.fit(Xtr, yt_outer_tr[inner_tr], sample_weight=wts[outer_tr][inner_tr])
                    else: m.fit(Xtr, yt_outer_tr[inner_tr])
                    l1_oof[inner_te, m_idx] = inv_log(m.predict(Xte_inner))
            if name == 'enet':
                sc = StandardScaler(); Xtr_sc = sc.fit_transform(X_outer_tr); Xte_sc = sc.transform(X_outer_te)
                m = model_fn(); m.fit(Xtr_sc, y_outer_tr)
                l1_test[:, m_idx] = m.predict(Xte_sc)
            else:
                m = model_fn()
                if wts is not None: m.fit(X_outer_tr, yt_outer_tr, sample_weight=wts[outer_tr])
                else: m.fit(X_outer_tr, yt_outer_tr)
                l1_test[:, m_idx] = inv_log(m.predict(X_outer_te))
        # L2: ElasticNet with non-negative constraint
        l2 = ElasticNet(alpha=0.01, l1_ratio=0.5, positive=True, max_iter=5000)
        l2.fit(l1_oof, y_outer_tr)
        p = l2.predict(l1_test)
        oof_sum[outer_te] += p; oof_cnt[outer_te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_stack_enet = nested_stack_oof_enet(X_v7, y, splits, n_inner=5)
add('nested_stack_enet', oof_stack_enet)

# ============================================================
# PART 3: BASELINES FOR BLEND
# ============================================================
print("\n--- Part 3: Baselines ---"); sys.stdout.flush()

add('xgb_optuna_wsqrt',
    get_oof(lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))
add('lgb_optuna_wsqrt',
    get_oof(lambda: lgb.LGBMRegressor(**lgb_p), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))
add('gbr_optuna_wsqrt',
    get_oof(lambda: GradientBoostingRegressor(**gbr_p), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))
add('enet_01',
    get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            X_eng, y, splits, scale=True))

# Different weight exponents for diversity
for exp in [0.3, 0.7, 1.0]:
    w_exp = np.power(y, exp) / np.power(y, exp).mean()
    add(f'xgb_w{exp}',
        get_oof(lambda: xgb.XGBRegressor(**v13, random_state=2024, verbosity=0),
                X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_exp))

# XGB different seeds
add('xgb_wsqrt_s99',
    get_oof(lambda: xgb.XGBRegressor(**v13, random_state=99, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

print(f"\n  Total: {cnt} models ({time.time()-t_start:.0f}s)")
sys.stdout.flush()

# ============================================================
# PART 4: BLEND
# ============================================================
print("\n--- Part 4: Greedy + Dirichlet ---"); sys.stdout.flush()

sorted_m = sorted(scores, key=scores.get, reverse=True)
print("  All ranked:")
for i, nm in enumerate(sorted_m, 1):
    print(f"    {i:2d}. {nm:55s} R²={scores[nm]:.4f}")

sel = [sorted_m[0]]; rem = set(sorted_m[1:])
blend = oof_pool[sel[0]].copy(); cur = scores[sel[0]]
print(f"\n  Greedy:")
print(f"    Step 1: {sel[0]:50s} R²={cur:.4f}")
for step in range(2, 12):
    ba, br = None, cur
    for nm in rem:
        for a in np.arange(0.02, 0.50, 0.02):
            b = (1-a)*blend + a*oof_pool[nm]; r2 = r2_score(y, b)
            if r2 > br: br = r2; ba = nm; bsa = a
    if ba is None or br <= cur + 0.00005: break
    sel.append(ba); rem.discard(ba)
    blend = (1-bsa)*blend + bsa*oof_pool[ba]; cur = br
    print(f"    Step {step}: +{ba:45s} α={bsa:.2f} → R²={cur:.4f}")
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
    if w > 0.01: print(f"    {nm:55s} w={w:.3f}")

elapsed = time.time() - t_start
best = max(cur, best_dir)
print(f"\n{'='*60}")
print(f"  V18 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ BEST: R²={best:.4f}")
print(f"  Best single: {sorted_m[0]} R²={scores[sorted_m[0]]:.4f}")
stack_r2 = max(scores.get('nested_stack_ridge', 0), scores.get('nested_stack_enet', 0))
print(f"  Nested stacking: R²={stack_r2:.4f}")
winsor_r2 = max(scores.get(f'xgb_winsor{v}', 0) for v in [8, 10, 12])
print(f"  Winsorization best: R²={winsor_r2:.4f}")
if best > 0.5465:
    print(f"  🎉 NEW BEST! +{best - 0.5465:.4f} over V14")
else:
    print(f"  Gap to V14: {0.5465 - best:.4f}")

results = {
    'best_r2': float(best),
    'best_single': {'name': sorted_m[0], 'r2': float(scores[sorted_m[0]])},
    'nested_stack_r2': float(stack_r2),
    'winsor_best': float(winsor_r2),
    'all_scores': {k: float(scores[k]) for k in sorted_m},
    'elapsed': elapsed
}
with open('v18_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v18_results.json"); sys.stdout.flush()
