#!/usr/bin/env python3
"""
V16: Nested Target Encoding + Piecewise Models + Diverse Blending

Strategy to break 0.546:
1. NESTED target encoding: compute TE inside each fold to avoid leakage
2. Piecewise models: separate low/high HOMA specialists with soft routing
3. Diverse model families for better blending
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V16: NESTED TARGET ENCODING + PIECEWISE + BLEND")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, _, all_cols, _ = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()

# --- V7 feature engineering ---
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
v7_cols = list(X_v7_df.columns)

log_fn = np.log1p; inv_log = np.expm1

# --- Standard OOF ---
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

# --- NESTED Target Encoding OOF (no leakage) ---
# Key: TE features are computed INSIDE each fold using only that fold's train data
BIN_CONFIGS_1D = [('bmi', 6), ('glucose', 6), ('triglycerides', 5), ('hdl', 5)]
BIN_CONFIGS_2D = [('bmi', 'glucose', 4, 4), ('bmi', 'triglycerides', 4, 4),
                   ('glucose', 'triglycerides', 4, 4), ('glucose', 'hdl', 4, 4)]

def compute_te_for_split(X_df_vals, y_arr, tr, te, smoothing=10):
    """Compute target-encoded features using ONLY training data, apply to test."""
    global_mean = y_arr[tr].mean()
    te_train = np.zeros((len(tr), 0))
    te_test = np.zeros((len(te), 0))
    
    # 1D bins
    for col, nbins in BIN_CONFIGS_1D:
        col_idx = all_cols.index(col)
        col_vals = X_df_vals[:, col_idx]
        # Compute bin edges from training data only
        train_vals = col_vals[tr]
        try:
            _, bin_edges = pd.qcut(train_vals, q=nbins, labels=False, duplicates='drop', retbins=True)
        except:
            # Fallback: use equal-width bins
            bin_edges = np.linspace(train_vals.min(), train_vals.max(), nbins+1)
        
        train_bins = np.digitize(col_vals[tr], bin_edges[1:-1])
        test_bins = np.digitize(col_vals[te], bin_edges[1:-1])
        
        # Compute bin means from training data
        bin_df = pd.DataFrame({'bin': train_bins, 'y': y_arr[tr]})
        stats = bin_df.groupby('bin')['y'].agg(['mean', 'count'])
        bin_enc = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        tr_enc = np.array([bin_enc.get(b, global_mean) for b in train_bins])
        te_enc = np.array([bin_enc.get(b, global_mean) for b in test_bins])
        
        te_train = np.column_stack([te_train, tr_enc]) if te_train.shape[1] > 0 else tr_enc.reshape(-1,1)
        te_test = np.column_stack([te_test, te_enc]) if te_test.shape[1] > 0 else te_enc.reshape(-1,1)
    
    # 2D bins
    for col1, col2, n1, n2 in BIN_CONFIGS_2D:
        idx1 = all_cols.index(col1); idx2 = all_cols.index(col2)
        v1_tr = X_df_vals[tr, idx1]; v2_tr = X_df_vals[tr, idx2]
        v1_te = X_df_vals[te, idx1]; v2_te = X_df_vals[te, idx2]
        
        try:
            _, e1 = pd.qcut(v1_tr, q=n1, labels=False, duplicates='drop', retbins=True)
            _, e2 = pd.qcut(v2_tr, q=n2, labels=False, duplicates='drop', retbins=True)
        except:
            e1 = np.linspace(v1_tr.min(), v1_tr.max(), n1+1)
            e2 = np.linspace(v2_tr.min(), v2_tr.max(), n2+1)
        
        b1_tr = np.digitize(v1_tr, e1[1:-1]); b2_tr = np.digitize(v2_tr, e2[1:-1])
        b1_te = np.digitize(v1_te, e1[1:-1]); b2_te = np.digitize(v2_te, e2[1:-1])
        combo_tr = b1_tr * 100 + b2_tr
        combo_te = b1_te * 100 + b2_te
        
        bin_df = pd.DataFrame({'bin': combo_tr, 'y': y_arr[tr]})
        stats = bin_df.groupby('bin')['y'].agg(['mean', 'count'])
        bin_enc = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        
        tr_enc = np.array([bin_enc.get(b, global_mean) for b in combo_tr])
        te_enc = np.array([bin_enc.get(b, global_mean) for b in combo_te])
        
        te_train = np.column_stack([te_train, tr_enc])
        te_test = np.column_stack([te_test, te_enc])
    
    return te_train, te_test

def get_oof_with_te(model_fn, X_base, X_raw, y_arr, splits, target_fn=None, inv_fn=None, weights=None):
    """OOF with nested target encoding — TE computed fresh inside each fold."""
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = target_fn(y_arr) if target_fn else y_arr
    for fold_i, (tr, te) in enumerate(splits):
        # Compute TE features from train data only
        te_train, te_test = compute_te_for_split(X_raw, y_arr, tr, te)
        
        # Concatenate base features + TE features
        Xtr = np.column_stack([X_base[tr], te_train])
        Xte = np.column_stack([X_base[te], te_test])
        
        m = model_fn()
        if weights is not None:
            m.fit(Xtr, yt[tr], sample_weight=weights[tr])
        else:
            m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if inv_fn: p = inv_fn(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

X_raw = X_df[all_cols].values  # raw features for bin computation

oof_pool = {}; scores = {}; cnt = 0
def add(name, oof):
    global cnt; cnt += 1
    r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
    print(f"  [{cnt:2d}] {name:55s} R²={r2:.4f}"); sys.stdout.flush()

# ============================================================
# PART 1: CORE MODELS (baselines)
# ============================================================
print("\n--- Part 1: Core Models ---"); sys.stdout.flush()

v13_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017, subsample=0.52,
    colsample_bytree=0.78, min_child_weight=29, reg_alpha=2.8, reg_lambda=0.045)

add('xgb_optuna_wsqrt_v7',
    get_oof(lambda: xgb.XGBRegressor(**v13_params, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

lgb_params = {"n_estimators": 768, "max_depth": 4, "learning_rate": 0.0129,
    "subsample": 0.409, "colsample_bytree": 0.889, "min_child_samples": 36,
    "reg_alpha": 3.974, "reg_lambda": 0.203, "num_leaves": 10, "verbose": -1, "random_state": 42}
add('lgb_optuna_wsqrt_v7',
    get_oof(lambda: lgb.LGBMRegressor(**lgb_params), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

gbr_params = {"n_estimators": 373, "max_depth": 3, "learning_rate": 0.0313,
    "subsample": 0.470, "min_samples_leaf": 12, "max_features": 0.556, "random_state": 42}
add('gbr_optuna_wsqrt_v7',
    get_oof(lambda: GradientBoostingRegressor(**gbr_params), X_v7, y, splits,
            target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# ============================================================
# PART 2: NESTED TARGET ENCODING MODELS (honest, no leakage)
# ============================================================
print("\n--- Part 2: Nested Target Encoding (honest) ---"); sys.stdout.flush()
print("  Computing TE features inside each fold..."); sys.stdout.flush()

add('xgb_optuna_wsqrt_v7+te',
    get_oof_with_te(lambda: xgb.XGBRegressor(**v13_params, random_state=2024, verbosity=0),
                     X_v7, X_raw, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

add('lgb_optuna_wsqrt_v7+te',
    get_oof_with_te(lambda: lgb.LGBMRegressor(**lgb_params), X_v7, X_raw, y, splits,
                     target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

add('gbr_optuna_wsqrt_v7+te',
    get_oof_with_te(lambda: GradientBoostingRegressor(**gbr_params), X_v7, X_raw, y, splits,
                     target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# ============================================================
# PART 3: PIECEWISE MODELS
# ============================================================
print("\n--- Part 3: Piecewise Models ---"); sys.stdout.flush()

def get_piecewise_oof(X, y_arr, splits, threshold=2.5, steepness=2.0):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = log_fn(y_arr)
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        ytr = yt[tr]; y_raw_tr = y_arr[tr]; w_tr = w_sqrt[tr]
        
        m_full = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.6, colsample_bytree=0.7, random_state=2024, verbosity=0)
        m_full.fit(Xtr, ytr, sample_weight=w_tr)
        p_full = inv_log(m_full.predict(Xte))
        
        low_mask = y_raw_tr < threshold * 1.5
        if low_mask.sum() > 80:
            m_low = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                subsample=0.6, colsample_bytree=0.7, random_state=2024, verbosity=0)
            m_low.fit(Xtr[low_mask], ytr[low_mask])
            p_low = inv_log(m_low.predict(Xte))
        else:
            p_low = p_full
        
        high_mask = y_raw_tr > threshold * 0.5
        if high_mask.sum() > 80:
            w_h = y_raw_tr[high_mask] / y_raw_tr[high_mask].mean()
            m_high = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.6, colsample_bytree=0.7, random_state=2024, verbosity=0)
            m_high.fit(Xtr[high_mask], ytr[high_mask], sample_weight=w_h)
            p_high = inv_log(m_high.predict(Xte))
        else:
            p_high = p_full
        
        sigmoid = 1 / (1 + np.exp(-(p_full - threshold) * steepness))
        p_piece = (1 - sigmoid) * p_low + sigmoid * p_high
        
        oof_sum[te] += p_piece; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

for thresh, steep in [(2.0, 2.0), (2.5, 2.0)]:
    add(f'piecewise_t{thresh}_s{steep}',
        get_piecewise_oof(X_v7, y, splits, threshold=thresh, steepness=steep))

# (Skipped LGB piecewise — too slow for marginal gain)

# ============================================================
# PART 4: DIVERSITY MODELS
# ============================================================
print("\n--- Part 4: Diversity Models ---"); sys.stdout.flush()

# XGB different seeds
for seed in [99, 7777]:
    add(f'xgb_wsqrt_v7_s{seed}',
        get_oof(lambda s=seed: xgb.XGBRegressor(**v13_params, random_state=s, verbosity=0),
                X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

# XGB d3 unweighted
xgb_d3 = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
               colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01)
add('xgb_d3_log_v7',
    get_oof(lambda: xgb.XGBRegressor(**xgb_d3, random_state=2024, verbosity=0),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))

# ElasticNet
add('enet_01_eng',
    get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            X_eng, y, splits, scale=True))

# Huber
add('huber_eng',
    get_oof(lambda: HuberRegressor(max_iter=500, alpha=0.01, epsilon=1.5),
            X_eng, y, splits, scale=True))

# LGB unweighted
add('lgb_d3_log_v7',
    get_oof(lambda: lgb.LGBMRegressor(n_estimators=500, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=20, verbose=-1, random_state=42),
     X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log))

# ExtraTrees (fast with n_jobs=-1, provides bagging diversity)
add('extratrees_wsqrt_v7',
    get_oof(lambda: ExtraTreesRegressor(n_estimators=300, max_depth=12, min_samples_leaf=10,
                                         max_features=0.74, n_jobs=-1, random_state=42),
            X_v7, y, splits, target_fn=log_fn, inv_fn=inv_log, weights=w_sqrt))

print(f"\n  Total: {cnt} models ({time.time()-t_start:.0f}s)")
sys.stdout.flush()

# ============================================================
# PART 5: GREEDY BLEND + DIRICHLET
# ============================================================
print("\n--- Part 5: Greedy + Dirichlet ---"); sys.stdout.flush()

sorted_m = sorted(scores, key=scores.get, reverse=True)
print("  All models ranked:")
for i, nm in enumerate(sorted_m, 1):
    print(f"    {i:2d}. {nm:55s} R²={scores[nm]:.4f}")

# Greedy
selected = [sorted_m[0]]; remaining = set(sorted_m[1:])
blend = oof_pool[selected[0]].copy(); cur_r2 = scores[selected[0]]
print(f"\n  Greedy Blend:")
print(f"    Step 1: {selected[0]:50s} R²={cur_r2:.4f}")
for step in range(2, 15):
    best_add = None; best_r2 = cur_r2
    for nm in remaining:
        for alpha in np.arange(0.02, 0.50, 0.02):
            b = (1-alpha)*blend + alpha*oof_pool[nm]
            r2 = r2_score(y, b)
            if r2 > best_r2: best_r2 = r2; best_add = nm; best_a = alpha
    if best_add is None or best_r2 <= cur_r2 + 0.00005: break
    selected.append(best_add); remaining.discard(best_add)
    blend = (1-best_a)*blend + best_a*oof_pool[best_add]; cur_r2 = best_r2
    print(f"    Step {step}: +{best_add:45s} α={best_a:.2f} → R²={cur_r2:.4f}")
    sys.stdout.flush()

# Dirichlet over selected
sel_oofs = np.column_stack([oof_pool[k] for k in selected])
best_dir = -999; rng = np.random.RandomState(42)
for _ in range(3000000):
    w = rng.dirichlet(np.ones(len(selected)))
    r2 = 1 - np.sum((y - sel_oofs@w)**2) / np.sum((y - y.mean())**2)
    if r2 > best_dir: best_dir = r2; best_w = w

print(f"\n  ★ Greedy: R²={cur_r2:.4f}")
print(f"  ★ Dirichlet ({len(selected)} models): R²={best_dir:.4f}")
print("  Weights:")
for nm, w in zip(selected, best_w):
    if w > 0.01: print(f"    {nm:55s} w={w:.3f}")

# Dirichlet over top-12
sel_all = sorted_m[:min(12, len(sorted_m))]
all_oofs = np.column_stack([oof_pool[k] for k in sel_all])
best_all_dir = -999
for _ in range(3000000):
    w = rng.dirichlet(np.ones(len(sel_all)))
    r2 = 1 - np.sum((y - all_oofs@w)**2) / np.sum((y - y.mean())**2)
    if r2 > best_all_dir: best_all_dir = r2; best_all_w = w

print(f"  ★ Dirichlet (top-12): R²={best_all_dir:.4f}")
if best_all_dir > best_dir:
    print("  Top-12 weights:")
    for nm, w in zip(sel_all, best_all_w):
        if w > 0.01: print(f"    {nm:55s} w={w:.3f}")

elapsed = time.time() - t_start
best_overall = max(cur_r2, best_dir, best_all_dir)
pw_scores = [scores[k] for k in scores if 'piece' in k]
pw_best = max(pw_scores) if pw_scores else 0
te_scores = [scores[k] for k in scores if '+te' in k]
te_best = max(te_scores) if te_scores else 0

print(f"\n{'='*60}")
print(f"  V16 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
print(f"  ★ BEST BLEND: R²={best_overall:.4f}")
print(f"  Best single: {sorted_m[0]} R²={scores[sorted_m[0]]:.4f}")
print(f"  Best nested TE single: R²={te_best:.4f}")
print(f"  Piecewise best: R²={pw_best:.4f}")
if best_overall > 0.5465:
    print(f"  🎉 NEW BEST! +{best_overall - 0.5465:.4f} over V14")
else:
    print(f"  Gap to V14: {0.5465 - best_overall:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_overall:.4f}")

results = {
    'best_r2': float(best_overall),
    'best_single': {'name': sorted_m[0], 'r2': float(scores[sorted_m[0]])},
    'nested_te_best': float(te_best),
    'piecewise_best': float(pw_best),
    'greedy_r2': float(cur_r2),
    'dirichlet_r2': float(best_dir),
    'dirichlet_top12_r2': float(best_all_dir),
    'all_scores': {k: float(scores[k]) for k in sorted_m},
    'elapsed': elapsed
}
with open('v16_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v16_results.json"); sys.stdout.flush()
