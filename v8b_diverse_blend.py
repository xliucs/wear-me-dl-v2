#!/usr/bin/env python3
"""
V8b: Diverse model pool + mega-blend.

Skip PyTorch (SIGSEGV + poor convergence on this dataset).
Focus on maximizing diversity in tree/linear models for better blending.

V7 best: single 0.5271, blend 0.5368.
Key: log target helps +0.015. V7 features best for trees.

New ideas:
1. GBR (sklearn) — different implementation, different predictions
2. Multiple random seeds for XGBoost → diversity
3. Quantile XGBoost targets (predict median vs mean)  
4. Target power transforms (Box-Cox)
5. Aggressive Dirichlet with 1M+ trials
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor, BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import boxcox
from scipy.special import inv_boxcox

t_start = time.time()
print("="*60)
print("  V8b: DIVERSE BLEND + TARGET TRANSFORMS")
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
print(f"V7: {X_v7.shape[1]}, eng: {X_eng.shape[1]}")

# Box-Cox transform of target
y_bc, bc_lambda = boxcox(y + 0.01)
print(f"Box-Cox lambda: {bc_lambda:.4f}")

def get_oof(model_fn, X, y_arr, splits, scale=False, log_target=False, bc_target=False):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    if bc_target:
        yt = y_bc
    elif log_target:
        yt = np.log1p(y_arr)
    else:
        yt = y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if bc_target:
            p = inv_boxcox(p, bc_lambda) - 0.01
            p = np.clip(p, 0, None)
        elif log_target:
            p = np.expm1(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_pool = {}
scores = {}

# ============================================================
# PART 1: XGBoost diversity (multiple seeds + configs + targets)
# ============================================================
print("\n--- Part 1: XGBoost Diversity ---")
sys.stdout.flush()

xgb_base = dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
                colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01)

# Best config with different seeds
for seed in [42, 123, 456, 789, 2024]:
    for fs_name, X_fs in [('v7', X_v7), ('eng', X_eng)]:
        name = f'xgb_d3_log_s{seed}_{fs_name}'
        t0 = time.time()
        oof = get_oof(lambda s=seed: xgb.XGBRegressor(**xgb_base, random_state=s, verbosity=0),
                      X_fs, y, splits, log_target=True)
        r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
        print(f"  {name:45s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# XGBoost with Box-Cox target
for fs_name, X_fs in [('v7', X_v7), ('eng', X_eng)]:
    name = f'xgb_d3_bc_{fs_name}'
    t0 = time.time()
    oof = get_oof(lambda: xgb.XGBRegressor(**xgb_base, random_state=42, verbosity=0),
                  X_fs, y, splits, bc_target=True)
    r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
    print(f"  {name:45s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

# Deeper XGBoost configs with log target
for cfg_name, cfg in [
    ('xgb_d4_lr02', dict(n_estimators=500, max_depth=4, learning_rate=0.02, subsample=0.7,
                          colsample_bytree=0.7, min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0)),
    ('xgb_d6_lr01', dict(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.6,
                          colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0)),
    ('xgb_d2_lr05', dict(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5)),
]:
    for fs_name, X_fs in [('v7', X_v7), ('eng', X_eng)]:
        name = f'{cfg_name}_log_{fs_name}'
        t0 = time.time()
        oof = get_oof(lambda c=cfg: xgb.XGBRegressor(**c, random_state=42, verbosity=0),
                      X_fs, y, splits, log_target=True)
        r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
        print(f"  {name:45s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# ============================================================
# PART 2: LightGBM + HGBR + GBR diversity
# ============================================================
print("\n--- Part 2: LGB/HGBR/GBR Diversity ---")
sys.stdout.flush()

for fs_name, X_fs in [('v7', X_v7), ('eng', X_eng)]:
    # LightGBM
    for name, params in [
        ('lgb_d4_log', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.7, min_child_samples=15, verbose=-1)),
        ('lgb_d3_log', dict(n_estimators=500, max_depth=3, learning_rate=0.03, subsample=0.7,
                            colsample_bytree=0.7, min_child_samples=20, verbose=-1)),
    ]:
        full_name = f'{name}_{fs_name}'
        oof = get_oof(lambda p=params: lgb.LGBMRegressor(**p, random_state=42), X_fs, y, splits, log_target=True)
        r2 = r2_score(y, oof); oof_pool[full_name] = oof; scores[full_name] = r2
        print(f"  {full_name:45s} R²={r2:.4f}")
        sys.stdout.flush()
    
    # HGBR
    for name, params in [
        ('hgbr_d4_log', dict(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=10)),
        ('hgbr_d3_log', dict(max_iter=500, max_depth=3, learning_rate=0.05, min_samples_leaf=15)),
    ]:
        full_name = f'{name}_{fs_name}'
        oof = get_oof(lambda p=params: HistGradientBoostingRegressor(**p, random_state=42),
                      X_fs, y, splits, log_target=True)
        r2 = r2_score(y, oof); oof_pool[full_name] = oof; scores[full_name] = r2
        print(f"  {full_name:45s} R²={r2:.4f}")
        sys.stdout.flush()
    
    # GBR (sklearn — different implementation for diversity)
    name = f'gbr_d3_log_{fs_name}'
    t0 = time.time()
    oof = get_oof(lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                      subsample=0.8, random_state=42),
                  X_fs, y, splits, log_target=True)
    r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
    print(f"  {name:45s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

# ============================================================
# PART 3: Linear models + SVR
# ============================================================
print("\n--- Part 3: Linear + SVR ---")
sys.stdout.flush()

for fs_name, X_fs in [('v7', X_v7), ('eng', X_eng)]:
    for mname, mfn in [
        ('enet_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ('enet_005', lambda: ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=5000)),
        ('ridge_100', lambda: Ridge(alpha=100)),
        ('ridge_500', lambda: Ridge(alpha=500)),
        ('bayesian', lambda: BayesianRidge()),
        ('svr_c10', lambda: SVR(kernel='rbf', C=10, gamma='scale')),
        ('svr_c50', lambda: SVR(kernel='rbf', C=50, gamma='scale')),
        ('knn_15', lambda: KNeighborsRegressor(n_neighbors=15, weights='distance')),
    ]:
        full_name = f'{mname}_{fs_name}'
        t0 = time.time()
        oof = get_oof(mfn, X_fs, y, splits, scale=True)
        r2 = r2_score(y, oof); oof_pool[full_name] = oof; scores[full_name] = r2
        print(f"  {full_name:45s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# ET
for fs_name, X_fs in [('v7', X_v7), ('eng', X_eng)]:
    name = f'et200_{fs_name}'
    oof = get_oof(lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
                  X_fs, y, splits)
    r2 = r2_score(y, oof); oof_pool[name] = oof; scores[name] = r2
    print(f"  {name:45s} R²={r2:.4f}")
    sys.stdout.flush()

print(f"\n  Total: {len(oof_pool)} models ({time.time()-t_start:.0f}s)")

# ============================================================
# PART 4: MEGA-BLEND with 1M trials
# ============================================================
print("\n--- Part 4: Mega-Blend ---")
sys.stdout.flush()

sorted_models = sorted(scores, key=scores.get, reverse=True)
print(f"  Top 15:")
for i, name in enumerate(sorted_models[:15], 1):
    print(f"    {i:2d}. {name:45s} R²={scores[name]:.4f}")

best_blend_r2 = -999
for top_k in [5, 8, 10, 15, 20, 25, 30]:
    if top_k > len(sorted_models): break
    top_names = sorted_models[:top_k]
    top_oofs = np.column_stack([oof_pool[k] for k in top_names])
    best_r2 = -999
    rng = np.random.RandomState(42)
    n_trials = 1000000
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(top_k))
        pred = top_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > best_r2: best_r2 = r2; best_w = w
    print(f"  Top-{top_k:2d} Dirichlet (1M): R²={best_r2:.4f}")
    if best_r2 > best_blend_r2:
        best_blend_r2 = best_r2
        best_blend_k = top_k
        best_blend_weights = dict(zip(top_names, [f'{w:.4f}' for w in best_w]))
    sys.stdout.flush()

# Ridge stacking
print("\n  Stacking:")
oof_matrix = np.column_stack([oof_pool[k] for k in sorted_models[:20]])
for sname, sfn in [
    ('ridge_10', lambda: Ridge(alpha=10)),
    ('enet_05', lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000)),
]:
    oof = get_oof(sfn, oof_matrix, y, splits, scale=True)
    r2 = r2_score(y, oof)
    print(f"    Stack {sname:10s}: R²={r2:.4f}")
sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  V8b SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
best_single = sorted_models[0]
print(f"  ★ BEST SINGLE: {best_single} R²={scores[best_single]:.4f}")
print(f"  ★ BEST BLEND (top-{best_blend_k}): R²={best_blend_r2:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_blend_r2:.4f}")

results = {
    'best_single': {'name': best_single, 'r2': float(scores[best_single])},
    'best_blend_r2': float(best_blend_r2),
    'best_blend_k': best_blend_k,
    'best_blend_weights': best_blend_weights,
    'all_scores': {k: float(scores[k]) for k in sorted_models[:25]},
    'n_models': len(oof_pool),
    'elapsed': elapsed
}
with open('v8b_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v8b_results.json")
sys.stdout.flush()
