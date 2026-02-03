#!/usr/bin/env python3
"""
V28: Maximum Diversity Blending

From post-hoc + V27 learnings:
- All feature signal is extracted (residual-feature corr ~0)
- Stratification hurts (not enough data per stratum)
- Train-test gap is information loss, not overfitting
- V20 best (0.5467) came from LGB_QT + ElasticNet diversity

STRATEGY: Create maximally diverse models via:
1. Different feature transformations (raw, QT, log, rank)
2. Different model types (XGB, LGB, ElasticNet, Ridge, GBR)
3. Different hyperparams (shallow/deep, more/less reg)
4. Different targets (log1p, sqrt, direct)
5. Different sample weighting (uniform, sqrt, quantile)

Then blend EVERYTHING via Dirichlet.
The key insight: even weak models add value IF their errors are uncorrelated.
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V28: MAXIMUM DIVERSITY BLEND")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# Targets
y_log = np.log1p(y)
y_sqrt = np.sqrt(y)

# Weights
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
w_uniform = np.ones(n)

# V7 features
def eng_v7(X_df, cols):
    X = X_df[cols].copy()
    g=X['glucose'].clip(lower=1); t=X['triglycerides'].clip(lower=1)
    h=X['hdl'].clip(lower=1); b=X['bmi']; l=X['ldl']
    tc=X['total cholesterol']; nh=X['non hdl']; ch=X['chol/hdl']; a=X['age']
    rhr=X['Resting Heart Rate (mean)']; hrv=X['HRV (mean)'].clip(lower=1)
    stp=X['STEPS (mean)'].clip(lower=1); slp=X['SLEEP Duration (mean)']
    for nm,v in [('tyg',np.log(t*g/2)),('tyg_bmi',np.log(t*g/2)*b),
        ('mets_ir',np.log(2*g+t)*b/np.log(h)),('trig_hdl',t/h),('trig_hdl_log',np.log1p(t/h)),
        ('vat_proxy',b*t/h),('ir_proxy',g*b*t/(h*100)),('glucose_bmi',g*b),
        ('glucose_sq',g**2),('glucose_log',np.log(g)),('glucose_hdl',g/h),
        ('glucose_trig',g*t/1000),('glucose_non_hdl',g*nh/100),('glucose_chol_hdl',g*ch),
        ('bmi_sq',b**2),('bmi_cubed',b**3),('bmi_age',b*a),('age_sq',a**2),
        ('bmi_rhr',b*rhr),('bmi_hrv_inv',b/hrv),('bmi_stp_inv',b/stp*1000),
        ('rhr_hrv',rhr/hrv),('cardio_fitness',hrv*stp/rhr),
        ('met_load',b*rhr/stp*1000),('sed_risk',b**2*rhr/(stp*hrv)),
        ('glucose_rhr',g*rhr),('glucose_hrv_inv',g/hrv),('tyg_bmi_rhr',np.log(t*g/2)*b*rhr/1000),
        ('log_homa_proxy',np.log(g)+np.log(b)+np.log(t)-np.log(h)),
        ('ir_proxy_log',np.log1p(g*b*t/(h*100))),('ir_proxy_rhr',g*b*t/(h*100)*rhr/100),
        ('non_hdl_ratio',nh/h),('chol_glucose',tc*g/1000),('ldl_glucose',l*g/1000),
        ('bmi_trig',b*t/1000),('bmi_sex',b*X['sex_num']),('glucose_age',g*a/100),
        ('rhr_skew',(rhr-X['Resting Heart Rate (median)'])/X['Resting Heart Rate (std)'].clip(lower=0.01)),
        ('hrv_skew',(hrv-X['HRV (median)'])/X['HRV (std)'].clip(lower=0.01)),
        ('stp_skew',(stp-X['STEPS (median)'])/X['STEPS (std)'].clip(lower=0.01)),
        ('slp_skew',(slp-X['SLEEP Duration (median)'])/X['SLEEP Duration (std)'].clip(lower=0.01)),
        ('azm_skew',(X['AZM Weekly (mean)']-X['AZM Weekly (median)'])/X['AZM Weekly (std)'].clip(lower=0.01)),
        ('rhr_cv',X['Resting Heart Rate (std)'].clip(lower=0.01)/rhr),
        ('hrv_cv',X['HRV (std)'].clip(lower=0.01)/hrv),
        ('stp_cv',X['STEPS (std)'].clip(lower=0.01)/stp),
        ('slp_cv',X['SLEEP Duration (std)'].clip(lower=0.01)/slp),
        ('azm_cv',X['AZM Weekly (std)'].clip(lower=0.01)/X['AZM Weekly (mean)'].clip(lower=0.01)),
    ]:
        X[nm] = v
    return X.fillna(0).values

X_v7 = eng_v7(X_df, all_cols)

def get_oof(model_fn, X, y_target, splits, weights=None, transform_X=None, inverse_target=None):
    """Flexible OOF with optional per-fold X transform and target inverse."""
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        Xtr, Xte = X[tr].copy(), X[te].copy()
        wtr = weights[tr] if weights is not None else None
        
        if transform_X == 'qt':
            qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=42)
            Xtr = qt.fit_transform(Xtr)
            Xte = qt.transform(Xte)
        elif transform_X == 'scale':
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        
        m = model_fn(Xtr, y_target[tr], wtr)
        pred = m.predict(Xte)
        oof_sum[te] += pred
        oof_count[te] += 1
    
    oof = oof_sum / np.clip(oof_count, 1, None)
    if inverse_target == 'log':
        return np.expm1(oof)
    elif inverse_target == 'sqrt':
        return oof ** 2
    return oof

all_preds = {}

# ============================================================
# GROUP A: XGB variants (different params)
# ============================================================
print("\n--- Group A: XGB variants ---")

configs_xgb = {
    'xgb_v13': dict(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045),
    'xgb_d3': dict(n_estimators=800, max_depth=3, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045),
    'xgb_d3_s0': dict(n_estimators=800, max_depth=3, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045),  # seed=0
}

for name, params in configs_xgb.items():
    seed = 0 if 's0' in name else 42
    def fn(Xtr, ytr, wtr, p=params, s=seed):
        m = xgb.XGBRegressor(**p, random_state=s, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m
    oof = get_oof(fn, X_v7, y_log, splits, weights=w_sqrt, inverse_target='log')
    r2 = r2_score(y, oof)
    print(f"  {name}: R² = {r2:.4f}")
    all_preds[name] = oof
sys.stdout.flush()

# XGB with QT input
def xgb_v13_fn(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof = get_oof(xgb_v13_fn, X_v7, y_log, splits, weights=w_sqrt, transform_X='qt', inverse_target='log')
print(f"  xgb_qt: R² = {r2_score(y, oof):.4f}")
all_preds['xgb_qt'] = oof

# XGB with uniform weights
oof = get_oof(xgb_v13_fn, X_v7, y_log, splits, weights=None, inverse_target='log')
print(f"  xgb_noweight: R² = {r2_score(y, oof):.4f}")
all_preds['xgb_noweight'] = oof

# XGB with sqrt target
def xgb_sqrt_fn(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m
oof = get_oof(xgb_sqrt_fn, X_v7, y_sqrt, splits, weights=w_sqrt, inverse_target='sqrt')
print(f"  xgb_sqrt_target: R² = {r2_score(y, oof):.4f}")
all_preds['xgb_sqrt_target'] = oof
sys.stdout.flush()

# ============================================================
# GROUP B: LGB variants
# ============================================================
print("\n--- Group B: LGB variants ---")

def lgb_v14(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof = get_oof(lgb_v14, X_v7, y_log, splits, weights=w_sqrt, inverse_target='log')
print(f"  lgb_v14: R² = {r2_score(y, oof):.4f}")
all_preds['lgb_v14'] = oof

oof = get_oof(lgb_v14, X_v7, y_log, splits, weights=w_sqrt, transform_X='qt', inverse_target='log')
print(f"  lgb_qt: R² = {r2_score(y, oof):.4f}")
all_preds['lgb_qt'] = oof

# LGB with different params
def lgb_shallow(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(n_estimators=1000, max_depth=3, learning_rate=0.01,
        subsample=0.5, colsample_bytree=0.7, min_child_samples=50,
        num_leaves=8, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m
oof = get_oof(lgb_shallow, X_v7, y_log, splits, weights=w_sqrt, inverse_target='log')
print(f"  lgb_shallow: R² = {r2_score(y, oof):.4f}")
all_preds['lgb_shallow'] = oof
sys.stdout.flush()

# ============================================================
# GROUP C: Linear models
# ============================================================
print("\n--- Group C: Linear models ---")

for alpha, l1 in [(0.1, 0.9), (0.01, 0.5), (0.001, 0.1)]:
    name = f'enet_a{alpha}_l{l1}'
    def fn(Xtr, ytr, wtr, a=alpha, r=l1):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        m = ElasticNet(alpha=a, l1_ratio=r, max_iter=10000, random_state=42)
        m.fit(Xtr_s, ytr)
        class W:
            def predict(self, X): return m.predict(sc.transform(X))
        return W()
    oof = get_oof(fn, X_v7, y_log, splits, inverse_target='log')
    r2 = r2_score(y, oof)
    print(f"  {name}: R² = {r2:.4f}")
    all_preds[name] = oof

def ridge_fn(Xtr, ytr, wtr):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    m = Ridge(alpha=1.0, random_state=42)
    m.fit(Xtr_s, ytr)
    class W:
        def predict(self, X): return m.predict(sc.transform(X))
    return W()
oof = get_oof(ridge_fn, X_v7, y_log, splits, inverse_target='log')
print(f"  ridge: R² = {r2_score(y, oof):.4f}")
all_preds['ridge'] = oof
sys.stdout.flush()

# ============================================================
# GROUP D: GBR (sklearn)
# ============================================================
print("\n--- Group D: GBR ---")
def gbr_fn(Xtr, ytr, wtr):
    m = GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.5, min_samples_leaf=30, random_state=42)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m
oof = get_oof(gbr_fn, X_v7, y_log, splits, weights=w_sqrt, inverse_target='log')
print(f"  gbr: R² = {r2_score(y, oof):.4f}")
all_preds['gbr'] = oof
sys.stdout.flush()

# ============================================================
# DIRICHLET BLEND
# ============================================================
print("\n" + "="*60)
print("  DIRICHLET BLEND")
print("="*60)

def dirichlet_blend(pred_dict, y_true, n_trials=3000000):
    names = list(pred_dict.keys())
    preds = np.array([pred_dict[k] for k in names])
    k = len(names)
    best_r2, best_w = -999, None
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(k))
        blend = (w[:, None] * preds).sum(axis=0)
        r2 = r2_score(y_true, blend)
        if r2 > best_r2:
            best_r2, best_w = r2, w.copy()
    return best_r2, {names[i]: best_w[i] for i in range(k)}

print(f"\n  All {len(all_preds)} models:")
for name, pred in sorted(all_preds.items(), key=lambda x: r2_score(y, x[1]), reverse=True):
    print(f"    {name}: R² = {r2_score(y, pred):.4f}")
sys.stdout.flush()

best_r2, best_w = dirichlet_blend(all_preds, y, n_trials=3000000)
print(f"\n  Full blend: R² = {best_r2:.4f}")
print(f"  Weights: {json.dumps({k:round(v,3) for k,v in sorted(best_w.items(), key=lambda x:-x[1]) if v>0.01})}")

# Top-6
top6 = sorted(all_preds.keys(), key=lambda k: r2_score(y, all_preds[k]), reverse=True)[:6]
top6_r2, top6_w = dirichlet_blend({k: all_preds[k] for k in top6}, y, n_trials=3000000)
print(f"  Top-6 blend: R² = {top6_r2:.4f}")
print(f"  Weights: {json.dumps({k:round(v,3) for k,v in sorted(top6_w.items(), key=lambda x:-x[1]) if v>0.01})}")

best_a = max(best_r2, top6_r2)

# ============================================================
# POST-HOC: Error correlation between top models
# ============================================================
print("\n--- Error correlation matrix (top 6) ---")
top6_names = sorted(all_preds.keys(), key=lambda k: r2_score(y, all_preds[k]), reverse=True)[:6]
residuals_dict = {k: y - all_preds[k] for k in top6_names}
print(f"  {'':20s}", end='')
for k in top6_names:
    print(f" {k[:8]:>8s}", end='')
print()
for k1 in top6_names:
    print(f"  {k1:20s}", end='')
    for k2 in top6_names:
        corr = np.corrcoef(residuals_dict[k1], residuals_dict[k2])[0, 1]
        print(f" {corr:8.3f}", end='')
    print()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_single = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_r2 = r2_score(y, all_preds[best_single])

print("\n" + "="*60)
print(f"  V28 SUMMARY")
print("="*60)
print(f"  Best single: {best_single} R² = {best_single_r2:.4f}")
print(f"  Best blend:  R² = {best_a:.4f}")
print(f"  Previous best: 0.5467")
print(f"  Delta: {best_a - 0.5467:+.4f}")
print(f"  Elapsed: {elapsed:.1f}s")

results = {
    'best_r2_a': best_a,
    'best_single': {'name': best_single, 'r2': best_single_r2},
    'all_scores': {k: float(r2_score(y, v)) for k, v in all_preds.items()},
    'blend_weights': {k: float(v) for k, v in best_w.items()},
    'elapsed': elapsed,
}
with open('v28_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved v28_results.json")
