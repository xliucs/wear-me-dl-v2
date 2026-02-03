#!/usr/bin/env python3
"""
V24: Target Decomposition — Predict insulin, not HOMA_IR

INSIGHT: HOMA_IR = glucose × insulin / 405
- We KNOW glucose (it's a feature)
- The hard part is predicting insulin
- Currently we predict log1p(HOMA_IR) which mixes easy (glucose) + hard (insulin)
- What if we predict log(insulin) directly? Then HOMA = glucose × exp(pred) / 405

This changes the learning problem:
- Target: log(insulin) = log(HOMA × 405 / glucose)
- The model focuses 100% on predicting insulin from {BMI, trig, HDL, age, sex, wearables}
- Glucose is NOT used as a feature (it's in the reconstruction formula)
- Reconstruction: HOMA_pred = glucose × exp(insulin_pred) / 405

Also try:
- Target: log(HOMA/glucose) = log(insulin/405) — equivalent but numerically different
- Blend decomposed predictions with standard predictions
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V24: TARGET DECOMPOSITION — PREDICT INSULIN")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
y_log = np.log1p(y)

glucose = X_df['glucose'].values
# Derived targets
insulin = y * 405.0 / np.clip(glucose, 1, None)  # true fasting insulin
log_insulin = np.log1p(insulin)
log_homa_over_glucose = np.log1p(y / np.clip(glucose, 1, None))

print(f"True insulin: mean={insulin.mean():.1f}, std={insulin.std():.1f}, "
      f"median={np.median(insulin):.1f}")
print(f"Correlation insulin vs HOMA: {np.corrcoef(insulin, y)[0,1]:.4f}")
print(f"Correlation glucose vs HOMA: {np.corrcoef(glucose, y)[0,1]:.4f}")
sys.stdout.flush()

# === Feature engineering ===
# Standard V7 for standard models
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

# Non-glucose features for insulin prediction
# Remove glucose and ALL glucose-derived features
non_glucose_raw = [c for c in all_cols if c != 'glucose']
X_ng = X_df[non_glucose_raw].copy()
b = X_ng['bmi']; a = X_ng['age']
t = X_ng['triglycerides'].clip(lower=1); h = X_ng['hdl'].clip(lower=1)
rhr = X_ng['Resting Heart Rate (mean)']; hrv = X_ng['HRV (mean)'].clip(lower=1)
stp = X_ng['STEPS (mean)'].clip(lower=1); slp = X_ng['SLEEP Duration (mean)']
# Only non-glucose interactions
for nm,v in [('bmi_sq',b**2),('bmi_cubed',b**3),('bmi_age',b*a),('age_sq',a**2),
    ('trig_hdl',t/h),('trig_hdl_log',np.log1p(t/h)),('vat_proxy',b*t/h),
    ('bmi_rhr',b*rhr),('bmi_hrv_inv',b/hrv),('bmi_stp_inv',b/stp*1000),
    ('rhr_hrv',rhr/hrv),('cardio_fitness',hrv*stp/rhr),
    ('met_load',b*rhr/stp*1000),('sed_risk',b**2*rhr/(stp*hrv)),
    ('non_hdl_ratio',X_ng['non hdl']/h),('bmi_trig',b*t/1000),
    ('bmi_sex',b*X_ng['sex_num']),
    ('rhr_skew',(rhr-X_ng['Resting Heart Rate (median)'])/X_ng['Resting Heart Rate (std)'].clip(lower=0.01)),
    ('hrv_skew',(hrv-X_ng['HRV (median)'])/X_ng['HRV (std)'].clip(lower=0.01)),
    ('stp_skew',(stp-X_ng['STEPS (median)'])/X_ng['STEPS (std)'].clip(lower=0.01)),
    ('slp_skew',(slp-X_ng['SLEEP Duration (median)'])/X_ng['SLEEP Duration (std)'].clip(lower=0.01)),
    ('azm_skew',(X_ng['AZM Weekly (mean)']-X_ng['AZM Weekly (median)'])/X_ng['AZM Weekly (std)'].clip(lower=0.01)),
    ('rhr_cv',X_ng['Resting Heart Rate (std)'].clip(lower=0.01)/rhr),
    ('hrv_cv',X_ng['HRV (std)'].clip(lower=0.01)/hrv),
    ('stp_cv',X_ng['STEPS (std)'].clip(lower=0.01)/stp),
    ('slp_cv',X_ng['SLEEP Duration (std)'].clip(lower=0.01)/slp),
    ('azm_cv',X_ng['AZM Weekly (std)'].clip(lower=0.01)/X_ng['AZM Weekly (mean)'].clip(lower=0.01)),
]:
    X_ng[nm] = v
X_ng_arr = X_ng.fillna(0).values
print(f"\nV7 features (full): {X_v7.shape[1]}")
print(f"Non-glucose features: {X_ng_arr.shape[1]}")
sys.stdout.flush()

# === OOF helper ===
def get_oof(model_fn, X, y_target, splits, weights=None, scale=False):
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr = y_target[tr]
        wtr = weights[tr] if weights is not None else None
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        model = model_fn(Xtr, ytr, wtr)
        oof_sum[te] += model.predict(Xte)
        oof_count[te] += 1
    return oof_sum / np.clip(oof_count, 1, None)

all_preds = {}

# ============================================================
# 1. BASELINE: Standard log(HOMA) prediction
# ============================================================
print("\n--- Baseline: predict log1p(HOMA_IR) ---")
sys.stdout.flush()

def xgb_opt(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof_log = get_oof(xgb_opt, X_v7, y_log, splits, weights=w_sqrt)
oof_std = np.expm1(oof_log)
r2_std = r2_score(y, oof_std)
print(f"  XGB standard (log1p HOMA, all feats): R² = {r2_std:.4f}")
all_preds['xgb_standard'] = oof_std

def lgb_opt(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof_log = get_oof(lgb_opt, X_v7, y_log, splits, weights=w_sqrt)
oof_lgb_std = np.expm1(oof_log)
r2_lgb = r2_score(y, oof_lgb_std)
print(f"  LGB standard (log1p HOMA, all feats): R² = {r2_lgb:.4f}")
all_preds['lgb_standard'] = oof_lgb_std
sys.stdout.flush()

# ============================================================
# 2. DECOMPOSED: predict log(insulin) from non-glucose features
# ============================================================
print("\n--- Decomposed: predict log1p(insulin) from non-glucose features ---")
sys.stdout.flush()

# Weight by sqrt(insulin) instead of sqrt(HOMA)
w_insulin = np.sqrt(insulin) / np.sqrt(insulin).mean()

oof_ins = get_oof(xgb_opt, X_ng_arr, log_insulin, splits, weights=w_insulin)
ins_pred = np.expm1(oof_ins)
homa_recon = ins_pred * glucose / 405.0
r2_decomp = r2_score(y, homa_recon)
r2_insulin = r2_score(insulin, ins_pred)
print(f"  XGB insulin predictor R²: {r2_insulin:.4f}")
print(f"  Reconstructed HOMA R²: {r2_decomp:.4f}")
all_preds['xgb_decomposed'] = homa_recon
sys.stdout.flush()

# LGB decomposed
oof_ins_lgb = get_oof(lgb_opt, X_ng_arr, log_insulin, splits, weights=w_insulin)
ins_pred_lgb = np.expm1(oof_ins_lgb)
homa_recon_lgb = ins_pred_lgb * glucose / 405.0
r2_decomp_lgb = r2_score(y, homa_recon_lgb)
print(f"  LGB insulin predictor R²: {r2_score(insulin, ins_pred_lgb):.4f}")
print(f"  Reconstructed HOMA R²: {r2_decomp_lgb:.4f}")
all_preds['lgb_decomposed'] = homa_recon_lgb
sys.stdout.flush()

# ============================================================
# 3. DECOMPOSED with ALL features (including glucose)
# ============================================================
print("\n--- Decomposed with all features (insulin target, full feature set) ---")
sys.stdout.flush()

oof_ins_full = get_oof(xgb_opt, X_v7, log_insulin, splits, weights=w_insulin)
ins_pred_full = np.expm1(oof_ins_full)
homa_recon_full = ins_pred_full * glucose / 405.0
r2_decomp_full = r2_score(y, homa_recon_full)
print(f"  XGB insulin (all feats) R²: {r2_score(insulin, ins_pred_full):.4f}")
print(f"  Reconstructed HOMA R²: {r2_decomp_full:.4f}")
all_preds['xgb_decomposed_full'] = homa_recon_full
sys.stdout.flush()

# LGB
oof_ins_lgb_full = get_oof(lgb_opt, X_v7, log_insulin, splits, weights=w_insulin)
ins_pred_lgb_full = np.expm1(oof_ins_lgb_full)
homa_recon_lgb_full = ins_pred_lgb_full * glucose / 405.0
r2_decomp_lgb_full = r2_score(y, homa_recon_lgb_full)
print(f"  LGB insulin (all feats) R²: {r2_score(insulin, ins_pred_lgb_full):.4f}")
print(f"  Reconstructed HOMA R²: {r2_decomp_lgb_full:.4f}")
all_preds['lgb_decomposed_full'] = homa_recon_lgb_full
sys.stdout.flush()

# ============================================================
# 4. HYBRID: predict log(HOMA/glucose) target
# ============================================================
print("\n--- Hybrid: predict log1p(HOMA/glucose) ---")
sys.stdout.flush()

oof_hog = get_oof(xgb_opt, X_v7, log_homa_over_glucose, splits, weights=w_sqrt)
homa_hybrid = np.expm1(oof_hog) * glucose
r2_hybrid = r2_score(y, homa_hybrid)
print(f"  XGB log(HOMA/glucose) + reconstruct: R² = {r2_hybrid:.4f}")
all_preds['xgb_hybrid'] = homa_hybrid

oof_hog_lgb = get_oof(lgb_opt, X_v7, log_homa_over_glucose, splits, weights=w_sqrt)
homa_hybrid_lgb = np.expm1(oof_hog_lgb) * glucose
r2_hybrid_lgb = r2_score(y, homa_hybrid_lgb)
print(f"  LGB log(HOMA/glucose) + reconstruct: R² = {r2_hybrid_lgb:.4f}")
all_preds['lgb_hybrid'] = homa_hybrid_lgb
sys.stdout.flush()

# ============================================================
# 5. ElasticNet for blending diversity
# ============================================================
print("\n--- ElasticNet ---")
def enet_fn(Xtr, ytr, wtr):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    m = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
    m.fit(Xtr_s, ytr)
    class W:
        def predict(self, X): return m.predict(sc.transform(X))
    return W()

oof_enet = get_oof(enet_fn, X_v7, y_log, splits)
all_preds['enet'] = np.expm1(oof_enet)
print(f"  ElasticNet: R² = {r2_score(y, all_preds['enet']):.4f}")
sys.stdout.flush()

# ============================================================
# 6. DIRICHLET BLEND
# ============================================================
print("\n" + "="*60)
print("  DIRICHLET BLEND")
print("="*60)

def dirichlet_blend(pred_dict, y_true, n_trials=2000000):
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

best_r2, best_w = dirichlet_blend(all_preds, y, n_trials=2000000)
print(f"\n  Full blend: R² = {best_r2:.4f}")
print(f"  Top weights: {json.dumps({k:round(v,3) for k,v in sorted(best_w.items(), key=lambda x:-x[1]) if v>0.01})}")
sys.stdout.flush()

best_a = best_r2

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_single = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_r2 = r2_score(y, all_preds[best_single])

print("\n" + "="*60)
print(f"  V24 SUMMARY")
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
    'insulin_decomp_r2': {
        'xgb_insulin_r2': float(r2_insulin),
        'homa_reconstructed_r2': float(r2_decomp),
    },
    'blend_weights': {k: float(v) for k, v in best_w.items()},
    'elapsed': elapsed,
}
with open('v24_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved v24_results.json")
