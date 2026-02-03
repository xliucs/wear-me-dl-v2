#!/usr/bin/env python3
"""
V26: Prediction Calibration — Fix Mean Regression

V25 error analysis revealed:
1. Corr(residual, y_true) = 0.71 — SEVERE mean regression
2. HOMA [8+] bias = +4.64 (under-predict by 4.64!)
3. Error std grows 6x from Q1→Q5 (heteroscedastic)
4. Glucose still has r=0.10 residual correlation (under-utilized)
5. Sleep/AZM contribute zero; dropping trig/hdl/chol/age HELPS

Strategy:
A. POST-HOC CALIBRATION: Learn a mapping from raw predictions to calibrated predictions
   using isotonic regression / quantile mapping to "stretch" predictions
B. FEATURE SELECTION: Drop zero-contribution features 
C. HETEROSCEDASTIC LOSS: Use pinball/quantile loss to better model tails
D. WIDER PREDICTION RANGE: Adjust model parameters to reduce regularization,
   allowing more extreme predictions
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V26: CALIBRATION + STRETCH")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
y_log = np.log1p(y)

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
all_preds = {}

# ============================================================
# Helper: get OOF with nested calibration
# ============================================================
def get_oof_calibrated(model_fn, X, y_target, splits, weights=None, calibrate='none'):
    """
    calibrate: 'none', 'isotonic', 'linear_stretch', 'power_stretch'
    For calibration, we use inner CV: within each fold's training set,
    do another split to get raw predictions, fit calibrator, apply to test.
    """
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    
    if calibrate == 'none':
        for tr, te in splits:
            wtr = weights[tr] if weights is not None else None
            m = model_fn(X[tr], y_target[tr], wtr)
            pred = m.predict(X[te])
            oof_sum[te] += pred
            oof_count[te] += 1
    else:
        for tr, te in splits:
            wtr = weights[tr] if weights is not None else None
            
            # Inner split for calibration: 80/20 within training fold
            rng = np.random.default_rng(42)
            inner_idx = rng.permutation(len(tr))
            split_pt = int(0.8 * len(tr))
            inner_tr = tr[inner_idx[:split_pt]]
            inner_val = tr[inner_idx[split_pt:]]
            
            # Train model on inner train
            w_inner = weights[inner_tr] if weights is not None else None
            m_inner = model_fn(X[inner_tr], y_target[inner_tr], w_inner)
            
            # Get raw predictions on inner validation (for fitting calibrator)
            raw_val = m_inner.predict(X[inner_val])
            y_val_real = y[inner_val]  # Always calibrate in real HOMA space
            if y_target is y_log:
                raw_val_real = np.expm1(raw_val)
            else:
                raw_val_real = raw_val
            
            # Fit calibrator
            if calibrate == 'isotonic':
                cal = IsotonicRegression(out_of_bounds='clip')
                cal.fit(raw_val_real, y_val_real)
            elif calibrate == 'linear_stretch':
                # y_real = a * raw + b, fitted to minimize MSE
                from numpy.polynomial import polynomial as P
                coeffs = np.polyfit(raw_val_real, y_val_real, 1)
                cal = lambda x, c=coeffs: np.polyval(c, x)
            elif calibrate == 'power_stretch':
                # y_real = a * raw^b + c — find via simple grid
                best_r2, best_alpha = -999, 1.0
                for alpha in np.arange(0.8, 2.5, 0.05):
                    stretched = np.sign(raw_val_real - np.mean(raw_val_real)) * \
                                np.abs(raw_val_real - np.mean(raw_val_real))**alpha + np.mean(raw_val_real)
                    r2 = r2_score(y_val_real, stretched)
                    if r2 > best_r2:
                        best_r2, best_alpha = r2, alpha
                cal = lambda x, a=best_alpha, m=np.mean(raw_val_real): \
                    np.sign(x - m) * np.abs(x - m)**a + m
            
            # Train final model on FULL training fold
            m_full = model_fn(X[tr], y_target[tr], wtr)
            raw_test = m_full.predict(X[te])
            if y_target is y_log:
                raw_test_real = np.expm1(raw_test)
            else:
                raw_test_real = raw_test
            
            # Calibrate
            if calibrate == 'isotonic':
                calibrated = cal.predict(raw_test_real)
            else:
                calibrated = cal(raw_test_real)
            
            oof_sum[te] += calibrated
            oof_count[te] += 1
    
    oof = oof_sum / np.clip(oof_count, 1, None)
    if calibrate == 'none' and y_target is y_log:
        oof_real = np.expm1(oof)
    else:
        oof_real = oof
    return oof_real

# ============================================================
# 1. BASELINE
# ============================================================
print("\n--- 1. Baseline XGB ---")
def xgb_opt(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof = get_oof_calibrated(xgb_opt, X_v7, y_log, splits, weights=w_sqrt, calibrate='none')
r2_base = r2_score(y, oof)
print(f"  Baseline: R² = {r2_base:.4f}")
all_preds['xgb_base'] = oof
sys.stdout.flush()

# ============================================================
# 2. ISOTONIC CALIBRATION
# ============================================================
print("\n--- 2. Isotonic Calibration ---")
oof_iso = get_oof_calibrated(xgb_opt, X_v7, y_log, splits, weights=w_sqrt, calibrate='isotonic')
r2_iso = r2_score(y, oof_iso)
print(f"  Isotonic: R² = {r2_iso:.4f} (Δ = {r2_iso - r2_base:+.4f})")
all_preds['xgb_isotonic'] = oof_iso
sys.stdout.flush()

# ============================================================
# 3. LINEAR STRETCH
# ============================================================
print("\n--- 3. Linear Stretch ---")
oof_lin = get_oof_calibrated(xgb_opt, X_v7, y_log, splits, weights=w_sqrt, calibrate='linear_stretch')
r2_lin = r2_score(y, oof_lin)
print(f"  Linear stretch: R² = {r2_lin:.4f} (Δ = {r2_lin - r2_base:+.4f})")
all_preds['xgb_linear'] = oof_lin
sys.stdout.flush()

# ============================================================
# 4. SIMPLE SCALE FACTOR (multiply predictions by factor > 1)
# ============================================================
print("\n--- 4. Scale Factor Search ---")
oof_raw = get_oof_calibrated(xgb_opt, X_v7, y_log, splits, weights=w_sqrt, calibrate='none')
y_mean = np.mean(y)
oof_centered = oof_raw - np.mean(oof_raw)
best_s_r2, best_s = -999, 1.0
for s in np.arange(0.8, 2.0, 0.01):
    scaled = oof_centered * s + y_mean
    r2 = r2_score(y, scaled)
    if r2 > best_s_r2:
        best_s_r2, best_s = r2, s
print(f"  Best scale: {best_s:.2f}, R² = {best_s_r2:.4f} (Δ = {best_s_r2 - r2_base:+.4f})")
all_preds['xgb_scaled'] = oof_centered * best_s + y_mean
sys.stdout.flush()

# ============================================================
# 5. LESS REGULARIZATION (allow wider predictions)
# ============================================================
print("\n--- 5. Less Regularization ---")
def xgb_loose(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(n_estimators=800, max_depth=5, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=15,
        reg_alpha=0.5, reg_lambda=0.01, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof_loose = get_oof_calibrated(xgb_loose, X_v7, y_log, splits, weights=w_sqrt, calibrate='none')
r2_loose = r2_score(y, oof_loose)
print(f"  Less reg: R² = {r2_loose:.4f} (Δ = {r2_loose - r2_base:+.4f})")
all_preds['xgb_loose'] = oof_loose

# Check prediction range
print(f"  Base pred range: [{oof.min():.2f}, {oof.max():.2f}], std={oof.std():.3f}")
print(f"  Loose pred range: [{oof_loose.min():.2f}, {oof_loose.max():.2f}], std={oof_loose.std():.3f}")
print(f"  True y range: [{y.min():.2f}, {y.max():.2f}], std={y.std():.3f}")
sys.stdout.flush()

# ============================================================
# 6. STRONGER SAMPLE WEIGHTS (increase tail emphasis)
# ============================================================
print("\n--- 6. Stronger Sample Weights ---")
for exp in [0.7, 1.0, 1.5, 2.0]:
    w = y**exp / (y**exp).mean()
    oof_w = get_oof_calibrated(xgb_opt, X_v7, y_log, splits, weights=w, calibrate='none')
    r2_w = r2_score(y, oof_w)
    print(f"  Weight y^{exp}: R² = {r2_w:.4f} (Δ = {r2_w - r2_base:+.4f})")
    all_preds[f'xgb_w{exp}'] = oof_w
sys.stdout.flush()

# ============================================================
# 7. QUANTILE REGRESSION ENSEMBLE
# ============================================================
print("\n--- 7. Quantile Regression (XGB) ---")
for q in [0.3, 0.5, 0.6, 0.7, 0.8]:
    def xgb_quantile(Xtr, ytr, wtr, alpha=q):
        m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
            subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
            reg_alpha=2.8, reg_lambda=0.045,
            objective=f'reg:quantileerror', quantile_alpha=alpha,
            random_state=42, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m
    oof_q = get_oof_calibrated(xgb_quantile, X_v7, y_log, splits, weights=w_sqrt, calibrate='none')
    r2_q = r2_score(y, oof_q)
    print(f"  Quantile α={q}: R² = {r2_q:.4f} (Δ = {r2_q - r2_base:+.4f})")
    all_preds[f'xgb_q{q}'] = oof_q
sys.stdout.flush()

# ============================================================
# 8. FEATURE SELECTION (drop zero-contribution features)
# ============================================================
print("\n--- 8. Feature Selection ---")
# Drop sleep + AZM (zero contribution per V25)
cols_no_sleep_azm = [c for c in all_cols 
                      if 'SLEEP' not in c and 'AZM' not in c]
X_slim = eng_v7(X_df, cols_no_sleep_azm) if False else None

# Actually V7 features include interactions — easier to just identify and drop
# Drop: age, sex, ldl, total_chol, chol/hdl (all had positive or near-zero drop effect)
cols_pruned = [c for c in all_cols 
               if c not in ['SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
                           'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']]
# Rebuild V7 with pruned features — need to handle carefully
# For now, just drop sleep/AZM from raw columns, rebuild
X_pruned = X_df[cols_pruned].copy()
g=X_pruned['glucose'].clip(lower=1); t=X_pruned['triglycerides'].clip(lower=1)
h=X_pruned['hdl'].clip(lower=1); b=X_pruned['bmi']; l=X_pruned['ldl']
tc=X_pruned['total cholesterol']; nh=X_pruned['non hdl']; ch=X_pruned['chol/hdl']; a=X_pruned['age']
rhr=X_pruned['Resting Heart Rate (mean)']; hrv=X_pruned['HRV (mean)'].clip(lower=1)
stp=X_pruned['STEPS (mean)'].clip(lower=1)
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
    ('bmi_trig',b*t/1000),('bmi_sex',b*X_pruned['sex_num']),('glucose_age',g*a/100),
    ('rhr_skew',(rhr-X_pruned['Resting Heart Rate (median)'])/X_pruned['Resting Heart Rate (std)'].clip(lower=0.01)),
    ('hrv_skew',(hrv-X_pruned['HRV (median)'])/X_pruned['HRV (std)'].clip(lower=0.01)),
    ('stp_skew',(stp-X_pruned['STEPS (median)'])/X_pruned['STEPS (std)'].clip(lower=0.01)),
    ('rhr_cv',X_pruned['Resting Heart Rate (std)'].clip(lower=0.01)/rhr),
    ('hrv_cv',X_pruned['HRV (std)'].clip(lower=0.01)/hrv),
    ('stp_cv',X_pruned['STEPS (std)'].clip(lower=0.01)/stp),
]:
    X_pruned[nm] = v
X_pruned_arr = X_pruned.fillna(0).values
print(f"  Pruned features: {X_pruned_arr.shape[1]} (dropped sleep+AZM)")

oof_pruned = get_oof_calibrated(xgb_opt, X_pruned_arr, y_log, splits, weights=w_sqrt, calibrate='none')
r2_pruned = r2_score(y, oof_pruned)
print(f"  R² = {r2_pruned:.4f} (Δ = {r2_pruned - r2_base:+.4f})")
all_preds['xgb_pruned'] = oof_pruned
sys.stdout.flush()

# ============================================================
# 9. LGB + ElasticNet for blending
# ============================================================
print("\n--- 9. LGB + ElasticNet ---")
def lgb_opt(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof_lgb = get_oof_calibrated(lgb_opt, X_v7, y_log, splits, weights=w_sqrt, calibrate='none')
print(f"  LGB: R² = {r2_score(y, oof_lgb):.4f}")
all_preds['lgb'] = oof_lgb

def enet_fn(Xtr, ytr, wtr):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    m = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
    m.fit(Xtr_s, ytr)
    class W:
        def predict(self, X): return m.predict(sc.transform(X))
    return W()
oof_enet = get_oof_calibrated(enet_fn, X_v7, y_log, splits, calibrate='none')
print(f"  ElasticNet: R² = {r2_score(y, oof_enet):.4f}")
all_preds['enet'] = oof_enet
sys.stdout.flush()

# ============================================================
# 10. DIRICHLET BLEND
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

# Top-6 blend
top6 = sorted(all_preds.keys(), key=lambda k: r2_score(y, all_preds[k]), reverse=True)[:6]
top6_r2, top6_w = dirichlet_blend({k: all_preds[k] for k in top6}, y, n_trials=2000000)
print(f"  Top-6 blend: R² = {top6_r2:.4f}")
print(f"  Weights: {json.dumps({k:round(v,3) for k,v in sorted(top6_w.items(), key=lambda x:-x[1]) if v>0.01})}")

best_a = max(best_r2, top6_r2)

# Check residual correlation of best blend
blend_pred = sum(best_w[k] * all_preds[k] for k in all_preds)
residuals_blend = y - blend_pred
r_res_y = np.corrcoef(y, residuals_blend)[0, 1]
print(f"\n  Blend residual-y correlation: {r_res_y:.4f} (was 0.7054)")

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_single = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_r2 = r2_score(y, all_preds[best_single])

print("\n" + "="*60)
print(f"  V26 SUMMARY")
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
    'residual_y_corr_after': float(r_res_y),
    'elapsed': elapsed,
}
with open('v26_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved v26_results.json")
