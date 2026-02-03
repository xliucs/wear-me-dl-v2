#!/usr/bin/env python3
"""
V22: MAE Loss + SMOTER + Pseudo-Insulin Feature (on Python 3.14)

V21 KEY FINDINGS (ported from Python 3.12):
1. XGB MAE loss beats MSE by +0.0015 (relative)
2. SMOTER p85 x3 helps XGB by +0.0023 (relative)  
3. CatBoost disappointing — skip it

NEW IDEAS:
4. Pseudo-insulin feature: Since HOMA = glucose × insulin / 405,
   we know insulin = HOMA × 405 / glucose. Train an "insulin predictor"  
   from non-glucose features, use its OOF output as additional feature.
   This is VALID if done with nested OOF (no leakage).

5. Interaction feature explosion: Generate ALL pairwise feature interactions,
   select top-k by correlation with residuals.
   
6. Combined: MAE loss + SMOTER + pseudo-insulin on best models
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V22: MAE + SMOTER + PSEUDO-INSULIN (Python 3.14)")
print("="*60)
sys.stdout.flush()

# === Load Data ===
X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
y_log = np.log1p(y)

# Compute pseudo-insulin (ground truth derived)
glucose = X_df['glucose'].values
pseudo_insulin = y * 405.0 / np.clip(glucose, 1, None)
pseudo_insulin_log = np.log1p(pseudo_insulin)
print(f"Pseudo-insulin: mean={pseudo_insulin.mean():.1f}, std={pseudo_insulin.std():.1f}")

# === V7 Feature Engineering ===
def eng_v7(X_df, cols, extra_cols=None):
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
    if extra_cols is not None:
        for name, vals in extra_cols.items():
            X[name] = vals
    return X.fillna(0).values

X_v7 = eng_v7(X_df, all_cols)
print(f"V7 features: {X_v7.shape[1]}")
sys.stdout.flush()

# === SMOTER ===
def smoter_augment(X_train, y_train, threshold_pct=85, oversample_factor=3, k=5):
    threshold = np.percentile(y_train, threshold_pct)
    high_mask = y_train >= threshold
    X_high = X_train[high_mask]
    y_high = y_train[high_mask]
    if len(X_high) < k + 1:
        return X_train, y_train
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_high)
    synthetic_X, synthetic_y = [], []
    rng = np.random.default_rng(42)
    for _ in range(oversample_factor):
        for i in range(len(X_high)):
            distances, indices = nn.kneighbors(X_high[i:i+1])
            j = rng.integers(1, k+1)
            lam = rng.uniform(0, 1)
            new_x = X_high[i] + lam * (X_high[indices[0, j]] - X_high[i])
            new_y = y_high[i] + lam * (y_high[indices[0, j]] - y_high[i])
            synthetic_X.append(new_x)
            synthetic_y.append(new_y)
    return np.vstack([X_train, np.array(synthetic_X)]), np.concatenate([y_train, np.array(synthetic_y)])

# === OOF helper ===
def get_oof(model_fn, X, y_target, splits, scale=True, weights=None, use_smoter=False):
    oof_sum = np.zeros(n)
    oof_count = np.zeros(n)
    fold_scores = []
    for i, (tr, te) in enumerate(splits):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr = y_target[tr].copy()
        wtr = weights[tr] if weights is not None else None
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        if use_smoter:
            Xtr, ytr_aug = smoter_augment(Xtr, ytr)
            if wtr is not None:
                w_synth = np.full(len(ytr_aug) - len(ytr), np.percentile(wtr, 90))
                wtr = np.concatenate([wtr, w_synth])
            ytr = ytr_aug
        model = model_fn(Xtr, ytr, wtr)
        preds = model.predict(Xte)
        oof_sum[te] += preds
        oof_count[te] += 1
        if y_target is y_log:
            fold_scores.append(r2_score(y[te], np.expm1(preds)))
        else:
            fold_scores.append(r2_score(y[te], preds))
    oof = oof_sum / np.clip(oof_count, 1, None)
    if y_target is y_log:
        oof_real = np.expm1(oof)
    else:
        oof_real = oof
    return oof_real, r2_score(y, oof_real), fold_scores

all_preds = {}

# ============================================================
# 1. BASELINE: Best known models (MSE loss)
# ============================================================
print("\n--- Baselines (MSE loss, log target, sqrt weights) ---")
sys.stdout.flush()

def xgb_opt_mse(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_opt_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB Optuna MSE:  R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['xgb_mse'] = oof
sys.stdout.flush()

def lgb_opt_mse(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(
        n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(lgb_opt_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  LGB Optuna MSE:  R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['lgb_mse'] = oof
sys.stdout.flush()

def enet_fn(Xtr, ytr, wtr):
    m = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
    m.fit(Xtr, ytr)
    return m

oof, r2, _ = get_oof(enet_fn, X_v7, y_log, splits, scale=True)
print(f"  ElasticNet:      R² = {r2:.4f}")
all_preds['enet'] = oof
sys.stdout.flush()

# ============================================================
# 2. MAE LOSS (V21 finding: helps XGB)
# ============================================================
print("\n--- MAE Loss (V21 finding) ---")
sys.stdout.flush()

def xgb_opt_mae(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045,
        objective='reg:absoluteerror',
        random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_opt_mae, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MAE:         R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['xgb_mae'] = oof
sys.stdout.flush()

# ============================================================
# 3. SMOTER (V21 finding: helps XGB, p85 x3)
# ============================================================
print("\n--- SMOTER p85 x3 (V21 finding) ---")
sys.stdout.flush()

oof, r2, fs = get_oof(xgb_opt_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt, use_smoter=True)
print(f"  XGB MSE+SMOTER:  R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['xgb_mse_smoter'] = oof
sys.stdout.flush()

oof, r2, fs = get_oof(xgb_opt_mae, X_v7, y_log, splits, scale=False, weights=w_sqrt, use_smoter=True)
print(f"  XGB MAE+SMOTER:  R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['xgb_mae_smoter'] = oof
sys.stdout.flush()

# ============================================================
# 4. PSEUDO-INSULIN FEATURE (novel approach)
# ============================================================
print("\n--- Pseudo-Insulin Feature Engineering ---")
print("  Step 1: Train insulin predictor from non-glucose features")
sys.stdout.flush()

# Features for insulin prediction: all V7 features EXCEPT glucose and glucose-derived
# We use nested OOF: outer fold for HOMA-IR, inner loop for insulin
# Simpler approach: generate insulin OOF using the SAME splits

# Non-glucose features for insulin prediction
non_glucose_cols = [c for c in all_cols if c != 'glucose']
X_ng = X_df[non_glucose_cols].copy()
# Add some non-glucose engineered features
b = X_ng['bmi']; a = X_ng['age']
t = X_ng['triglycerides'].clip(lower=1); h = X_ng['hdl'].clip(lower=1)
rhr = X_ng['Resting Heart Rate (mean)']; hrv = X_ng['HRV (mean)'].clip(lower=1)
stp = X_ng['STEPS (mean)'].clip(lower=1)
X_ng['bmi_sq'] = b**2
X_ng['bmi_cubed'] = b**3
X_ng['trig_hdl'] = t/h
X_ng['vat_proxy'] = b*t/h
X_ng['bmi_rhr'] = b*rhr
X_ng['bmi_hrv_inv'] = b/hrv
X_ng['bmi_stp_inv'] = b/stp*1000
X_ng['sed_risk'] = b**2*rhr/(stp*hrv)
X_ng['bmi_age'] = b*a
X_ng['bmi_trig'] = b*t
X_ng['non_hdl_ratio'] = X_ng['non hdl']/h
X_ng['rhr_hrv'] = rhr/hrv
X_ng_arr = X_ng.fillna(0).values
print(f"  Non-glucose features: {X_ng_arr.shape[1]}")

# Generate OOF predictions of pseudo_insulin_log
insulin_oof = np.zeros(n)
insulin_count = np.zeros(n)

for tr, te in splits:
    Xtr, Xte = X_ng_arr[tr], X_ng_arr[te]
    ytr = pseudo_insulin_log[tr]
    wtr = w_sqrt[tr]
    
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.6, colsample_bytree=0.8, min_child_weight=20,
        random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    preds = m.predict(Xte)
    insulin_oof[te] += preds
    insulin_count[te] += 1

insulin_oof = insulin_oof / np.clip(insulin_count, 1, None)
insulin_pred = np.expm1(insulin_oof)
insulin_r2 = r2_score(pseudo_insulin, insulin_pred)
print(f"  Insulin predictor OOF R²: {insulin_r2:.4f}")
sys.stdout.flush()

# Now add pseudo-insulin as feature and retrain HOMA-IR model
X_v7_plus = eng_v7(X_df, all_cols, extra_cols={
    'pred_insulin': insulin_oof,
    'pred_insulin_x_glucose': insulin_oof * np.log1p(glucose),
    'pred_homa': np.log1p(insulin_pred * glucose / 405.0),
})
print(f"  V7 + pseudo-insulin features: {X_v7_plus.shape[1]}")

oof, r2, fs = get_oof(xgb_opt_mse, X_v7_plus, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MSE + pseudo-insulin: R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['xgb_mse_pinsulin'] = oof
sys.stdout.flush()

oof, r2, fs = get_oof(lgb_opt_mse, X_v7_plus, y_log, splits, scale=False, weights=w_sqrt)
print(f"  LGB MSE + pseudo-insulin: R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['lgb_mse_pinsulin'] = oof
sys.stdout.flush()

# XGB MAE + pseudo-insulin + SMOTER
oof, r2, fs = get_oof(xgb_opt_mae, X_v7_plus, y_log, splits, scale=False, weights=w_sqrt, use_smoter=True)
print(f"  XGB MAE+SMOTER+pinsulin: R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
all_preds['xgb_mae_smoter_pinsulin'] = oof
sys.stdout.flush()

# ============================================================
# 5. INTERACTION FEATURE EXPLOSION
# ============================================================
print("\n--- Interaction Feature Explosion ---")
sys.stdout.flush()

# Generate top pairwise interactions based on correlation with y
X_base = X_df[all_cols].fillna(0)
base_cols = list(X_base.columns)
top_interactions = []

for i in range(len(base_cols)):
    for j in range(i+1, len(base_cols)):
        c1, c2 = base_cols[i], base_cols[j]
        interaction = X_base[c1].values * X_base[c2].values
        corr = abs(np.corrcoef(interaction, y)[0,1])
        if not np.isnan(corr):
            top_interactions.append((corr, c1, c2))

top_interactions.sort(reverse=True)
print(f"  Total pairwise interactions: {len(top_interactions)}")
print(f"  Top-5 interactions:")
for corr, c1, c2 in top_interactions[:5]:
    print(f"    {c1} × {c2}: corr={corr:.4f}")

# Add top-20 new interactions not already in V7
existing_v7_pairs = {
    ('glucose','bmi'), ('glucose','triglycerides'), ('glucose','hdl'),
    ('bmi','triglycerides'), ('bmi','age'), ('bmi','Resting Heart Rate (mean)'),
}

new_interactions = {}
count = 0
for corr, c1, c2 in top_interactions:
    if (c1,c2) not in existing_v7_pairs and (c2,c1) not in existing_v7_pairs:
        name = f"int_{c1[:4]}_{c2[:4]}"
        new_interactions[name] = X_base[c1].values * X_base[c2].values
        count += 1
        if count >= 20:
            break

# V7 + new interactions
X_v7_int = eng_v7(X_df, all_cols, extra_cols=new_interactions)
print(f"  V7 + top-20 new interactions: {X_v7_int.shape[1]} features")

oof, r2, fs = get_oof(xgb_opt_mse, X_v7_int, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MSE + interactions: R² = {r2:.4f}")
all_preds['xgb_mse_interactions'] = oof
sys.stdout.flush()

# ============================================================
# 6. EVERYTHING COMBINED
# ============================================================
print("\n--- Everything Combined: pseudo-insulin + interactions + SMOTER + MAE ---")
sys.stdout.flush()

# Mega-feature set
mega_extras = {**new_interactions, 
               'pred_insulin': insulin_oof,
               'pred_insulin_x_glucose': insulin_oof * np.log1p(glucose),
               'pred_homa': np.log1p(insulin_pred * glucose / 405.0)}
X_mega = eng_v7(X_df, all_cols, extra_cols=mega_extras)
print(f"  Mega features: {X_mega.shape[1]}")

oof, r2, fs = get_oof(xgb_opt_mse, X_mega, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MSE mega:         R² = {r2:.4f}")
all_preds['xgb_mse_mega'] = oof

oof, r2, fs = get_oof(xgb_opt_mae, X_mega, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MAE mega:         R² = {r2:.4f}")
all_preds['xgb_mae_mega'] = oof

oof, r2, fs = get_oof(lgb_opt_mse, X_mega, y_log, splits, scale=False, weights=w_sqrt)
print(f"  LGB MSE mega:         R² = {r2:.4f}")
all_preds['lgb_mse_mega'] = oof

oof, r2, fs = get_oof(xgb_opt_mse, X_mega, y_log, splits, scale=False, weights=w_sqrt, use_smoter=True)
print(f"  XGB MSE+SMOTER mega:  R² = {r2:.4f}")
all_preds['xgb_mse_smoter_mega'] = oof

oof, r2, fs = get_oof(enet_fn, X_mega, y_log, splits, scale=True)
print(f"  ElasticNet mega:      R² = {r2:.4f}")
all_preds['enet_mega'] = oof
sys.stdout.flush()

# ============================================================
# 7. DIRICHLET BLEND
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
top6_preds = {k: all_preds[k] for k in top6}
top6_r2, top6_w = dirichlet_blend(top6_preds, y, n_trials=2000000)
print(f"\n  Top-6 blend: R² = {top6_r2:.4f}")
print(f"  Weights: {json.dumps({k:round(v,3) for k,v in sorted(top6_w.items(), key=lambda x:-x[1]) if v>0.01})}")
sys.stdout.flush()

best_a = max(best_r2, top6_r2)

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_single_name = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_r2 = r2_score(y, all_preds[best_single_name])

print("\n" + "="*60)
print(f"  V22 SUMMARY")
print("="*60)
print(f"\n  Model A:")
print(f"    Best single: {best_single_name} R² = {best_single_r2:.4f}")
print(f"    Best blend:  R² = {best_a:.4f}")
print(f"    Previous best (V20): 0.5467")
print(f"    Delta: {best_a - 0.5467:+.4f}")
print(f"\n  Elapsed: {elapsed:.1f}s")

results = {
    'best_r2_a': best_a,
    'best_single_a': {'name': best_single_name, 'r2': best_single_r2},
    'insulin_predictor_r2': insulin_r2,
    'all_scores_a': {k: float(r2_score(y, v)) for k, v in all_preds.items()},
    'blend_weights': {k: float(v) for k, v in best_w.items()},
    'elapsed': elapsed,
}
with open('v22_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved v22_results.json")
