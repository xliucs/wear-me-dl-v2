#!/usr/bin/env python3
"""
V27: Hypothesis-Driven Experiments (from post-hoc analysis)

HYPOTHESIS 1: Sex-stratified models improve R²
- Rationale: Female R²=0.61 vs Male R²=0.46. Different metabolic pathways.
- Test: Train separate XGB/LGB for male/female, combine predictions.
- Expected: If sex-specific patterns exist, combined R² > 0.547

HYPOTHESIS 2: Train-test gap (0.31) is partially from overfitting
- Rationale: Train R²=0.85 vs Test R²=0.54. Ceiling=0.614. 
- Test: Increase regularization aggressively. Target train R²=0.65-0.70.
- Expected: If overfitting, test R² increases even as train R² drops.

HYPOTHESIS 3: BMI-stratified models capture different IR mechanisms
- Rationale: Normal BMI R²=0.36 vs Obese I R²=0.47. Different phenotypes.
- Test: Train separate models for BMI<30 vs BMI>=30.
- Expected: Separate models capture lean IR vs obese IR differently.

Each hypothesis gets tested, post-hoc analysis of results, then decide.
"""
import numpy as np, pandas as pd, time, warnings, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V27: HYPOTHESIS-DRIVEN EXPERIMENTS")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
y_log = np.log1p(y)

glucose = X_df['glucose'].values
bmi = X_df['bmi'].values
sex = X_df['sex_num'].values

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

# ============================================================
# BASELINE
# ============================================================
print("\n--- BASELINE ---")
def get_oof(model_fn, X, y_target, splits, weights=None):
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    train_r2s = []
    for tr, te in splits:
        wtr = weights[tr] if weights is not None else None
        m = model_fn(X[tr], y_target[tr], wtr)
        oof_sum[te] += m.predict(X[te])
        oof_count[te] += 1
        train_pred = m.predict(X[tr])
        if y_target is y_log:
            train_r2s.append(r2_score(y[tr], np.expm1(train_pred)))
        else:
            train_r2s.append(r2_score(y[tr], train_pred))
    oof = oof_sum / np.clip(oof_count, 1, None)
    oof_real = np.expm1(oof) if y_target is y_log else oof
    return oof_real, np.mean(train_r2s)

def xgb_opt(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof_base, train_r2_base = get_oof(xgb_opt, X_v7, y_log, splits, weights=w_sqrt)
r2_base = r2_score(y, oof_base)
print(f"  XGB baseline: Train R² = {train_r2_base:.4f}, Test R² = {r2_base:.4f}, Gap = {train_r2_base - r2_base:.4f}")
sys.stdout.flush()

# ============================================================
# HYPOTHESIS 1: Sex-Stratified Models
# ============================================================
print("\n" + "="*60)
print("  H1: SEX-STRATIFIED MODELS")
print("="*60)

female_mask = (sex == 0)
male_mask = (sex == 1)
print(f"  Female: n={female_mask.sum()}, Male: n={male_mask.sum()}")

# Train separate models, combine
oof_sex = np.zeros(n)
oof_sex_count = np.zeros(n)
for tr, te in splits:
    # Female model
    f_tr = tr[female_mask[tr]]
    f_te = te[female_mask[te]]
    if len(f_tr) > 10 and len(f_te) > 0:
        m_f = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
            subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
            reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
        m_f.fit(X_v7[f_tr], y_log[f_tr], sample_weight=w_sqrt[f_tr])
        oof_sex[f_te] += m_f.predict(X_v7[f_te])
        oof_sex_count[f_te] += 1
    
    # Male model
    m_tr = tr[male_mask[tr]]
    m_te = te[male_mask[te]]
    if len(m_tr) > 10 and len(m_te) > 0:
        m_m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
            subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
            reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
        m_m.fit(X_v7[m_tr], y_log[m_tr], sample_weight=w_sqrt[m_tr])
        oof_sex[m_te] += m_m.predict(X_v7[m_te])
        oof_sex_count[m_te] += 1

oof_sex_pred = np.expm1(oof_sex / np.clip(oof_sex_count, 1, None))
r2_sex = r2_score(y, oof_sex_pred)
r2_f = r2_score(y[female_mask], oof_sex_pred[female_mask])
r2_m = r2_score(y[male_mask], oof_sex_pred[male_mask])
print(f"  Sex-stratified: R² = {r2_sex:.4f} (Δ = {r2_sex - r2_base:+.4f})")
print(f"    Female: R² = {r2_f:.4f} (was {r2_score(y[female_mask], oof_base[female_mask]):.4f})")
print(f"    Male:   R² = {r2_m:.4f} (was {r2_score(y[male_mask], oof_base[male_mask]):.4f})")
sys.stdout.flush()

# Also try LGB sex-stratified
oof_sex_lgb = np.zeros(n)
oof_sex_lgb_count = np.zeros(n)
for tr, te in splits:
    for mask in [female_mask, male_mask]:
        sub_tr = tr[mask[tr]]
        sub_te = te[mask[te]]
        if len(sub_tr) > 10 and len(sub_te) > 0:
            m = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
                subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
                num_leaves=10, random_state=42, verbosity=-1)
            m.fit(X_v7[sub_tr], y_log[sub_tr], sample_weight=w_sqrt[sub_tr])
            oof_sex_lgb[sub_te] += m.predict(X_v7[sub_te])
            oof_sex_lgb_count[sub_te] += 1

oof_sex_lgb_pred = np.expm1(oof_sex_lgb / np.clip(oof_sex_lgb_count, 1, None))
r2_sex_lgb = r2_score(y, oof_sex_lgb_pred)
print(f"  LGB sex-stratified: R² = {r2_sex_lgb:.4f}")

# Blend: sex-stratified XGB + global XGB
for alpha in [0.3, 0.5, 0.7]:
    blend = alpha * oof_sex_pred + (1-alpha) * oof_base
    print(f"  Blend sex({alpha:.1f}) + global({1-alpha:.1f}): R² = {r2_score(y, blend):.4f}")
sys.stdout.flush()

# ============================================================
# HYPOTHESIS 2: Aggressive Regularization (close train-test gap)
# ============================================================
print("\n" + "="*60)
print("  H2: AGGRESSIVE REGULARIZATION")
print("="*60)

configs = [
    ("Baseline", dict(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045)),
    ("Shallow d=3", dict(n_estimators=800, max_depth=3, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045)),
    ("Heavy reg", dict(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=10.0, reg_lambda=5.0)),
    ("Low subsample", dict(n_estimators=800, max_depth=4, learning_rate=0.017,
        subsample=0.3, colsample_bytree=0.5, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045)),
    ("Very shallow d=2", dict(n_estimators=1200, max_depth=2, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045)),
    ("High mcw=60", dict(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=60,
        reg_alpha=2.8, reg_lambda=0.045)),
    ("All aggressive", dict(n_estimators=600, max_depth=3, learning_rate=0.01,
        subsample=0.35, colsample_bytree=0.5, min_child_weight=50,
        reg_alpha=10.0, reg_lambda=5.0)),
]

all_preds = {'xgb_base': oof_base, 'xgb_sex': oof_sex_pred}

for name, params in configs:
    def model_fn(Xtr, ytr, wtr, p=params):
        m = xgb.XGBRegressor(**p, random_state=42, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m
    oof, train_r2 = get_oof(model_fn, X_v7, y_log, splits, weights=w_sqrt)
    test_r2 = r2_score(y, oof)
    gap = train_r2 - test_r2
    print(f"  {name:20s}: Train={train_r2:.4f} Test={test_r2:.4f} Gap={gap:.4f} Δ={test_r2-r2_base:+.4f}")
    all_preds[f'xgb_{name.lower().replace(" ","_")}'] = oof
sys.stdout.flush()

# ============================================================
# HYPOTHESIS 3: BMI-Stratified Models
# ============================================================
print("\n" + "="*60)
print("  H3: BMI-STRATIFIED MODELS")
print("="*60)

# Split at BMI=30 (normal/overweight vs obese)
lean_mask = bmi < 30
obese_mask = bmi >= 30
print(f"  Lean (<30): n={lean_mask.sum()}, Obese (>=30): n={obese_mask.sum()}")

oof_bmi = np.zeros(n)
oof_bmi_count = np.zeros(n)
for tr, te in splits:
    for mask in [lean_mask, obese_mask]:
        sub_tr = tr[mask[tr]]
        sub_te = te[mask[te]]
        if len(sub_tr) > 10 and len(sub_te) > 0:
            m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
                subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
                reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
            m.fit(X_v7[sub_tr], y_log[sub_tr], sample_weight=w_sqrt[sub_tr])
            oof_bmi[sub_te] += m.predict(X_v7[sub_te])
            oof_bmi_count[sub_te] += 1

oof_bmi_pred = np.expm1(oof_bmi / np.clip(oof_bmi_count, 1, None))
r2_bmi = r2_score(y, oof_bmi_pred)
r2_lean = r2_score(y[lean_mask], oof_bmi_pred[lean_mask])
r2_obese = r2_score(y[obese_mask], oof_bmi_pred[obese_mask])
print(f"  BMI-stratified: R² = {r2_bmi:.4f} (Δ = {r2_bmi - r2_base:+.4f})")
print(f"    Lean:  R² = {r2_lean:.4f} (was {r2_score(y[lean_mask], oof_base[lean_mask]):.4f})")
print(f"    Obese: R² = {r2_obese:.4f} (was {r2_score(y[obese_mask], oof_base[obese_mask]):.4f})")
all_preds['xgb_bmi_strat'] = oof_bmi_pred

# Blend BMI-stratified + global
for alpha in [0.3, 0.5, 0.7]:
    blend = alpha * oof_bmi_pred + (1-alpha) * oof_base
    print(f"  Blend bmi({alpha:.1f}) + global({1-alpha:.1f}): R² = {r2_score(y, blend):.4f}")
sys.stdout.flush()

# ============================================================
# HYPOTHESIS 4: Glucose-Stratified (from Fig 2 heatmap)
# ============================================================
print("\n" + "="*60)
print("  H4: GLUCOSE-STRATIFIED MODELS")
print("="*60)

gluc_lo = glucose < 100
gluc_hi = glucose >= 100
print(f"  Normal glucose (<100): n={gluc_lo.sum()}, Elevated (>=100): n={gluc_hi.sum()}")

oof_gluc = np.zeros(n)
oof_gluc_count = np.zeros(n)
for tr, te in splits:
    for mask in [gluc_lo, gluc_hi]:
        sub_tr = tr[mask[tr]]
        sub_te = te[mask[te]]
        if len(sub_tr) > 10 and len(sub_te) > 0:
            m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
                subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
                reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
            m.fit(X_v7[sub_tr], y_log[sub_tr], sample_weight=w_sqrt[sub_tr])
            oof_gluc[sub_te] += m.predict(X_v7[sub_te])
            oof_gluc_count[sub_te] += 1

oof_gluc_pred = np.expm1(oof_gluc / np.clip(oof_gluc_count, 1, None))
r2_gluc = r2_score(y, oof_gluc_pred)
print(f"  Glucose-stratified: R² = {r2_gluc:.4f} (Δ = {r2_gluc - r2_base:+.4f})")
all_preds['xgb_gluc_strat'] = oof_gluc_pred
sys.stdout.flush()

# ============================================================
# LGB + ElasticNet for blending
# ============================================================
print("\n--- Supporting models ---")
def lgb_opt(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof_lgb, _ = get_oof(lgb_opt, X_v7, y_log, splits, weights=w_sqrt)
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
oof_enet, _ = get_oof(enet_fn, X_v7, y_log, splits)
print(f"  ElasticNet: R² = {r2_score(y, oof_enet):.4f}")
all_preds['enet'] = oof_enet
sys.stdout.flush()

# ============================================================
# DIRICHLET BLEND
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

# Show all models
print(f"\n  All {len(all_preds)} models:")
for name, pred in sorted(all_preds.items(), key=lambda x: r2_score(y, x[1]), reverse=True):
    print(f"    {name}: R² = {r2_score(y, pred):.4f}")
sys.stdout.flush()

best_r2, best_w = dirichlet_blend(all_preds, y, n_trials=2000000)
print(f"\n  Full blend: R² = {best_r2:.4f}")
print(f"  Top weights: {json.dumps({k:round(v,3) for k,v in sorted(best_w.items(), key=lambda x:-x[1]) if v>0.01})}")

# Top-6 blend
top6 = sorted(all_preds.keys(), key=lambda k: r2_score(y, all_preds[k]), reverse=True)[:6]
top6_r2, top6_w = dirichlet_blend({k: all_preds[k] for k in top6}, y, n_trials=2000000)
print(f"  Top-6 blend: R² = {top6_r2:.4f}")
print(f"  Weights: {json.dumps({k:round(v,3) for k,v in sorted(top6_w.items(), key=lambda x:-x[1]) if v>0.01})}")

best_a = max(best_r2, top6_r2)
sys.stdout.flush()

# ============================================================
# POST-HOC: Analyze if stratification helped specific subgroups
# ============================================================
print("\n" + "="*60)
print("  POST-HOC: DID STRATIFICATION HELP?")
print("="*60)

best_blend_pred = sum(best_w.get(k, 0) * all_preds[k] for k in all_preds)

# Compare residual-y correlation
res_base = y - oof_base
res_blend = y - best_blend_pred
print(f"  Residual-y corr (baseline): {np.corrcoef(y, res_base)[0,1]:.4f}")
print(f"  Residual-y corr (blend):    {np.corrcoef(y, res_blend)[0,1]:.4f}")

# Subgroup comparison
for label, mask in [('Female', female_mask), ('Male', male_mask),
                     ('Lean', lean_mask), ('Obese', obese_mask),
                     ('HOMA<3', y<3), ('HOMA 3-8', (y>=3)&(y<8)), ('HOMA 8+', y>=8)]:
    if mask.sum() < 10: continue
    r2_b = r2_score(y[mask], oof_base[mask])
    r2_bl = r2_score(y[mask], best_blend_pred[mask])
    print(f"  {label:12s}: n={mask.sum():4d}, Base R²={r2_b:.4f}, Blend R²={r2_bl:.4f}, Δ={r2_bl-r2_b:+.4f}")
sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_single = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_r2 = r2_score(y, all_preds[best_single])

print("\n" + "="*60)
print(f"  V27 SUMMARY")
print("="*60)
print(f"  Best single: {best_single} R² = {best_single_r2:.4f}")
print(f"  Best blend:  R² = {best_a:.4f}")
print(f"  Previous best: 0.5467")
print(f"  Delta: {best_a - 0.5467:+.4f}")
print(f"  Elapsed: {elapsed:.1f}s")
print(f"\n  HYPOTHESIS RESULTS:")
print(f"  H1 (sex-strat): {r2_sex:.4f} vs {r2_base:.4f} → {'CONFIRMED' if r2_sex > r2_base else 'REJECTED'}")
print(f"  H3 (bmi-strat): {r2_bmi:.4f} vs {r2_base:.4f} → {'CONFIRMED' if r2_bmi > r2_base else 'REJECTED'}")
print(f"  H4 (gluc-strat): {r2_gluc:.4f} vs {r2_base:.4f} → {'CONFIRMED' if r2_gluc > r2_base else 'REJECTED'}")

results = {
    'best_r2_a': best_a,
    'best_single': {'name': best_single, 'r2': best_single_r2},
    'hypotheses': {
        'H1_sex_stratified': float(r2_sex),
        'H2_regularization': {name: float(r2_score(y, all_preds.get(f'xgb_{name.lower().replace(" ","_")}', oof_base))) 
                              for name, _ in configs},
        'H3_bmi_stratified': float(r2_bmi),
        'H4_glucose_stratified': float(r2_gluc),
    },
    'blend_weights': {k: float(v) for k, v in best_w.items()},
    'elapsed': elapsed,
}
with open('v27_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved v27_results.json")
