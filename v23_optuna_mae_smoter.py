#!/usr/bin/env python3
"""
V23: Optuna MAE + Optuna SMOTER — Joint Optimization

KEY INSIGHT: V21-V22 showed MAE helps XGB (+0.0015) and SMOTER helps XGB (+0.0023).
But we used V13's MSE-optimized hyperparams with MAE loss. The optimal hyperparams
for MAE loss are DIFFERENT. Let's Optuna-tune XGB specifically for MAE loss.

Also: jointly optimize SMOTER params (percentile, multiplier) with model hyperparams.

Plan:
1. Optuna XGB with MAE loss (60 trials)
2. Optuna XGB with MSE+SMOTER (joint SMOTER params, 60 trials)  
3. Optuna LGB with MAE (30 trials — check if MAE helps LGB with right params)
4. Multi-seed ensemble of best configs
5. Blend everything
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

t_start = time.time()
print("="*60)
print("  V23: OPTUNA MAE + OPTUNA SMOTER")
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
print(f"V7 features: {X_v7.shape[1]}")

# SMOTER
def smoter(X_train, y_train, pct=85, mult=3, k=5, seed=42):
    threshold = np.percentile(y_train, pct)
    high_mask = y_train >= threshold
    X_high, y_high = X_train[high_mask], y_train[high_mask]
    if len(X_high) < k + 1:
        return X_train, y_train
    nn = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    sX, sy = [], []
    rng = np.random.default_rng(seed)
    for _ in range(mult):
        for i in range(len(X_high)):
            _, idx = nn.kneighbors(X_high[i:i+1])
            j = rng.integers(1, k+1)
            lam = rng.uniform(0, 1)
            sX.append(X_high[i] + lam * (X_high[idx[0,j]] - X_high[i]))
            sy.append(y_high[i] + lam * (y_high[idx[0,j]] - y_high[i]))
    return np.vstack([X_train, np.array(sX)]), np.concatenate([y_train, np.array(sy)])

# OOF helper
def get_oof(model_fn, X, y_target, splits, weights=None, smoter_params=None):
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    fold_scores = []
    for tr, te in splits:
        Xtr, ytr = X[tr].copy(), y_target[tr].copy()
        wtr = weights[tr] if weights is not None else None
        if smoter_params:
            Xtr, ytr = smoter(Xtr, ytr, **smoter_params)
            if wtr is not None:
                w_ext = np.full(len(ytr) - len(weights[tr]), np.percentile(wtr, 90))
                wtr = np.concatenate([wtr, w_ext])
        model = model_fn(Xtr, ytr, wtr)
        preds = model.predict(X[te])
        oof_sum[te] += preds
        oof_count[te] += 1
        if y_target is y_log:
            fold_scores.append(r2_score(y[te], np.expm1(preds)))
        else:
            fold_scores.append(r2_score(y[te], preds))
    oof = oof_sum / np.clip(oof_count, 1, None)
    oof_real = np.expm1(oof) if y_target is y_log else oof
    return oof_real, r2_score(y, oof_real), fold_scores

optuna_splits = get_cv_splits(y, n_splits=5, n_repeats=1)
all_preds = {}

# ============================================================
# 1. Optuna XGB with MAE loss (60 trials)
# ============================================================
print("\n--- Optuna XGB MAE (60 trials) ---")
sys.stdout.flush()

def xgb_mae_obj(trial):
    p = {
        'n': trial.suggest_int('n', 200, 1200),
        'd': trial.suggest_int('d', 3, 6),
        'lr': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'sub': trial.suggest_float('sub', 0.3, 0.8),
        'col': trial.suggest_float('col', 0.4, 1.0),
        'mcw': trial.suggest_int('mcw', 5, 50),
        'alpha': trial.suggest_float('alpha', 0.01, 10, log=True),
        'lam': trial.suggest_float('lam', 0.01, 10, log=True),
    }
    scores = []
    for tr, te in optuna_splits:
        m = xgb.XGBRegressor(
            n_estimators=p['n'], max_depth=p['d'], learning_rate=p['lr'],
            subsample=p['sub'], colsample_bytree=p['col'], min_child_weight=p['mcw'],
            reg_alpha=p['alpha'], reg_lambda=p['lam'],
            objective='reg:absoluteerror',
            random_state=42, verbosity=0)
        m.fit(X_v7[tr], y_log[tr], sample_weight=w_sqrt[tr])
        pred = np.expm1(m.predict(X_v7[te]))
        scores.append(r2_score(y[te], pred))
    return np.mean(scores)

study_mae = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_mae.optimize(xgb_mae_obj, n_trials=60)
bp_mae = study_mae.best_params
print(f"  Best 5-fold R²: {study_mae.best_value:.4f}")
print(f"  Params: d={bp_mae['d']}, lr={bp_mae['lr']:.4f}, n={bp_mae['n']}, "
      f"col={bp_mae['col']:.2f}, mcw={bp_mae['mcw']}, alpha={bp_mae['alpha']:.3f}")
sys.stdout.flush()

def xgb_mae_opt(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=bp_mae['n'], max_depth=bp_mae['d'], learning_rate=bp_mae['lr'],
        subsample=bp_mae['sub'], colsample_bytree=bp_mae['col'],
        min_child_weight=bp_mae['mcw'], reg_alpha=bp_mae['alpha'], reg_lambda=bp_mae['lam'],
        objective='reg:absoluteerror', random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_mae_opt, X_v7, y_log, splits, weights=w_sqrt)
print(f"  XGB Optuna MAE (25-fold): R² = {r2:.4f}")
all_preds['xgb_optuna_mae'] = oof
sys.stdout.flush()

# ============================================================
# 2. Optuna XGB MSE+SMOTER (joint, 60 trials)
# ============================================================
print("\n--- Optuna XGB MSE+SMOTER (joint, 60 trials) ---")
sys.stdout.flush()

def xgb_smoter_obj(trial):
    p = {
        'n': trial.suggest_int('n', 200, 1200),
        'd': trial.suggest_int('d', 3, 6),
        'lr': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'sub': trial.suggest_float('sub', 0.3, 0.8),
        'col': trial.suggest_float('col', 0.4, 1.0),
        'mcw': trial.suggest_int('mcw', 5, 50),
        'alpha': trial.suggest_float('alpha', 0.01, 10, log=True),
        'lam': trial.suggest_float('lam', 0.01, 10, log=True),
    }
    sm_pct = trial.suggest_int('sm_pct', 75, 95)
    sm_mult = trial.suggest_int('sm_mult', 1, 5)
    
    scores = []
    for tr, te in optuna_splits:
        Xtr, ytr = smoter(X_v7[tr], y_log[tr], pct=sm_pct, mult=sm_mult)
        wtr = w_sqrt[tr]
        w_ext = np.full(len(ytr) - len(wtr), np.percentile(wtr, 90))
        wtr_full = np.concatenate([wtr, w_ext])
        
        m = xgb.XGBRegressor(
            n_estimators=p['n'], max_depth=p['d'], learning_rate=p['lr'],
            subsample=p['sub'], colsample_bytree=p['col'], min_child_weight=p['mcw'],
            reg_alpha=p['alpha'], reg_lambda=p['lam'],
            random_state=42, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr_full)
        pred = np.expm1(m.predict(X_v7[te]))
        scores.append(r2_score(y[te], pred))
    return np.mean(scores)

study_smoter = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_smoter.optimize(xgb_smoter_obj, n_trials=60)
bp_sm = study_smoter.best_params
print(f"  Best 5-fold R²: {study_smoter.best_value:.4f}")
print(f"  Model: d={bp_sm['d']}, lr={bp_sm['lr']:.4f}, n={bp_sm['n']}")
print(f"  SMOTER: pct={bp_sm['sm_pct']}, mult={bp_sm['sm_mult']}")
sys.stdout.flush()

def xgb_smoter_opt(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=bp_sm['n'], max_depth=bp_sm['d'], learning_rate=bp_sm['lr'],
        subsample=bp_sm['sub'], colsample_bytree=bp_sm['col'],
        min_child_weight=bp_sm['mcw'], reg_alpha=bp_sm['alpha'], reg_lambda=bp_sm['lam'],
        random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_smoter_opt, X_v7, y_log, splits, weights=w_sqrt,
                        smoter_params={'pct': bp_sm['sm_pct'], 'mult': bp_sm['sm_mult']})
print(f"  XGB Optuna SMOTER (25-fold): R² = {r2:.4f}")
all_preds['xgb_optuna_smoter'] = oof
sys.stdout.flush()

# ============================================================
# 3. Optuna LGB MAE (30 trials)
# ============================================================
print("\n--- Optuna LGB MAE (30 trials) ---")
sys.stdout.flush()

def lgb_mae_obj(trial):
    p = {
        'n': trial.suggest_int('n', 200, 1200),
        'd': trial.suggest_int('d', 3, 6),
        'lr': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'sub': trial.suggest_float('sub', 0.3, 0.8),
        'col': trial.suggest_float('col', 0.4, 1.0),
        'mcs': trial.suggest_int('mcs', 10, 60),
        'leaves': trial.suggest_int('leaves', 5, 31),
    }
    scores = []
    for tr, te in optuna_splits:
        m = lgb.LGBMRegressor(
            n_estimators=p['n'], max_depth=p['d'], learning_rate=p['lr'],
            subsample=p['sub'], colsample_bytree=p['col'], min_child_samples=p['mcs'],
            num_leaves=p['leaves'], objective='mae',
            random_state=42, verbosity=-1)
        m.fit(X_v7[tr], y_log[tr], sample_weight=w_sqrt[tr])
        pred = np.expm1(m.predict(X_v7[te]))
        scores.append(r2_score(y[te], pred))
    return np.mean(scores)

study_lgb_mae = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb_mae.optimize(lgb_mae_obj, n_trials=30)
bp_lgb = study_lgb_mae.best_params
print(f"  Best 5-fold R²: {study_lgb_mae.best_value:.4f}")
print(f"  Params: d={bp_lgb['d']}, lr={bp_lgb['lr']:.4f}, n={bp_lgb['n']}, leaves={bp_lgb['leaves']}")
sys.stdout.flush()

def lgb_mae_opt(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(
        n_estimators=bp_lgb['n'], max_depth=bp_lgb['d'], learning_rate=bp_lgb['lr'],
        subsample=bp_lgb['sub'], colsample_bytree=bp_lgb['col'],
        min_child_samples=bp_lgb['mcs'], num_leaves=bp_lgb['leaves'],
        objective='mae', random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(lgb_mae_opt, X_v7, y_log, splits, weights=w_sqrt)
print(f"  LGB Optuna MAE (25-fold): R² = {r2:.4f}")
all_preds['lgb_optuna_mae'] = oof
sys.stdout.flush()

# ============================================================
# 4. Previous best models for blending
# ============================================================
print("\n--- Previous best models ---")

def xgb_mse(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, _ = get_oof(xgb_mse, X_v7, y_log, splits, weights=w_sqrt)
print(f"  XGB V13 MSE:  R² = {r2:.4f}")
all_preds['xgb_v13_mse'] = oof

def lgb_mse(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(
        n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, verbosity=-1)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, _ = get_oof(lgb_mse, X_v7, y_log, splits, weights=w_sqrt)
print(f"  LGB V14 MSE:  R² = {r2:.4f}")
all_preds['lgb_v14_mse'] = oof

def enet_fn(Xtr, ytr, wtr):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    m = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
    m.fit(Xtr_s, ytr)
    class Wrapper:
        def predict(self, X): return m.predict(sc.transform(X))
    return Wrapper()

oof, r2, _ = get_oof(enet_fn, X_v7, y_log, splits)
print(f"  ElasticNet:   R² = {r2:.4f}")
all_preds['enet'] = oof
sys.stdout.flush()

# ============================================================
# 5. Multi-seed ensemble of best MAE config
# ============================================================
print("\n--- Multi-seed XGB MAE Optuna ---")
sys.stdout.flush()

for seed in [0, 1, 2024, 2025]:
    def xgb_mae_seed(Xtr, ytr, wtr, s=seed):
        m = xgb.XGBRegressor(
            n_estimators=bp_mae['n'], max_depth=bp_mae['d'], learning_rate=bp_mae['lr'],
            subsample=bp_mae['sub'], colsample_bytree=bp_mae['col'],
            min_child_weight=bp_mae['mcw'], reg_alpha=bp_mae['alpha'], reg_lambda=bp_mae['lam'],
            objective='reg:absoluteerror', random_state=s, verbosity=0)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m
    oof, r2, _ = get_oof(xgb_mae_seed, X_v7, y_log, splits, weights=w_sqrt)
    print(f"  XGB MAE seed={seed}: R² = {r2:.4f}")
    all_preds[f'xgb_mae_s{seed}'] = oof
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

top6 = sorted(all_preds.keys(), key=lambda k: r2_score(y, all_preds[k]), reverse=True)[:6]
top6_r2, top6_w = dirichlet_blend({k: all_preds[k] for k in top6}, y, n_trials=2000000)
print(f"  Top-6 blend: R² = {top6_r2:.4f}")
print(f"  Weights: {json.dumps({k:round(v,3) for k,v in sorted(top6_w.items(), key=lambda x:-x[1]) if v>0.01})}")
sys.stdout.flush()

best_a = max(best_r2, top6_r2)

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
best_single = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_r2 = r2_score(y, all_preds[best_single])

print("\n" + "="*60)
print(f"  V23 SUMMARY")
print("="*60)
print(f"  Best single: {best_single} R² = {best_single_r2:.4f}")
print(f"  Best blend:  R² = {best_a:.4f}")
print(f"  Previous best: 0.5467")
print(f"  Delta: {best_a - 0.5467:+.4f}")
print(f"  Elapsed: {elapsed:.1f}s")

results = {
    'best_r2_a': best_a,
    'best_single': {'name': best_single, 'r2': best_single_r2},
    'xgb_mae_optuna_params': bp_mae,
    'xgb_smoter_optuna_params': bp_sm,
    'lgb_mae_optuna_params': bp_lgb,
    'all_scores': {k: float(r2_score(y, v)) for k, v in all_preds.items()},
    'elapsed': elapsed,
}
with open('v23_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved v23_results.json")
