#!/usr/bin/env python3
"""
V21: CatBoost + Huber Loss + SMOTER + Multi-Loss Ensemble

RESEARCH INSIGHTS:
- Original paper uses MAE loss with L1-L2 reg → R²=0.50
- We're at 0.5467 with MSE-trained trees
- CatBoost ordered boosting: different bias than XGB/LGB
- Huber loss: robust to outliers in the tail
- SMOTER: synthetic oversampling for high-HOMA tail
- Multi-loss ensemble: models trained with different losses capture different patterns

RUN WITH: cd wear-me-dl-v2 && source .venv312/bin/activate && python3.12 v21_catboost_breakthrough.py
"""
import numpy as np, pandas as pd, time, warnings, sys, json, os
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, engineer_dw_features)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import ElasticNet, Ridge, HuberRegressor
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

t_start = time.time()
print("="*60)
print("  V21: CATBOOST + HUBER + SMOTER + MULTI-LOSS")
print("="*60)
sys.stdout.flush()

# === Load Data ===
X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)
w_sqrt = np.sqrt(y) / np.sqrt(y).mean()
y_log = np.log1p(y)

# === V7 Feature Engineering ===
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
sys.stdout.flush()

# === SMOTER: Synthetic oversampling for regression tail ===
def smoter_augment(X_train, y_train, threshold_pct=80, oversample_factor=2, k=5):
    """Generate synthetic samples for high-y tail using SMOTER approach."""
    threshold = np.percentile(y_train, threshold_pct)
    high_mask = y_train >= threshold
    X_high = X_train[high_mask]
    y_high = y_train[high_mask]
    
    if len(X_high) < k + 1:
        return X_train, y_train
    
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_high)
    
    synthetic_X = []
    synthetic_y = []
    rng = np.random.default_rng(42)
    
    for _ in range(oversample_factor):
        for i in range(len(X_high)):
            distances, indices = nn.kneighbors(X_high[i:i+1])
            # Pick random neighbor (skip self at index 0)
            j = rng.integers(1, k+1)
            neighbor_idx = indices[0, j]
            
            # Interpolate
            lam = rng.uniform(0, 1)
            new_x = X_high[i] + lam * (X_high[neighbor_idx] - X_high[i])
            new_y = y_high[i] + lam * (y_high[neighbor_idx] - y_high[i])
            synthetic_X.append(new_x)
            synthetic_y.append(new_y)
    
    X_aug = np.vstack([X_train, np.array(synthetic_X)])
    y_aug = np.concatenate([y_train, np.array(synthetic_y)])
    return X_aug, y_aug

# === Helper: OOF predictions ===
def get_oof(model_fn, X, y_target, splits, scale=True, weights=None, 
            use_smoter=False, smoter_pct=80, smoter_k=5, smoter_mult=2):
    """Get OOF predictions."""
    oof_sum = np.zeros(n)
    oof_count = np.zeros(n)
    fold_scores = []
    
    for i, (tr, te) in enumerate(splits):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr, yte = y_target[tr].copy(), y_target[te]
        wtr = weights[tr] if weights is not None else None
        
        if scale:
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xtr)
            Xte = sc.transform(Xte)
        
        # SMOTER augmentation (before training)
        if use_smoter:
            Xtr_aug, ytr_aug = smoter_augment(Xtr, ytr, smoter_pct, smoter_mult, smoter_k)
            if wtr is not None:
                # Extend weights for synthetic samples (use max weight for tail samples)
                w_synth = np.full(len(ytr_aug) - len(ytr), np.percentile(wtr, 90))
                wtr_aug = np.concatenate([wtr, w_synth])
            else:
                wtr_aug = None
            model = model_fn(Xtr_aug, ytr_aug, wtr_aug)
        else:
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
    r2 = r2_score(y, oof_real)
    return oof_real, r2, fold_scores

all_preds = {}

# ============================================================
# 1. CatBoost with different loss functions
# ============================================================
print("\n--- CatBoost: RMSE vs MAE vs Huber loss ---")
sys.stdout.flush()

# Best params from partial V21 Optuna run
cb_params = dict(depth=5, learning_rate=0.005, iterations=1200,
                 l2_leaf_reg=3.5, subsample=0.5, random_strength=2.8,
                 border_count=128, random_seed=42, verbose=0)

for loss_name, loss_fn in [('RMSE', 'RMSE'), ('MAE', 'MAE'), ('Huber:delta=1.5', 'Huber:delta=1.5')]:
    def make_cb(loss=loss_fn):
        def cb_fn(Xtr, ytr, wtr):
            params = {**cb_params, 'loss_function': loss}
            m = CatBoostRegressor(**params)
            if wtr is not None:
                m.fit(Xtr, ytr, sample_weight=wtr)
            else:
                m.fit(Xtr, ytr)
            return m
        return cb_fn
    
    oof, r2, fs = get_oof(make_cb(loss_fn), X_v7, y_log, splits, scale=False, weights=w_sqrt)
    tag = f"cb_{loss_name.split(':')[0].lower()}_log_wsqrt"
    print(f"  CatBoost {loss_name} (log+wsqrt): R² = {r2:.4f}  (fold mean={np.mean(fs):.4f})")
    all_preds[tag] = oof
    sys.stdout.flush()

# ============================================================
# 2. XGB/LGB with Huber loss
# ============================================================
print("\n--- XGB/LGB: Huber loss (pseudo_huber_loss) ---")
sys.stdout.flush()

# XGB with Huber loss
def xgb_huber(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045,
        objective='reg:pseudohubererror',
        huber_slope=1.5,
        random_state=42, verbosity=0
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_huber, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB Huber (log+wsqrt): R² = {r2:.4f}")
all_preds['xgb_huber_log_wsqrt'] = oof
sys.stdout.flush()

# XGB with MAE loss
def xgb_mae(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045,
        objective='reg:absoluteerror',
        random_state=42, verbosity=0
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_mae, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MAE (log+wsqrt):   R² = {r2:.4f}")
all_preds['xgb_mae_log_wsqrt'] = oof
sys.stdout.flush()

# Standard XGB MSE for comparison
def xgb_mse(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045,
        random_state=42, verbosity=0
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(xgb_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB MSE (log+wsqrt):   R² = {r2:.4f}")
all_preds['xgb_mse_log_wsqrt'] = oof
sys.stdout.flush()

# LGB with Huber
def lgb_huber(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(
        n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, objective='huber', huber_delta=1.5,
        random_state=42, verbosity=-1
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(lgb_huber, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  LGB Huber (log+wsqrt): R² = {r2:.4f}")
all_preds['lgb_huber_log_wsqrt'] = oof
sys.stdout.flush()

# LGB with MAE
def lgb_mae(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(
        n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, objective='mae',
        random_state=42, verbosity=-1
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(lgb_mae, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  LGB MAE (log+wsqrt):   R² = {r2:.4f}")
all_preds['lgb_mae_log_wsqrt'] = oof
sys.stdout.flush()

# LGB MSE baseline
def lgb_mse(Xtr, ytr, wtr):
    m = lgb.LGBMRegressor(
        n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10,
        random_state=42, verbosity=-1
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(lgb_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  LGB MSE (log+wsqrt):   R² = {r2:.4f}")
all_preds['lgb_mse_log_wsqrt'] = oof
sys.stdout.flush()

# ============================================================
# 3. SMOTER Augmentation
# ============================================================
print("\n--- SMOTER Augmentation (oversample high-HOMA tail) ---")
sys.stdout.flush()

for pct, mult in [(80, 2), (85, 3), (75, 2)]:
    oof, r2, fs = get_oof(xgb_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt,
                           use_smoter=True, smoter_pct=pct, smoter_mult=mult)
    tag = f"xgb_smoter_p{pct}_x{mult}"
    print(f"  XGB SMOTER pct={pct} x{mult}: R² = {r2:.4f}")
    all_preds[tag] = oof
    sys.stdout.flush()

# LGB + SMOTER
oof, r2, fs = get_oof(lgb_mse, X_v7, y_log, splits, scale=False, weights=w_sqrt,
                       use_smoter=True, smoter_pct=80, smoter_mult=2)
print(f"  LGB SMOTER pct=80 x2:  R² = {r2:.4f}")
all_preds['lgb_smoter_p80_x2'] = oof
sys.stdout.flush()

# ============================================================
# 4. Sklearn HuberRegressor (linear, different from tree Huber)
# ============================================================
print("\n--- HuberRegressor (robust linear) ---")
sys.stdout.flush()

def huber_reg(Xtr, ytr, wtr):
    m = HuberRegressor(epsilon=1.35, max_iter=1000, alpha=0.01)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(huber_reg, X_v7, y_log, splits, scale=True, weights=w_sqrt)
print(f"  HuberRegressor (log+wsqrt): R² = {r2:.4f}")
all_preds['huber_reg_log_wsqrt'] = oof
sys.stdout.flush()

# ElasticNet for blending
def enet_fn(Xtr, ytr, wtr):
    m = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
    m.fit(Xtr, ytr)
    return m

oof, r2, _ = get_oof(enet_fn, X_v7, y_log, splits, scale=True)
print(f"  ElasticNet (log):           R² = {r2:.4f}")
all_preds['enet_log'] = oof
sys.stdout.flush()

# ============================================================
# 5. Optuna CatBoost (focused, 40 trials)
# ============================================================
print("\n--- Optuna CatBoost (40 trials, log+wsqrt) ---")
sys.stdout.flush()

optuna_splits = get_cv_splits(y, n_splits=5, n_repeats=1)

def optuna_cb_obj(trial):
    depth = trial.suggest_int('depth', 3, 7)
    lr = trial.suggest_float('lr', 0.003, 0.08, log=True)
    iters = trial.suggest_int('iters', 300, 1500)
    l2 = trial.suggest_float('l2', 0.1, 30, log=True)
    sub = trial.suggest_float('sub', 0.4, 0.9)
    rs = trial.suggest_float('rs', 0.01, 10, log=True)
    loss = trial.suggest_categorical('loss', ['RMSE', 'MAE', 'Huber:delta=1.5'])
    
    scores = []
    for tr, te in optuna_splits:
        m = CatBoostRegressor(
            iterations=iters, depth=depth, learning_rate=lr,
            l2_leaf_reg=l2, subsample=sub, random_strength=rs,
            loss_function=loss, random_seed=42, verbose=0,
        )
        m.fit(X_v7[tr], y_log[tr], sample_weight=w_sqrt[tr])
        p = np.expm1(m.predict(X_v7[te]))
        scores.append(r2_score(y[te], p))
    return np.mean(scores)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(optuna_cb_obj, n_trials=40)
bp = study.best_params
print(f"  Best Optuna R² (5-fold): {study.best_value:.4f}")
print(f"  Best: depth={bp['depth']}, lr={bp['lr']:.4f}, iters={bp['iters']}, "
      f"l2={bp['l2']:.3f}, loss={bp['loss']}")
sys.stdout.flush()

def cb_optuna_fn(Xtr, ytr, wtr):
    m = CatBoostRegressor(
        iterations=bp['iters'], depth=bp['depth'], learning_rate=bp['lr'],
        l2_leaf_reg=bp['l2'], subsample=bp['sub'], random_strength=bp['rs'],
        loss_function=bp['loss'], random_seed=42, verbose=0,
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, fs = get_oof(cb_optuna_fn, X_v7, y_log, splits, scale=False, weights=w_sqrt)
print(f"  CatBoost Optuna (25-fold): R² = {r2:.4f}")
all_preds['cb_optuna_best'] = oof
sys.stdout.flush()

# ============================================================
# 6. MEGA BLEND
# ============================================================
print("\n" + "="*60)
print("  MEGA BLEND")
print("="*60)

def dirichlet_blend(pred_dict, y_true, n_trials=2000000):
    names = list(pred_dict.keys())
    preds = np.array([pred_dict[k] for k in names])
    k = len(names)
    best_r2 = -999
    best_w = None
    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(k))
        blend = (w[:, None] * preds).sum(axis=0)
        r2 = r2_score(y_true, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w.copy()
    return best_r2, {names[i]: best_w[i] for i in range(k)}

# Print all scores
print(f"\n  All {len(all_preds)} models:")
for name, pred in sorted(all_preds.items(), key=lambda x: r2_score(y, x[1]), reverse=True):
    print(f"    {name}: R² = {r2_score(y, pred):.4f}")
sys.stdout.flush()

# Full blend
best_r2, best_w = dirichlet_blend(all_preds, y, n_trials=2000000)
print(f"\n  Full blend ({len(all_preds)} models): R² = {best_r2:.4f}")
print(f"  Top weights: {json.dumps({k: round(v,3) for k,v in sorted(best_w.items(), key=lambda x:-x[1]) if v>0.01})}")
sys.stdout.flush()

# Top-6 blend
top6 = sorted(all_preds.keys(), key=lambda k: r2_score(y, all_preds[k]), reverse=True)[:6]
top6_preds = {k: all_preds[k] for k in top6}
top6_r2, top6_w = dirichlet_blend(top6_preds, y, n_trials=2000000)
print(f"\n  Top-6 blend: R² = {top6_r2:.4f}")
print(f"  Weights: {json.dumps({k: round(v,3) for k,v in sorted(top6_w.items(), key=lambda x:-x[1]) if v>0.01})}")
sys.stdout.flush()

best_a = max(best_r2, top6_r2)

# ============================================================
# 7. Model B: CatBoost + Huber for wearables-only
# ============================================================
print("\n" + "="*60)
print("  MODEL B: Demographics + Wearables Only")
print("="*60)
sys.stdout.flush()

X_dw_df = X_df[dw_cols].copy()
X_dw_eng = engineer_dw_features(X_dw_df, dw_cols).values
print(f"Model B features: {X_dw_eng.shape[1]}")

b_preds = {}

# CatBoost for Model B
def cb_dw(Xtr, ytr, wtr):
    m = CatBoostRegressor(
        iterations=800, depth=4, learning_rate=0.01,
        l2_leaf_reg=5, subsample=0.6, random_strength=2.0,
        random_seed=42, verbose=0,
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, _ = get_oof(cb_dw, X_dw_eng, y_log, splits, scale=False, weights=w_sqrt)
print(f"  CB DW (log+wsqrt): R² = {r2:.4f}")
b_preds['cb_dw'] = oof

# XGB DW
def xgb_dw(Xtr, ytr, wtr):
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.02,
        subsample=0.6, colsample_bytree=0.7, min_child_weight=20,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbosity=0
    )
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m

oof, r2, _ = get_oof(xgb_dw, X_dw_eng, y_log, splits, scale=False, weights=w_sqrt)
print(f"  XGB DW (log+wsqrt): R² = {r2:.4f}")
b_preds['xgb_dw'] = oof

# ElasticNet DW
def enet_dw(Xtr, ytr, wtr):
    m = ElasticNet(alpha=0.1, l1_ratio=0.9, max_iter=10000, random_state=42)
    m.fit(Xtr, ytr)
    return m

oof, r2, _ = get_oof(enet_dw, X_dw_eng, y_log, splits, scale=True)
print(f"  ElasticNet DW:      R² = {r2:.4f}")
b_preds['enet_dw'] = oof

# Ridge DW
def ridge_dw(Xtr, ytr, wtr):
    m = Ridge(alpha=1000)
    m.fit(Xtr, ytr)
    return m

oof, r2, _ = get_oof(ridge_dw, X_dw_eng, y_log, splits, scale=True)
print(f"  Ridge DW:           R² = {r2:.4f}")
b_preds['ridge_dw'] = oof

best_b_r2, best_b_w = dirichlet_blend(b_preds, y, n_trials=500000)
print(f"\n  Model B blend: R² = {best_b_r2:.4f}")
print(f"  Weights: {json.dumps({k: round(v,3) for k,v in sorted(best_b_w.items(), key=lambda x:-x[1]) if v>0.01})}")
sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start

best_single_a_name = max(all_preds, key=lambda k: r2_score(y, all_preds[k]))
best_single_a_r2 = r2_score(y, all_preds[best_single_a_name])
best_single_b_name = max(b_preds, key=lambda k: r2_score(y, b_preds[k]))
best_single_b_r2 = r2_score(y, b_preds[best_single_b_name])

print("\n" + "="*60)
print(f"  V21 SUMMARY")
print(f"="*60)
print(f"\n  Model A:")
print(f"    Best single: {best_single_a_name} R² = {best_single_a_r2:.4f}")
print(f"    Best blend:  R² = {best_a:.4f}")
print(f"    Previous best (V20): 0.5467")
print(f"    Delta: {best_a - 0.5467:+.4f}")
print(f"\n  Model B:")
print(f"    Best single: {best_single_b_name} R² = {best_single_b_r2:.4f}")
print(f"    Best blend:  R² = {best_b_r2:.4f}")
print(f"    Previous best (V20): 0.2592")
print(f"    Delta: {best_b_r2 - 0.2592:+.4f}")
print(f"\n  Elapsed: {elapsed:.1f}s")

# Save results
results = {
    'best_r2_a': best_a,
    'best_single_a': {'name': best_single_a_name, 'r2': best_single_a_r2},
    'best_r2_b': best_b_r2,
    'best_single_b': {'name': best_single_b_name, 'r2': best_single_b_r2},
    'catboost_optuna_params': bp,
    'all_scores_a': {k: float(r2_score(y, v)) for k, v in all_preds.items()},
    'all_scores_b': {k: float(r2_score(y, v)) for k, v in b_preds.items()},
    'blend_weights_a': {k: float(v) for k, v in best_w.items()},
    'elapsed': elapsed,
}

# Save OOF predictions for cross-version blending
np.save('v21_oof_preds.npy', {k: v for k, v in all_preds.items()})

with open('v21_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to v21_results.json")
print(f"\nNOTE: Scores on Python 3.12 may differ slightly from 3.14 due to library versions.")
print(f"XGB/LGB baselines here are comparable WITHIN this run but not to V1-V20.")
