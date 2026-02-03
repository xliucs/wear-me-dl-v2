#!/usr/bin/env python3
"""
V8: PyTorch Neural Network + CatBoost + refined blending.

V7 key findings:
- Log target consistently +0.015 R² across all models
- V7 features (72) > eng (63) > raw (25) for trees
- Best single: XGB d3 log eng R²=0.5271
- Best blend: Top-15 Dirichlet R²=0.5368

V8 strategy:
1. PyTorch FeatureGatedBlock (from reference repo) — adds DIVERSITY
2. CatBoost (strong for tabular, used in IR prediction papers)
3. SVR/KRR (kernel methods for different inductive bias)
4. Mega-blend V7 pool + V8 new models
"""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb

t_start = time.time()
print("="*60)
print("  V8: NEURAL NET + CATBOOST + MEGA-BLEND")
print("="*60)
sys.stdout.flush()

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
splits = get_cv_splits(y)
n = len(y)

# Reuse V7 feature engineering
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
            if s in X.columns:
                X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    X['log_glucose'] = np.log(g); X['log_trig'] = np.log(t)
    X['log_bmi'] = np.log(b.clip(lower=1)); X['log_hdl'] = np.log(h)
    X['log_homa_proxy'] = np.log(g) + np.log(b.clip(lower=1)) + np.log(t) - np.log(h)
    return X.fillna(0)

X_v7 = engineer_v7(X_df[all_cols], all_cols)
X_v7_all = X_v7.values
X_eng = engineer_all_features(X_df[all_cols], all_cols).values

def get_oof(model_fn, X, y_arr, splits, scale=False, log_target=False):
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_arr) if log_target else y_arr
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    return oof_sum / np.clip(oof_cnt, 1, None)

oof_pool = {}
scores = {}

# ============================================================
# PART 1: PyTorch Neural Network
# ============================================================
print("\n--- Part 1: PyTorch Neural Network ---")
sys.stdout.flush()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class WearMENet(nn.Module):
    """Simple residual MLP — avoids SIGSEGV from complex gating on Python 3.14."""
    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, dropout=0.15):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_blocks):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_pytorch_oof(X, y_arr, splits, hidden_dim=128, n_blocks=3, dropout=0.15,
                       lr=1e-3, epochs=200, batch_size=64, log_target=False, weight_decay=1e-4):
    """Train PyTorch model with OOF predictions."""
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_arr) if log_target else y_arr
    
    for fold_idx, (tr, te) in enumerate(splits):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        
        X_train_t = torch.FloatTensor(Xtr); y_train_t = torch.FloatTensor(yt[tr])
        X_test_t = torch.FloatTensor(Xte)
        
        ds = TensorDataset(X_train_t, y_train_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
        
        model = WearMENet(X.shape[1], hidden_dim, n_blocks, dropout)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_loss = float('inf'); best_state = None; patience = 30; wait = 0
        
        # Split train into train/val for early stopping (80/20)
        n_val = max(1, len(tr) // 5)
        val_idx = np.random.RandomState(42 + fold_idx).choice(len(tr), n_val, replace=False)
        train_mask = np.ones(len(tr), dtype=bool); train_mask[val_idx] = False
        X_val_inner = torch.FloatTensor(Xtr[val_idx]); y_val_inner = torch.FloatTensor(yt[tr][val_idx])
        
        for epoch in range(epochs):
            model.train()
            for xb, yb in dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = nn.MSELoss()(pred, yb)
                loss.backward(); optimizer.step()
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_inner)
                val_loss = nn.MSELoss()(val_pred, y_val_inner).item()
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience: break
        
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            p = model(X_test_t).numpy()
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_cnt[te] += 1
    
    return oof_sum / np.clip(oof_cnt, 1, None)

# PyTorch configs
nn_configs = [
    ('nn_128_3_log', dict(hidden_dim=128, n_blocks=3, dropout=0.15, lr=1e-3, epochs=200, log_target=True)),
    ('nn_256_4_log', dict(hidden_dim=256, n_blocks=4, dropout=0.2, lr=5e-4, epochs=250, log_target=True)),
    ('nn_64_2_log', dict(hidden_dim=64, n_blocks=2, dropout=0.1, lr=2e-3, epochs=200, log_target=True)),
    ('nn_128_3', dict(hidden_dim=128, n_blocks=3, dropout=0.15, lr=1e-3, epochs=200, log_target=False)),
]

for fs_name, X_fs in [('v7_all', X_v7_all), ('eng', X_eng)]:
    for nn_name, nn_params in nn_configs:
        full_name = f'{nn_name}_{fs_name}'
        t0 = time.time()
        try:
            oof = train_pytorch_oof(X_fs, y, splits, **nn_params)
            r2 = r2_score(y, oof)
            oof_pool[full_name] = oof
            scores[full_name] = r2
            print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  {full_name:40s} FAILED: {e}")
        sys.stdout.flush()

# ============================================================
# PART 2: CatBoost
# ============================================================
print("\n--- Part 2: CatBoost ---")
sys.stdout.flush()

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("  CatBoost not installed, skipping")

if HAS_CATBOOST:
    cb_configs = [
        ('cb_d4', dict(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=3, verbose=0), False),
        ('cb_d6', dict(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5, verbose=0), False),
        ('cb_d4_log', dict(iterations=500, depth=4, learning_rate=0.03, l2_leaf_reg=3, verbose=0), True),
        ('cb_d6_log', dict(iterations=500, depth=6, learning_rate=0.03, l2_leaf_reg=5, verbose=0), True),
    ]
    for fs_name, X_fs in [('v7_all', X_v7_all), ('eng', X_eng), ('raw', X_all_raw)]:
        for cb_name, cb_params, log_t in cb_configs:
            full_name = f'{cb_name}_{fs_name}'
            t0 = time.time()
            oof = get_oof(lambda p=cb_params: CatBoostRegressor(**p, random_seed=42),
                          X_fs, y, splits, log_target=log_t)
            r2 = r2_score(y, oof)
            oof_pool[full_name] = oof
            scores[full_name] = r2
            print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
            sys.stdout.flush()

# ============================================================
# PART 3: SVR + KNN (different inductive bias)
# ============================================================
print("\n--- Part 3: SVR + KNN ---")
sys.stdout.flush()

for fs_name, X_fs in [('v7_all', X_v7_all), ('eng', X_eng)]:
    for mname, mfn in [
        ('svr_rbf', lambda: SVR(kernel='rbf', C=10, gamma='scale')),
        ('svr_rbf_c100', lambda: SVR(kernel='rbf', C=100, gamma='scale')),
        ('knn_10', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance')),
        ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance')),
    ]:
        full_name = f'{mname}_{fs_name}'
        t0 = time.time()
        oof = get_oof(mfn, X_fs, y, splits, scale=True)
        r2 = r2_score(y, oof)
        oof_pool[full_name] = oof
        scores[full_name] = r2
        print(f"  {full_name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

# ============================================================
# PART 4: Reproduce V7 best models for mega-blend
# ============================================================
print("\n--- Part 4: V7 best models (for mega-blend) ---")
sys.stdout.flush()

v7_best_configs = [
    ('xgb_d3_log_eng', dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
                             colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01), X_eng, True),
    ('xgb_d6_log_v7', dict(n_estimators=800, max_depth=6, learning_rate=0.01, subsample=0.6,
                            colsample_bytree=0.5, min_child_weight=15, reg_alpha=0.1, reg_lambda=2.0), X_v7_all, True),
    ('xgb_d3_log_v7', dict(n_estimators=400, max_depth=3, learning_rate=0.03, subsample=0.55,
                            colsample_bytree=0.57, min_child_weight=17, reg_alpha=0.49, reg_lambda=0.01), X_v7_all, True),
    ('hgbr_d4_log_v7', dict(max_iter=500, max_depth=4, learning_rate=0.03, min_samples_leaf=10), X_v7_all, True),
    ('lgb_d4_log_v7', dict(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.7, min_child_samples=15, verbose=-1), X_v7_all, True),
    ('enet_eng', None, X_eng, False),
]

for name, params, X_fs, log_t in v7_best_configs:
    t0 = time.time()
    if params is None:  # ElasticNet
        oof = get_oof(lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_fs, y, splits, scale=True)
    elif 'max_iter' in params:  # HGBR
        oof = get_oof(lambda p=params: HistGradientBoostingRegressor(**p, random_state=42), X_fs, y, splits, log_target=log_t)
    elif 'verbose' in params:  # LGB
        oof = get_oof(lambda p=params: lgb.LGBMRegressor(**p, random_state=42), X_fs, y, splits, log_target=log_t)
    else:  # XGB
        oof = get_oof(lambda p=params: xgb.XGBRegressor(**p, random_state=42, verbosity=0), X_fs, y, splits, log_target=log_t)
    r2 = r2_score(y, oof)
    oof_pool[name] = oof
    scores[name] = r2
    print(f"  {name:40s} R²={r2:.4f} ({time.time()-t0:.1f}s)")
    sys.stdout.flush()

# ============================================================
# PART 5: MEGA-BLEND
# ============================================================
print(f"\n--- Part 5: Mega-Blend ({len(oof_pool)} models) ---")
sys.stdout.flush()

sorted_models = sorted(scores, key=scores.get, reverse=True)
print(f"  Top 15:")
for i, name in enumerate(sorted_models[:15], 1):
    print(f"    {i:2d}. {name:40s} R²={scores[name]:.4f}")

for top_k in [5, 8, 10, 15, 20, 25]:
    if top_k > len(sorted_models): break
    top_names = sorted_models[:top_k]
    top_oofs = np.column_stack([oof_pool[k] for k in top_names])
    best_r2 = -999
    rng = np.random.RandomState(42)
    n_trials = 500000
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(top_k))
        pred = top_oofs @ w
        r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
        if r2 > best_r2: best_r2 = r2; best_w = w
    print(f"  Top-{top_k:2d} Dirichlet: R²={best_r2:.4f}")
    if top_k == 15:
        best_blend_r2 = best_r2
        best_blend_w = dict(zip(top_names, [f'{w:.4f}' for w in best_w]))
sys.stdout.flush()

# ============================================================
# SUMMARY
# ============================================================
elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  V8 SUMMARY ({elapsed:.0f}s)")
print(f"{'='*60}")
best_name = sorted_models[0]
best_single = scores[best_name]
print(f"  ★ BEST SINGLE: {best_name} R²={best_single:.4f}")
print(f"  ★ BEST BLEND: R²={best_blend_r2:.4f}")
print(f"  Target: 0.65 | Gap: {0.65 - best_blend_r2:.4f}")

results = {
    'best_single': {'name': best_name, 'r2': float(best_single)},
    'best_blend_r2': float(best_blend_r2),
    'best_blend_weights': best_blend_w,
    'all_scores': {k: float(v) for k, v in sorted(scores.items(), key=lambda x: -x[1])[:20]},
    'elapsed': elapsed
}
with open('v8_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"  Saved to v8_results.json")
sys.stdout.flush()
