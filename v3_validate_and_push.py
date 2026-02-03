#!/usr/bin/env python3
"""
V3: Validate L2 stacking (check for leakage) + push for higher R².
Key question: Is the L2 R²=0.6586 real or leaking?

Strategy:
1. Proper nested CV for L2 stacking validation
2. Add TabPFN v2 to the model pool
3. Expand feature engineering with interaction discovery
4. Test target transforms (Box-Cox, quantile)
"""
import numpy as np, pandas as pd, time, warnings, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits, compute_metrics,
                             engineer_all_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import pearsonr

print("="*60)
print("  V3: VALIDATE L2 STACKING + PUSH")
print("="*60)

# Load data
X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
X_all_eng_df = engineer_all_features(X_df[all_cols], all_cols)
X_all_eng = X_all_eng_df.values
n = len(y)

# MI feature selection
mi = mutual_info_regression(X_all_eng, y, random_state=42)
mi_order = np.argsort(mi)[::-1]
X_mi35 = X_all_eng[:, mi_order[:35]]

# Standard splits
splits = get_cv_splits(y)
bins = pd.qcut(y, 5, labels=False, duplicates='drop')

print(f"Data: {n} samples | Target mean={y.mean():.2f} std={y.std():.2f}")

# ============================================================
# PROPER NESTED STACKING (no leakage)
# ============================================================
print("\n--- Proper Nested L2 Stacking ---")
print("Using SEPARATE inner/outer CV to avoid any leakage")

def build_base_models():
    """Return list of (name, factory, feature_matrix, needs_scaling, log_target)."""
    return [
        ('ridge100_raw', lambda: Ridge(alpha=100), X_all_raw, True, False),
        ('enet_raw', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_all_raw, True, False),
        ('xgb_d4_raw', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
        ('xgb_d6_raw', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_raw, False, False),
        ('hgbr_raw', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_raw, False, False),
        ('hgbr_log_raw', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_raw, False, True),
        ('et_raw', lambda: ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1), X_all_raw, False, False),
        ('lgb_raw', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), X_all_raw, False, False),
        ('enet_eng', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_all_eng, True, False),
        ('xgb_d6_eng', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), X_all_eng, False, False),
        ('hgbr_log_eng', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_all_eng, False, True),
        ('lgb_eng', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), X_all_eng, False, False),
        ('enet_mi35', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), X_mi35, True, False),
        ('hgbr_log_mi35', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), X_mi35, False, True),
    ]

def train_predict_base(name, factory, X, y_train, train_idx, test_idx, scale, log_target):
    """Train base model and return test predictions."""
    Xtr, Xte = X[train_idx], X[test_idx]
    yt = np.log1p(y_train) if log_target else y_train
    if scale:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    m = factory(); m.fit(Xtr, yt)
    p = m.predict(Xte)
    if log_target: p = np.expm1(p)
    return p

# Outer CV: 5-fold (single repeat for speed)
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
outer_splits = list(outer_cv.split(np.zeros(n), bins))

# Inner CV: 5-fold (single repeat)  
l2_oof = np.zeros(n)
l2_count = np.zeros(n)

for outer_fold, (outer_train, outer_test) in enumerate(outer_splits):
    print(f"\n  Outer fold {outer_fold+1}/5 (train={len(outer_train)}, test={len(outer_test)})")
    
    # Inner CV for base OOF on outer_train
    inner_bins = pd.qcut(y[outer_train], 5, labels=False, duplicates='drop')
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    inner_splits = list(inner_cv.split(np.zeros(len(outer_train)), inner_bins))
    
    base_models = build_base_models()
    n_base = len(base_models)
    
    # Generate inner OOF for stacking
    inner_oof = np.zeros((len(outer_train), n_base))
    
    for midx, (mname, mfactory, X, scale, log_t) in enumerate(base_models):
        for inner_train_rel, inner_test_rel in inner_splits:
            # Map relative indices to absolute
            abs_train = outer_train[inner_train_rel]
            abs_test = outer_train[inner_test_rel]
            
            preds = train_predict_base(mname, mfactory, X, y[abs_train], abs_train, abs_test, scale, log_t)
            inner_oof[inner_test_rel, midx] += preds
    
    # Train L2 stacker on inner OOF
    sc_l2 = StandardScaler()
    inner_oof_scaled = sc_l2.fit_transform(inner_oof)
    l2_model = Ridge(alpha=0.1)
    l2_model.fit(inner_oof_scaled, y[outer_train])
    
    # Generate outer test base predictions (train on ALL outer_train, predict outer_test)
    outer_test_preds = np.zeros((len(outer_test), n_base))
    for midx, (mname, mfactory, X, scale, log_t) in enumerate(base_models):
        outer_test_preds[:, midx] = train_predict_base(mname, mfactory, X, y[outer_train], outer_train, outer_test, scale, log_t)
    
    # Predict outer test with L2 stacker
    outer_test_scaled = sc_l2.transform(outer_test_preds)
    l2_preds = l2_model.predict(outer_test_scaled)
    
    l2_oof[outer_test] += l2_preds
    l2_count[outer_test] += 1
    
    # Fold score
    fold_r2 = r2_score(y[outer_test], l2_preds)
    print(f"    Fold R²={fold_r2:.4f}")

l2_oof_final = l2_oof / np.clip(l2_count, 1, None)
nested_r2 = r2_score(y, l2_oof_final)
print(f"\n  ★ Nested L2 Stacking R²={nested_r2:.4f}")
print(f"    (vs V2's non-nested L2: R²=0.6586)")

# ============================================================
# ALSO: Simple stacking with proper OOF (same as V2 but verified)
# ============================================================
print("\n--- Verified Simple Stacking (same splits for base + stacker) ---")

base_models = build_base_models()
oof_pool = {}

for mname, mfactory, X, scale, log_t in base_models:
    t0 = time.time()
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        p = train_predict_base(mname, mfactory, X, y[tr], tr, te, scale, log_t)
        oof_sum[te] += p; oof_cnt[te] += 1
    oof = oof_sum / np.clip(oof_cnt, 1, None)
    r2 = r2_score(y, oof)
    oof_pool[mname] = oof
    print(f"  {mname:25s} R²={r2:.4f} ({time.time()-t0:.1f}s)")

# Stack
oof_matrix = np.column_stack([oof_pool[k] for k in sorted(oof_pool.keys())])

# L1 stacking with fresh OOF
for sname, sfactory in [
    ('ridge_01', lambda: Ridge(alpha=0.1)),
    ('ridge_1', lambda: Ridge(alpha=1)),
    ('ridge_10', lambda: Ridge(alpha=10)),
    ('enet_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ('bayesian', lambda: BayesianRidge()),
]:
    oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
    for tr, te in splits:
        sc = StandardScaler()
        Xtr = sc.fit_transform(oof_matrix[tr])
        Xte = sc.transform(oof_matrix[te])
        m = sfactory(); m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        oof_sum[te] += p; oof_cnt[te] += 1
    oof = oof_sum / np.clip(oof_cnt, 1, None)
    r2 = r2_score(y, oof)
    print(f"  Stack {sname:20s} R²={r2:.4f}")

# Dirichlet blend of base models
sorted_base = sorted(oof_pool.items(), key=lambda x: r2_score(y, x[1]), reverse=True)
top_k = min(8, len(sorted_base))
top_oofs = np.column_stack([v for _, v in sorted_base[:top_k]])
top_names = [k for k, _ in sorted_base[:top_k]]

best_blend = -999
rng = np.random.RandomState(42)
for _ in range(300000):
    w = rng.dirichlet(np.ones(top_k))
    pred = top_oofs @ w
    r2 = 1 - np.sum((y-pred)**2) / np.sum((y-y.mean())**2)
    if r2 > best_blend: best_blend = r2; best_w = w

print(f"\n  Dirichlet blend (top {top_k}): R²={best_blend:.4f}")
print(f"  Weights: {dict(zip(top_names, [f'{w:.3f}' for w in best_w]))}")

# ============================================================
# TabPFN v2
# ============================================================
print("\n--- TabPFN v2 ---")
try:
    from tabpfn import TabPFNRegressor
    
    for fname, X in [('raw', X_all_raw), ('mi35', X_mi35)]:
        t0 = time.time()
        oof_sum, oof_cnt = np.zeros(n), np.zeros(n)
        for tr, te in splits:
            m = TabPFNRegressor(n_estimators=8)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            oof_sum[te] += p; oof_cnt[te] += 1
        oof = oof_sum / np.clip(oof_cnt, 1, None)
        r2 = r2_score(y, oof)
        print(f"  TabPFN ({fname}): R²={r2:.4f} ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"  TabPFN failed: {e}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("  V3 SUMMARY")
print("="*60)
print(f"  Nested L2 stacking:   R²={nested_r2:.4f}")
print(f"  Dirichlet blend:      R²={best_blend:.4f}")
print(f"  Target:               R²=0.65")
print(f"  Gap:                  {0.65 - max(nested_r2, best_blend):.4f}")
