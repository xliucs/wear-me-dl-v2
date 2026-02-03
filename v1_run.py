#!/usr/bin/env python3
"""V1: Fast baseline run — fewer models, focus on best performers."""
import numpy as np, pandas as pd, time, warnings, sys, json
warnings.filterwarnings('ignore')
from eval_framework import (load_data, get_feature_sets, get_cv_splits, compute_metrics,
                             engineer_all_features, engineer_dw_features,
                             DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import lightgbm as lgb

X_df, y, fn = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_df)
X_all_eng = engineer_all_features(X_df[all_cols], all_cols).values
X_dw_eng = engineer_dw_features(X_df[dw_cols], dw_cols).values
splits = get_cv_splits(y)
n = len(y)

print(f"Data: {n} samples | ALL raw:{X_all_raw.shape[1]} eng:{X_all_eng.shape[1]} | DW raw:{X_dw_raw.shape[1]} eng:{X_dw_eng.shape[1]}")
print(f"Target: mean={y.mean():.2f} std={y.std():.2f} | CV: {len(splits)} splits\n")

models = {
    'ridge_100': (lambda: Ridge(alpha=100), True),
    'ridge_1000': (lambda: Ridge(alpha=1000), True),
    'lasso_01': (lambda: Lasso(alpha=0.1, max_iter=5000), True),
    'elasticnet': (lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), True),
    'bayesian': (lambda: BayesianRidge(), True),
    'xgb_d3': (lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), False),
    'xgb_d4': (lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), False),
    'xgb_d6': (lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0), False),
    'hgbr': (lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), False),
    'hgbr_log': (lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), False),  # log target
    'rf': (lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1), False),
    'et': (lambda: ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=1), False),
    'lgb': (lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1), False),
    'kr_rbf': (lambda: KernelRidge(alpha=1.0, kernel='rbf'), True),
    'svr_rbf': (lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
    'knn_15': (lambda: KNeighborsRegressor(n_neighbors=15, weights='distance'), True),
}

def run_model(name, model_fn, X, y_target, splits, scale, log_target=False):
    oof_sum, oof_count = np.zeros(n), np.zeros(n)
    yt = np.log1p(y_target) if log_target else y_target
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
        m = model_fn(); m.fit(Xtr, yt[tr])
        p = m.predict(Xte)
        if log_target: p = np.expm1(p)
        oof_sum[te] += p; oof_count[te] += 1
    return oof_sum / np.clip(oof_count, 1, None)

all_results = []

for fs_name, datasets in [('ALL', [(X_all_raw, 'raw'), (X_all_eng, 'eng')]),
                            ('DW', [(X_dw_raw, 'raw'), (X_dw_eng, 'eng')])]:
    print(f"\n{'='*60}\n  {fs_name} FEATURES\n{'='*60}")
    for X, feat_tag in datasets:
        for mname, (mfn, scale) in models.items():
            log_t = mname == 'hgbr_log'
            t0 = time.time()
            try:
                oof = run_model(mname, mfn, X, y, splits, scale, log_target=log_t)
                r2 = r2_score(y, oof)
                elapsed = time.time() - t0
                tag = f"{'log_' if log_t else ''}{feat_tag}"
                print(f"  {mname:15s} [{tag:8s}] R²={r2:.4f} ({elapsed:.1f}s)")
                all_results.append({'set': fs_name, 'feat': feat_tag, 'model': mname, 'log': log_t, 'R2': r2, 'oof': oof.tolist()})
            except Exception as e:
                print(f"  {mname:15s} [{feat_tag:8s}] FAILED: {e}")

# Summary
print(f"\n{'='*60}\n  TOP 10 OVERALL\n{'='*60}")
all_results.sort(key=lambda x: x['R2'], reverse=True)
for i, r in enumerate(all_results[:10], 1):
    m = compute_metrics(y, np.array(r['oof']))
    print(f"  {i}. {r['set']:>3} {r['feat']:>4} {r['model']:>15} R²={r['R2']:.4f} r={m['Pearson_r']:.4f}")

print(f"\n  Best ALL: {max((r for r in all_results if r['set']=='ALL'), key=lambda x: x['R2'])['R2']:.4f}")
print(f"  Best DW:  {max((r for r in all_results if r['set']=='DW'), key=lambda x: x['R2'])['R2']:.4f}")

# Save
with open('v1_results.json', 'w') as f:
    json.dump([{k:v for k,v in r.items() if k!='oof'} for r in all_results], f, indent=2)
# Also save OOF predictions for stacking
np.savez('v1_oof.npz', **{f"{r['set']}_{r['feat']}_{r['model']}": np.array(r['oof']) for r in all_results})
print("\nSaved v1_results.json and v1_oof.npz")
