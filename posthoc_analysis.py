#!/usr/bin/env python3
"""
Post-hoc Analysis: Understanding the data and model behavior deeply.

1. t-SNE / UMAP visualization of feature space colored by HOMA-IR
2. t-SNE colored by prediction error — where in feature space do we fail?
3. Subgroup analysis: sex × BMI × glucose interactions
4. Feature importance stability across folds
5. Prediction vs truth scatter with density
6. Residual distribution by predicted quantile
7. Learning curves — do we need more data?
8. Correlation heatmap of top features with HOMA
"""
import numpy as np, pandas as pd, time, warnings, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from eval_framework import (load_data, get_feature_sets, get_cv_splits)
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import learning_curve, StratifiedKFold
import xgboost as xgb

t_start = time.time()
print("="*60)
print("  POST-HOC ANALYSIS")
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
    return X.fillna(0)

X_v7_df = eng_v7(X_df, all_cols)
X_v7 = X_v7_df.values
feat_names = list(X_v7_df.columns)

# Get OOF predictions
print("Computing OOF predictions...")
sys.stdout.flush()
oof_sum = np.zeros(n)
oof_count = np.zeros(n)
feat_imp_all = np.zeros(len(feat_names))
for tr, te in splits:
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(X_v7[tr], y_log[tr], sample_weight=w_sqrt[tr])
    oof_sum[te] += m.predict(X_v7[te])
    oof_count[te] += 1
    feat_imp_all += m.feature_importances_

oof = np.expm1(oof_sum / np.clip(oof_count, 1, None))
residuals = y - oof
abs_errors = np.abs(residuals)
feat_imp_all /= len(splits)

print(f"OOF R² = {r2_score(y, oof):.4f}")

# ============================================================
# FIGURE 1: t-SNE of feature space
# ============================================================
print("\nComputing t-SNE...")
sys.stdout.flush()

X_scaled = StandardScaler().fit_transform(X_v7)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle('Post-hoc Analysis: t-SNE Feature Space Visualization', fontsize=16, fontweight='bold')

# 1a: t-SNE colored by HOMA-IR
ax = axes[0, 0]
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.log1p(y), cmap='RdYlBu_r', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='log(1+HOMA_IR)')
ax.set_title('t-SNE colored by HOMA-IR (log scale)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# 1b: t-SNE colored by absolute prediction error
ax = axes[0, 1]
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(abs_errors, 0, 5), cmap='hot_r', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='|Prediction Error|')
ax.set_title('t-SNE colored by |Error| (clipped at 5)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# 1c: t-SNE colored by signed residual
ax = axes[0, 2]
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(residuals, -5, 5), cmap='RdBu_r', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='Residual (true - pred)')
ax.set_title('t-SNE colored by Residual (blue=over-pred, red=under-pred)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# 1d: t-SNE colored by glucose
ax = axes[1, 0]
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=X_df['glucose'].values, cmap='viridis', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='Glucose')
ax.set_title('t-SNE colored by Glucose')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# 1e: t-SNE colored by BMI
ax = axes[1, 1]
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=X_df['bmi'].values, cmap='plasma', s=8, alpha=0.6)
plt.colorbar(sc, ax=ax, label='BMI')
ax.set_title('t-SNE colored by BMI')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# 1f: t-SNE colored by sex
ax = axes[1, 2]
colors = ['tab:blue' if s == 0 else 'tab:red' for s in X_df['sex_num'].values]
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=8, alpha=0.5)
ax.set_title('t-SNE colored by Sex (blue=F, red=M)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('posthoc_fig1_tsne.png', dpi=150, bbox_inches='tight')
print("Saved posthoc_fig1_tsne.png")
plt.close()
sys.stdout.flush()

# ============================================================
# FIGURE 2: Prediction analysis
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle('Post-hoc Analysis: Prediction Quality', fontsize=16, fontweight='bold')

# 2a: Predicted vs True scatter
ax = axes[0, 0]
ax.scatter(y, oof, s=8, alpha=0.4, c='steelblue')
ax.plot([0, 15], [0, 15], 'r--', lw=2, label='Perfect')
ax.set_xlabel('True HOMA-IR'); ax.set_ylabel('Predicted HOMA-IR')
ax.set_title(f'Predicted vs True (R²={r2_score(y, oof):.4f})')
ax.set_xlim(0, 16); ax.set_ylim(0, 16)
ax.legend()

# 2b: Residual vs True
ax = axes[0, 1]
ax.scatter(y, residuals, s=8, alpha=0.4, c='steelblue')
ax.axhline(0, color='r', linestyle='--', lw=2)
ax.set_xlabel('True HOMA-IR'); ax.set_ylabel('Residual (True - Pred)')
ax.set_title(f'Residual vs True (corr={np.corrcoef(y, residuals)[0,1]:.3f})')

# 2c: Residual histogram
ax = axes[0, 2]
ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='r', linestyle='--', lw=2)
ax.set_xlabel('Residual'); ax.set_ylabel('Count')
ax.set_title(f'Residual Distribution (skew={pd.Series(residuals).skew():.2f})')

# 2d: Error by HOMA range (box plot style)
ax = axes[1, 0]
ranges = [(0, 1, '<1'), (1, 2, '1-2'), (2, 3, '2-3'), (3, 5, '3-5'), (5, 8, '5-8'), (8, 16, '8+')]
positions, data, labels = [], [], []
for i, (lo, hi, label) in enumerate(ranges):
    mask = (y >= lo) & (y < hi)
    if mask.sum() > 0:
        data.append(residuals[mask])
        positions.append(i)
        labels.append(f'{label}\n(n={mask.sum()})')
bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.6)
ax.axhline(0, color='r', linestyle='--', lw=1)
ax.set_xlabel('HOMA-IR Range'); ax.set_ylabel('Residual')
ax.set_title('Residual Distribution by HOMA Range')

# 2e: Subgroup R² heatmap: BMI bins × Glucose bins
ax = axes[1, 1]
bmi_bins = [(0, 25, 'BMI<25'), (25, 30, '25-30'), (30, 35, '30-35'), (35, 100, '35+')]
gluc_bins = [(0, 90, 'Gluc<90'), (90, 100, '90-100'), (100, 126, '100-126'), (126, 300, '126+')]
heatmap = np.full((len(bmi_bins), len(gluc_bins)), np.nan)
annot = np.empty((len(bmi_bins), len(gluc_bins)), dtype=object)
for i, (blo, bhi, _) in enumerate(bmi_bins):
    for j, (glo, ghi, _) in enumerate(gluc_bins):
        mask = (X_df['bmi'].values >= blo) & (X_df['bmi'].values < bhi) & \
               (X_df['glucose'].values >= glo) & (X_df['glucose'].values < ghi)
        cnt = mask.sum()
        if cnt >= 10 and np.std(y[mask]) > 0:
            r2 = r2_score(y[mask], oof[mask])
            heatmap[i, j] = r2
            annot[i, j] = f'{r2:.2f}\n(n={cnt})'
        else:
            annot[i, j] = f'n={cnt}'

im = ax.imshow(heatmap, cmap='RdYlGn', vmin=-0.5, vmax=0.8, aspect='auto')
plt.colorbar(im, ax=ax, label='R²')
ax.set_xticks(range(len(gluc_bins)))
ax.set_xticklabels([g[2] for g in gluc_bins])
ax.set_yticks(range(len(bmi_bins)))
ax.set_yticklabels([b[2] for b in bmi_bins])
for i in range(len(bmi_bins)):
    for j in range(len(gluc_bins)):
        ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=8)
ax.set_xlabel('Glucose'); ax.set_ylabel('BMI')
ax.set_title('Subgroup R²: BMI × Glucose')

# 2f: Prediction std vs true std analysis
ax = axes[1, 2]
# Bin by predicted value, show true y distribution
pred_bins = np.percentile(oof, np.arange(0, 101, 10))
bin_centers, true_means, true_stds, pred_means = [], [], [], []
for i in range(len(pred_bins)-1):
    mask = (oof >= pred_bins[i]) & (oof < pred_bins[i+1] + 0.001)
    if mask.sum() > 5:
        bin_centers.append((pred_bins[i] + pred_bins[i+1])/2)
        true_means.append(np.mean(y[mask]))
        true_stds.append(np.std(y[mask]))
        pred_means.append(np.mean(oof[mask]))
ax.errorbar(bin_centers, true_means, yerr=true_stds, fmt='o-', color='steelblue', 
            label='True y (mean ± std)', capsize=3)
ax.plot([0, 10], [0, 10], 'r--', lw=1, label='Perfect calibration')
ax.set_xlabel('Predicted HOMA-IR (binned)'); ax.set_ylabel('True HOMA-IR')
ax.set_title('Calibration: True vs Predicted (binned)')
ax.legend()

plt.tight_layout()
plt.savefig('posthoc_fig2_predictions.png', dpi=150, bbox_inches='tight')
print("Saved posthoc_fig2_predictions.png")
plt.close()
sys.stdout.flush()

# ============================================================
# FIGURE 3: Feature importance + correlations
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Post-hoc Analysis: Features & Correlations', fontsize=16, fontweight='bold')

# 3a: Top 20 feature importances
ax = axes[0, 0]
top_idx = np.argsort(-feat_imp_all)[:20]
top_names = [feat_names[i] for i in top_idx]
top_imp = feat_imp_all[top_idx]
ax.barh(range(len(top_names)), top_imp[::-1], color='steelblue')
ax.set_yticks(range(len(top_names)))
ax.set_yticklabels(top_names[::-1], fontsize=8)
ax.set_xlabel('Mean Feature Importance (gain)')
ax.set_title('Top 20 Feature Importances (averaged over 25 folds)')

# 3b: Feature importance stability (std across folds)
ax = axes[0, 1]
# Compute per-fold importances
fold_imps = []
for tr, te in splits:
    m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
        subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
        reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
    m.fit(X_v7[tr], y_log[tr], sample_weight=w_sqrt[tr])
    fold_imps.append(m.feature_importances_)
fold_imps = np.array(fold_imps)
imp_mean = fold_imps.mean(axis=0)
imp_std = fold_imps.std(axis=0)
# Top 20 by mean
top_idx2 = np.argsort(-imp_mean)[:20]
ax.barh(range(20), imp_mean[top_idx2][::-1], xerr=imp_std[top_idx2][::-1],
        color='steelblue', alpha=0.7, capsize=3)
ax.set_yticks(range(20))
ax.set_yticklabels([feat_names[i] for i in top_idx2][::-1], fontsize=8)
ax.set_xlabel('Feature Importance (mean ± std across folds)')
ax.set_title('Feature Importance Stability')

# 3c: Correlation of raw features with HOMA and with residuals
ax = axes[1, 0]
raw_feats = all_cols
corr_homa = [np.corrcoef(X_df[f].values, y)[0,1] for f in raw_feats]
corr_resid = [np.corrcoef(X_df[f].values, residuals)[0,1] for f in raw_feats]
x_pos = np.arange(len(raw_feats))
width = 0.35
ax.bar(x_pos - width/2, corr_homa, width, label='Corr with HOMA', color='steelblue', alpha=0.7)
ax.bar(x_pos + width/2, corr_resid, width, label='Corr with Residual', color='coral', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(raw_feats, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Pearson Correlation')
ax.set_title('Raw Feature Correlations: HOMA vs Residuals')
ax.legend()
ax.axhline(0, color='gray', linestyle='-', lw=0.5)

# 3d: Learning curve
ax = axes[1, 1]
print("Computing learning curve...")
sys.stdout.flush()
# Use stratified bins for stratification
y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
train_sizes_abs = (train_sizes * n * 0.8).astype(int)  # 80% train in 5-fold

# Manual learning curve
lc_train_r2 = []
lc_test_r2 = []
lc_sizes = []
cv5 = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_v7, y_bins))
for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    train_r2s, test_r2s = [], []
    for tr, te in cv5:
        rng = np.random.default_rng(42)
        n_use = max(10, int(frac * len(tr)))
        sub = rng.choice(len(tr), n_use, replace=False)
        tr_sub = tr[sub]
        w_sub = w_sqrt[tr_sub]
        m = xgb.XGBRegressor(n_estimators=612, max_depth=4, learning_rate=0.017,
            subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
            reg_alpha=2.8, reg_lambda=0.045, random_state=42, verbosity=0)
        m.fit(X_v7[tr_sub], y_log[tr_sub], sample_weight=w_sub)
        train_pred = np.expm1(m.predict(X_v7[tr_sub]))
        test_pred = np.expm1(m.predict(X_v7[te]))
        train_r2s.append(r2_score(y[tr_sub], train_pred))
        test_r2s.append(r2_score(y[te], test_pred))
    lc_sizes.append(int(frac * len(cv5[0][0])))
    lc_train_r2.append((np.mean(train_r2s), np.std(train_r2s)))
    lc_test_r2.append((np.mean(test_r2s), np.std(test_r2s)))

sizes = lc_sizes
train_mean = [x[0] for x in lc_train_r2]
train_std = [x[1] for x in lc_train_r2]
test_mean = [x[0] for x in lc_test_r2]
test_std = [x[1] for x in lc_test_r2]

ax.fill_between(sizes, [m-s for m,s in zip(train_mean, train_std)],
                [m+s for m,s in zip(train_mean, train_std)], alpha=0.1, color='steelblue')
ax.fill_between(sizes, [m-s for m,s in zip(test_mean, test_std)],
                [m+s for m,s in zip(test_mean, test_std)], alpha=0.1, color='coral')
ax.plot(sizes, train_mean, 'o-', color='steelblue', label='Train R²')
ax.plot(sizes, test_mean, 'o-', color='coral', label='Test R²')
ax.set_xlabel('Training Set Size')
ax.set_ylabel('R²')
ax.set_title('Learning Curve (XGB Optuna)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('posthoc_fig3_features.png', dpi=150, bbox_inches='tight')
print("Saved posthoc_fig3_features.png")
plt.close()
sys.stdout.flush()

# ============================================================
# FIGURE 4: Deep subgroup analysis
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle('Post-hoc Analysis: Subgroup Deep Dive', fontsize=16, fontweight='bold')

glucose = X_df['glucose'].values
bmi = X_df['bmi'].values
sex = X_df['sex_num'].values
trig = X_df['triglycerides'].values
hdl = X_df['hdl'].values
rhr = X_df['Resting Heart Rate (mean)'].values

# 4a: R² by sex × BMI category
ax = axes[0, 0]
bmi_cats = [('Normal\n<25', 0, 25), ('Overweight\n25-30', 25, 30), 
            ('Obese I\n30-35', 30, 35), ('Obese II+\n35+', 35, 100)]
x_pos = np.arange(len(bmi_cats))
width = 0.35
for sex_val, sex_label, color, offset in [(0, 'Female', 'steelblue', -width/2), 
                                            (1, 'Male', 'coral', width/2)]:
    r2s = []
    ns = []
    for _, blo, bhi in bmi_cats:
        mask = (sex == sex_val) & (bmi >= blo) & (bmi < bhi)
        cnt = mask.sum()
        ns.append(cnt)
        if cnt >= 10 and np.std(y[mask]) > 0:
            r2s.append(r2_score(y[mask], oof[mask]))
        else:
            r2s.append(0)
    bars = ax.bar(x_pos + offset, r2s, width, label=f'{sex_label}', color=color, alpha=0.7)
    for bar, n_val in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'n={n_val}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x_pos)
ax.set_xticklabels([b[0] for b in bmi_cats])
ax.set_ylabel('R²'); ax.set_title('R² by Sex × BMI Category')
ax.legend()

# 4b: MAE by sex × glucose category
ax = axes[0, 1]
gluc_cats = [('Normal\n<90', 0, 90), ('Borderline\n90-100', 90, 100), 
             ('Pre-DM\n100-126', 100, 126), ('DM\n126+', 126, 300)]
x_pos = np.arange(len(gluc_cats))
for sex_val, sex_label, color, offset in [(0, 'Female', 'steelblue', -width/2), 
                                            (1, 'Male', 'coral', width/2)]:
    maes = []
    ns = []
    for _, glo, ghi in gluc_cats:
        mask = (sex == sex_val) & (glucose >= glo) & (glucose < ghi)
        cnt = mask.sum()
        ns.append(cnt)
        if cnt >= 5:
            maes.append(mean_absolute_error(y[mask], oof[mask]))
        else:
            maes.append(0)
    bars = ax.bar(x_pos + offset, maes, width, label=f'{sex_label}', color=color, alpha=0.7)
    for bar, n_val in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={n_val}', ha='center', va='bottom', fontsize=7)
ax.set_xticks(x_pos)
ax.set_xticklabels([g[0] for g in gluc_cats])
ax.set_ylabel('MAE'); ax.set_title('MAE by Sex × Glucose Category')
ax.legend()

# 4c: Scatter of glucose vs HOMA, colored by prediction quality
ax = axes[0, 2]
good_mask = abs_errors < np.percentile(abs_errors, 50)
bad_mask = abs_errors >= np.percentile(abs_errors, 90)
ax.scatter(glucose[good_mask], y[good_mask], s=10, alpha=0.3, c='green', label=f'Good (|err|<P50)')
ax.scatter(glucose[~good_mask & ~bad_mask], y[~good_mask & ~bad_mask], s=10, alpha=0.3, c='gray', label='Middle')
ax.scatter(glucose[bad_mask], y[bad_mask], s=20, alpha=0.6, c='red', label=f'Bad (|err|>P90)')
ax.set_xlabel('Glucose'); ax.set_ylabel('HOMA-IR')
ax.set_title('Glucose vs HOMA: Good vs Bad Predictions')
ax.legend(fontsize=8)

# 4d: BMI vs HOMA, colored by prediction quality
ax = axes[1, 0]
ax.scatter(bmi[good_mask], y[good_mask], s=10, alpha=0.3, c='green', label='Good')
ax.scatter(bmi[~good_mask & ~bad_mask], y[~good_mask & ~bad_mask], s=10, alpha=0.3, c='gray')
ax.scatter(bmi[bad_mask], y[bad_mask], s=20, alpha=0.6, c='red', label='Bad (|err|>P90)')
ax.set_xlabel('BMI'); ax.set_ylabel('HOMA-IR')
ax.set_title('BMI vs HOMA: Good vs Bad Predictions')
ax.legend(fontsize=8)

# 4e: Trig/HDL ratio vs HOMA
ax = axes[1, 1]
trig_hdl = trig / hdl
ax.scatter(trig_hdl[good_mask], y[good_mask], s=10, alpha=0.3, c='green', label='Good')
ax.scatter(trig_hdl[~good_mask & ~bad_mask], y[~good_mask & ~bad_mask], s=10, alpha=0.3, c='gray')
ax.scatter(trig_hdl[bad_mask], y[bad_mask], s=20, alpha=0.6, c='red', label='Bad')
ax.set_xlabel('Triglycerides/HDL'); ax.set_ylabel('HOMA-IR')
ax.set_title('Trig/HDL vs HOMA: Good vs Bad Predictions')
ax.set_xlim(0, 15)
ax.legend(fontsize=8)

# 4f: RHR vs HOMA
ax = axes[1, 2]
ax.scatter(rhr[good_mask], y[good_mask], s=10, alpha=0.3, c='green', label='Good')
ax.scatter(rhr[~good_mask & ~bad_mask], y[~good_mask & ~bad_mask], s=10, alpha=0.3, c='gray')
ax.scatter(rhr[bad_mask], y[bad_mask], s=20, alpha=0.6, c='red', label='Bad')
ax.set_xlabel('Resting Heart Rate'); ax.set_ylabel('HOMA-IR')
ax.set_title('RHR vs HOMA: Good vs Bad Predictions')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('posthoc_fig4_subgroups.png', dpi=150, bbox_inches='tight')
print("Saved posthoc_fig4_subgroups.png")
plt.close()
sys.stdout.flush()

# ============================================================
# TEXT SUMMARY
# ============================================================
print("\n" + "="*60)
print("  QUANTITATIVE SUBGROUP SUMMARY")
print("="*60)

print("\n  --- Sex × BMI × Glucose Interaction ---")
for sex_val, sex_label in [(0, 'Female'), (1, 'Male')]:
    for bmi_label, blo, bhi in [('Normal', 0, 25), ('Overweight', 25, 30), ('Obese', 30, 100)]:
        for gluc_label, glo, ghi in [('Normal', 0, 100), ('Pre-DM', 100, 126), ('DM', 126, 300)]:
            mask = (sex == sex_val) & (bmi >= blo) & (bmi < bhi) & (glucose >= glo) & (glucose < ghi)
            cnt = mask.sum()
            if cnt >= 10:
                r2 = r2_score(y[mask], oof[mask]) if np.std(y[mask]) > 0 else float('nan')
                mae = mean_absolute_error(y[mask], oof[mask])
                bias = np.mean(residuals[mask])
                print(f"    {sex_label:6s} {bmi_label:10s} {gluc_label:6s}: "
                      f"n={cnt:4d}, R²={r2:+.3f}, MAE={mae:.3f}, bias={bias:+.3f}")

# Wearable-only subgroup analysis
print("\n  --- Wearable Feature Subgroups ---")
wearable_groups = [
    ('RHR', rhr, [(0,60,'Low'), (60,70,'Normal'), (70,80,'Elevated'), (80,100,'High')]),
    ('Steps', X_df['STEPS (mean)'].values.astype(float), [(0,5000,'Sedentary'), (5000,8000,'Low'), (8000,12000,'Active'), (12000,30000,'Very Active')]),
    ('HRV', X_df['HRV (mean)'].values.astype(float), [(0,25,'Low'), (25,40,'Normal'), (40,60,'Good'), (60,200,'Excellent')]),
]
for feat_name, feat_vals, cuts in wearable_groups:
    print(f"\n  {feat_name}:")
    for label, lo, hi in cuts:
        mask = (feat_vals >= lo) & (feat_vals < hi)
        cnt = mask.sum()
        if cnt >= 15:
            r2 = r2_score(y[mask], oof[mask]) if np.std(y[mask]) > 0 else float('nan')
            mae = mean_absolute_error(y[mask], oof[mask])
            print(f"    {label:15s}: n={cnt:4d}, R²={r2:.3f}, MAE={mae:.3f}, "
                  f"mean_HOMA={np.mean(y[mask]):.2f}")

elapsed = time.time() - t_start
print(f"\n  Completed in {elapsed:.1f}s")
print(f"  Generated: posthoc_fig1_tsne.png, posthoc_fig2_predictions.png, "
      f"posthoc_fig3_features.png, posthoc_fig4_subgroups.png")
