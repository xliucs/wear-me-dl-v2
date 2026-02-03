"""
Figure: Deep Prediction Analysis — Predictions, calibration, subgroups, learning curve.
Lightweight: uses 5-fold CV (not 25) for OOF predictions to generate figures.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from eval_framework import load_data, get_feature_sets, engineer_all_features
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet

def fprint(*a, **k):
    print(*a, **k, flush=True)

# Load data
X_raw, y_raw, feat_names = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_raw)
X_raw_df = pd.DataFrame(X_all_raw, columns=all_cols)
X_a = engineer_all_features(X_raw_df, all_cols)
y = y_raw
log_y = np.log1p(y)
w = np.sqrt(y)

fprint(f"Data loaded: {X_a.shape[0]} samples, {X_a.shape[1]} features")

# Quick 5-fold OOF predictions
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

oof_xgb = np.zeros(len(y))
oof_lgb = np.zeros(len(y))
oof_enet = np.zeros(len(y))

xgb_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017,
    subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
    reg_alpha=2.8, reg_lambda=0.045, random_state=42, n_jobs=-1)

for fold_i, (tr, te) in enumerate(cv.split(X_a, y_binned)):
    X_tr, X_te = X_a.iloc[tr], X_a.iloc[te]
    y_tr = log_y[tr]
    w_tr = w[tr]

    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    oof_xgb[te] = np.expm1(m.predict(X_te))

    m2 = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, n_jobs=-1, verbose=-1)
    m2.fit(X_tr, y_tr, sample_weight=w_tr)
    oof_lgb[te] = np.expm1(m2.predict(X_te))

    sc = StandardScaler()
    m3 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
    m3.fit(sc.fit_transform(X_tr), y_tr, sample_weight=w_tr)
    oof_enet[te] = np.expm1(m3.predict(sc.transform(X_te)))

    fprint(f"  Fold {fold_i+1}/5 done")

# Blend
oof_blend = 0.50 * oof_xgb + 0.30 * oof_enet + 0.20 * oof_lgb
residuals = y - oof_blend
r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
fprint(f"Blend R² (5-fold): {r2:.4f}")

# ============================================================
# Create figure
# ============================================================
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Panel A: Predicted vs True
ax1 = fig.add_subplot(gs[0, 0])
hb = ax1.hexbin(y, oof_blend, gridsize=30, cmap='YlOrRd', mincnt=1)
ax1.plot([0, 15], [0, 15], 'k--', linewidth=1, alpha=0.5, label='Perfect')
ax1.set_xlabel('True HOMA-IR', fontsize=10)
ax1.set_ylabel('Predicted HOMA-IR', fontsize=10)
ax1.set_title('(A) Predicted vs True HOMA-IR', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 15); ax1.set_ylim(0, 12)
ax1.text(0.05, 0.95, f'R²={r2:.3f}\nn={len(y)}', transform=ax1.transAxes, fontsize=10, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.legend(loc='lower right', fontsize=9)
plt.colorbar(hb, ax=ax1, label='Count')

# Panel B: Residual vs True
ax2 = fig.add_subplot(gs[0, 1])
sc = ax2.scatter(y, residuals, c=np.abs(residuals), cmap='RdYlGn_r', s=15, alpha=0.6, vmin=0, vmax=5)
ax2.axhline(0, color='black', linewidth=0.5)
bins = np.arange(0, 15, 1)
bc, bm, bs = [], [], []
for i in range(len(bins)-1):
    mask = (y >= bins[i]) & (y < bins[i+1])
    if mask.sum() > 5:
        bc.append((bins[i]+bins[i+1])/2); bm.append(residuals[mask].mean()); bs.append(residuals[mask].std())
ax2.plot(bc, bm, 'ko-', linewidth=2, markersize=6, label='Binned mean', zorder=10)
ax2.fill_between(bc, np.array(bm)-np.array(bs), np.array(bm)+np.array(bs), alpha=0.15, color='black')
corr_res = np.corrcoef(y, residuals)[0, 1]
ax2.text(0.05, 0.95, f'Corr(residual, y)={corr_res:.3f}\nSystematic mean regression', transform=ax2.transAxes,
    fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.9))
ax2.set_xlabel('True HOMA-IR', fontsize=10); ax2.set_ylabel('Residual (True - Pred)', fontsize=10)
ax2.set_title('(B) Residuals Show Mean Regression', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
plt.colorbar(sc, ax=ax2, label='|Residual|')

# Panel C: Subgroup performance (Sex × BMI)
ax3 = fig.add_subplot(gs[0, 2])
sex_vals = X_raw_df['sex_num'].values if 'sex_num' in X_raw_df.columns else np.zeros(len(y))
bmi_vals = X_raw_df['bmi'].values
bmi_cat = pd.cut(bmi_vals, bins=[0, 18.5, 25, 30, 35, 100], labels=['Under', 'Normal', 'Over', 'Obese I', 'Obese II+'])
sex_label = np.where(sex_vals == 0, 'Female', 'Male')

subgroups = []
for sex in ['Female', 'Male']:
    for bmi in ['Normal', 'Over', 'Obese I', 'Obese II+']:
        mask = (sex_label == sex) & (bmi_cat == bmi)
        if mask.sum() >= 20:
            yt = y[mask]; yp = oof_blend[mask]
            r2s = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2) if np.sum((yt-yt.mean())**2)>0 else 0
            subgroups.append({'label': f'{sex}\n{bmi}', 'r2': r2s, 'n': mask.sum(), 'sex': sex})

labels_sg = [s['label'] for s in subgroups]
r2s_sg = [s['r2'] for s in subgroups]
ns_sg = [s['n'] for s in subgroups]
cols_sg = ['#E91E63' if s['sex']=='Female' else '#2196F3' for s in subgroups]
ax3.bar(range(len(subgroups)), r2s_sg, color=cols_sg, edgecolor='white')
ax3.set_xticks(range(len(subgroups))); ax3.set_xticklabels(labels_sg, fontsize=8)
for i, (r, n) in enumerate(zip(r2s_sg, ns_sg)):
    ax3.text(i, max(r, 0)+0.01, f'n={n}', ha='center', fontsize=7, color='#666')
ax3.set_ylabel('R²', fontsize=10)
ax3.set_title('(C) Performance by Sex × BMI', fontsize=12, fontweight='bold')
ax3.axhline(r2, color='black', linestyle='--', linewidth=1, alpha=0.5, label=f'Overall R²={r2:.3f}')
ax3.legend(fontsize=9); ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

# Panel D: Calibration curve
ax4 = fig.add_subplot(gs[1, 0])
n_bins = 10
bin_edges = np.percentile(oof_blend, np.linspace(0, 100, n_bins + 1))
cal_p, cal_t, cal_n = [], [], []
for i in range(n_bins):
    mask = (oof_blend >= bin_edges[i]) & (oof_blend <= bin_edges[min(i+1, n_bins)])
    if mask.sum() > 0:
        cal_p.append(oof_blend[mask].mean()); cal_t.append(y[mask].mean()); cal_n.append(mask.sum())
ax4.plot([0, 10], [0, 10], 'k--', linewidth=1, alpha=0.5, label='Perfect')
ax4.scatter(cal_p, cal_t, s=[n*2 for n in cal_n], c='#2196F3', edgecolors='black', linewidth=0.5, zorder=5)
ax4.plot(cal_p, cal_t, 'b-', linewidth=1.5, alpha=0.7)
for p, t, n in zip(cal_p, cal_t, cal_n):
    if t - p > 1.5:
        ax4.annotate(f'Under-predicts\nby {t-p:.1f}', (p, t), textcoords='offset points',
            xytext=(20, -10), fontsize=8, arrowprops=dict(arrowstyle='->', color='red'), color='red')
ax4.set_xlabel('Mean Predicted', fontsize=10); ax4.set_ylabel('Mean True', fontsize=10)
ax4.set_title('(D) Calibration Curve', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9); ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

# Panel E: Learning curve (lightweight — 3 trials, 5-fold)
ax5 = fig.add_subplot(gs[1, 1])
sample_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
lc_r2s, lc_stds = [], []
fprint("\nComputing learning curve...")
for frac in sample_fracs:
    n_samp = int(len(y) * frac)
    trials = []
    for t in range(3):
        rng = np.random.RandomState(t)
        idx = rng.choice(len(y), n_samp, replace=False)
        cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_b2 = kbd.fit_transform(y[idx].reshape(-1, 1)).ravel().astype(int)
        fr2 = []
        for tr2, te2 in cv2.split(X_a.iloc[idx], y_b2):
            m = xgb.XGBRegressor(**xgb_params)
            m.fit(X_a.iloc[idx].iloc[tr2], log_y[idx][tr2], sample_weight=w[idx][tr2])
            pred = np.expm1(m.predict(X_a.iloc[idx].iloc[te2]))
            true = y[idx][te2]
            fr2.append(1 - np.sum((true-pred)**2)/np.sum((true-true.mean())**2))
        trials.append(np.mean(fr2))
    lc_r2s.append(np.mean(trials)); lc_stds.append(np.std(trials))
    fprint(f"  {frac:.0%}: n={n_samp}, R²={np.mean(trials):.4f}")

ns_lc = [int(len(y)*f) for f in sample_fracs]
ax5.plot(ns_lc, lc_r2s, 'o-', color='#2196F3', linewidth=2, markersize=6)
ax5.fill_between(ns_lc, np.array(lc_r2s)-np.array(lc_stds), np.array(lc_r2s)+np.array(lc_stds), alpha=0.2, color='#2196F3')
ax5.axhline(0.614, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.7, label='Ceiling (0.614)')
ax5.set_xlabel('Training Samples', fontsize=10); ax5.set_ylabel('R² (5-fold CV)', fontsize=10)
ax5.set_title('(E) Learning Curve', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9); ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

# Panel F: HOMA range performance
ax6 = fig.add_subplot(gs[1, 2])
ranges = ['[0-1)', '[1-2)', '[2-3)', '[3-5)', '[5-8)', '[8+)']
bins_h = [0, 1, 2, 3, 5, 8, 100]
maes, biases, counts = [], [], []
for i in range(len(bins_h)-1):
    mask = (y >= bins_h[i]) & (y < bins_h[i+1])
    if mask.sum() > 0:
        maes.append(np.mean(np.abs(residuals[mask])))
        biases.append(np.mean(residuals[mask]))
        counts.append(mask.sum())

x_pos = np.arange(len(ranges))
w2 = 0.35
ax6.bar(x_pos - w2/2, maes, w2, color='#FF9800', label='MAE', edgecolor='white')
ax6.bar(x_pos + w2/2, biases, w2, color='#2196F3', label='Bias', edgecolor='white')
for i, (m, b, n) in enumerate(zip(maes, biases, counts)):
    ax6.text(i, max(m, b)+0.1, f'n={n}', ha='center', fontsize=7, color='#666')
ax6.axhline(0, color='black', linewidth=0.5)
ax6.set_xticks(x_pos); ax6.set_xticklabels(ranges, fontsize=9)
ax6.set_xlabel('True HOMA-IR Range', fontsize=10); ax6.set_ylabel('Error', fontsize=10)
ax6.set_title('(F) Error by HOMA-IR Range', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9); ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)

plt.suptitle('Prediction Analysis: Model Performance and Limitations', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_predictions_deep.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_predictions_deep.pdf', bbox_inches='tight')
fprint("\nDone! Saved fig_predictions_deep.png + .pdf")
