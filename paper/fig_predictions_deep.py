"""
Figure 3: Prediction Analysis — 6 panels, blue palette, square subplots.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2/paper')
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from style import *
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from eval_framework import load_data, get_feature_sets, engineer_all_features
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import ElasticNet

setup_style()

# Load data
X_raw, y_raw, feat_names = load_data()
X_all_raw, _, all_cols, _ = get_feature_sets(X_raw)
X_raw_df = pd.DataFrame(X_all_raw, columns=all_cols)
X_a = engineer_all_features(X_raw_df, all_cols)
y = y_raw
log_y = np.log1p(y)
w = np.sqrt(y)

fprint("Data loaded, running CV...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

oof_xgb = np.zeros(len(y))
oof_lgb = np.zeros(len(y))
oof_enet = np.zeros(len(y))

xgb_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017,
    subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
    reg_alpha=2.8, reg_lambda=0.045, random_state=42, n_jobs=-1)

for i, (tr, te) in enumerate(cv.split(X_a, y_binned)):
    X_tr, X_te = X_a.iloc[tr], X_a.iloc[te]
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_tr, log_y[tr], sample_weight=w[tr])
    oof_xgb[te] = np.expm1(m.predict(X_te))
    m2 = lgb.LGBMRegressor(n_estimators=768, max_depth=4, learning_rate=0.013,
        subsample=0.41, colsample_bytree=0.89, min_child_samples=36,
        num_leaves=10, random_state=42, n_jobs=-1, verbose=-1)
    m2.fit(X_tr, log_y[tr], sample_weight=w[tr])
    oof_lgb[te] = np.expm1(m2.predict(X_te))
    sc = StandardScaler()
    m3 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000, random_state=42)
    m3.fit(sc.fit_transform(X_tr), log_y[tr], sample_weight=w[tr])
    oof_enet[te] = np.expm1(m3.predict(sc.transform(X_te)))
    fprint(f"  Fold {i+1}/5")

oof_blend = 0.50 * oof_xgb + 0.30 * oof_enet + 0.20 * oof_lgb
residuals = y - oof_blend
r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
fprint(f"Blend R²: {r2:.4f}")

# ============================================================
fig = plt.figure(figsize=(12, 11))
gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)

# Panel A: Predicted vs True
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y, oof_blend, c=BLUE_MED, s=8, alpha=0.3, edgecolors='none')
ax1.plot([0, 15], [0, 15], '--', color=ACCENT_GRAY, linewidth=1, alpha=0.7)
ax1.set_xlabel('True HOMA-IR'); ax1.set_ylabel('Predicted HOMA-IR')
ax1.set_title('(a) Predicted vs True', fontsize=11)
ax1.set_xlim(0, 15); ax1.set_ylim(0, 12)
ax1.set_aspect('equal', adjustable='box')
ax1.text(0.05, 0.92, f'R²={r2:.3f}\nn={len(y)}', transform=ax1.transAxes, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor=BLUE_BG, edgecolor=BLUE_PALE, linewidth=0.5))

# Panel B: Residual vs True
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y, residuals, c=BLUE_MED, s=8, alpha=0.3, edgecolors='none')
ax2.axhline(0, color='black', linewidth=0.3)
# Binned means
bins = np.arange(0, 15, 1)
bc, bm = [], []
for j in range(len(bins)-1):
    mask = (y >= bins[j]) & (y < bins[j+1])
    if mask.sum() > 5:
        bc.append((bins[j]+bins[j+1])/2); bm.append(residuals[mask].mean())
ax2.plot(bc, bm, 'o-', color=BLUE_DARK, linewidth=2, markersize=5, zorder=10)
corr_res = np.corrcoef(y, residuals)[0, 1]
ax2.text(0.05, 0.92, f'ρ(residual, y)={corr_res:.2f}', transform=ax2.transAxes, fontsize=9,
    bbox=dict(boxstyle='round,pad=0.3', facecolor=BLUE_BG, edgecolor=BLUE_PALE, linewidth=0.5))
ax2.set_xlabel('True HOMA-IR'); ax2.set_ylabel('Residual')
ax2.set_title('(b) Mean Regression in Residuals', fontsize=11)

# Panel C: Subgroup R² (Sex × BMI)
ax3 = fig.add_subplot(gs[0, 2])
sex_vals = X_raw_df['sex_num'].values
bmi_vals = X_raw_df['bmi'].values
bmi_cat = pd.cut(bmi_vals, bins=[0, 25, 30, 35, 100], labels=['Normal', 'Over', 'Obese I', 'Obese II+'])
sex_label = np.where(sex_vals == 0, 'F', 'M')

subgroups = []
for sex in ['F', 'M']:
    for bmi in ['Normal', 'Over', 'Obese I', 'Obese II+']:
        mask = (sex_label == sex) & (bmi_cat == bmi)
        if mask.sum() >= 20:
            yt = y[mask]; yp = oof_blend[mask]
            r2s = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2) if np.var(yt)>0 else 0
            subgroups.append({'label': f'{sex}\n{bmi}', 'r2': r2s, 'n': mask.sum(), 'sex': sex})

cols_sg = [ACCENT_RED if s['sex']=='F' else BLUE_MED for s in subgroups]
ax3.bar(range(len(subgroups)), [s['r2'] for s in subgroups], color=cols_sg,
        edgecolor='white', linewidth=0.3, width=0.7)
ax3.set_xticks(range(len(subgroups)))
ax3.set_xticklabels([s['label'] for s in subgroups], fontsize=7.5)
for i, s in enumerate(subgroups):
    ax3.text(i, max(s['r2'], 0)+0.01, f'n={s["n"]}', ha='center', fontsize=6.5, color=ACCENT_GRAY)
ax3.axhline(r2, color=ACCENT_GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
ax3.set_ylabel('R²'); ax3.set_title('(c) R² by Sex × BMI', fontsize=11)
from matplotlib.patches import Patch
ax3.legend([Patch(color=ACCENT_RED), Patch(color=BLUE_MED)], ['Female', 'Male'],
           fontsize=8, loc='upper right', framealpha=0.9, edgecolor=ACCENT_LIGHT_GRAY)

# Panel D: Calibration
ax4 = fig.add_subplot(gs[1, 0])
n_bins = 10
be = np.percentile(oof_blend, np.linspace(0, 100, n_bins + 1))
cp, ct = [], []
for j in range(n_bins):
    mask = (oof_blend >= be[j]) & (oof_blend <= be[min(j+1, n_bins)])
    if mask.sum() > 0: cp.append(oof_blend[mask].mean()); ct.append(y[mask].mean())
ax4.plot([0, 10], [0, 10], '--', color=ACCENT_GRAY, linewidth=1)
ax4.scatter(cp, ct, s=40, c=BLUE_MED, edgecolors=BLUE_DARK, linewidth=0.8, zorder=5)
ax4.plot(cp, ct, '-', color=BLUE_MED, linewidth=1.5, alpha=0.7)
ax4.set_xlabel('Mean Predicted'); ax4.set_ylabel('Mean True')
ax4.set_title('(d) Calibration Curve', fontsize=11)
ax4.set_aspect('equal', adjustable='box')
ax4.set_xlim(0, 8); ax4.set_ylim(0, 8)

# Panel E: Learning curve
ax5 = fig.add_subplot(gs[1, 1])
fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
lc_r2s, lc_stds = [], []
fprint("Learning curve...")
for frac in fracs:
    n_s = int(len(y) * frac)
    trials = []
    for t in range(3):
        rng = np.random.RandomState(t)
        idx = rng.choice(len(y), n_s, replace=False)
        cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_b2 = kbd.fit_transform(y[idx].reshape(-1, 1)).ravel().astype(int)
        fr2 = []
        for tr2, te2 in cv2.split(X_a.iloc[idx], y_b2):
            m = xgb.XGBRegressor(**xgb_params)
            m.fit(X_a.iloc[idx].iloc[tr2], log_y[idx][tr2], sample_weight=w[idx][tr2])
            p = np.expm1(m.predict(X_a.iloc[idx].iloc[te2]))
            t_vals = y[idx][te2]
            fr2.append(1 - np.sum((t_vals-p)**2)/np.sum((t_vals-t_vals.mean())**2))
        trials.append(np.mean(fr2))
    lc_r2s.append(np.mean(trials)); lc_stds.append(np.std(trials))
    fprint(f"  {frac:.0%}: R²={np.mean(trials):.4f}")

ns = [int(len(y)*f) for f in fracs]
ax5.plot(ns, lc_r2s, 'o-', color=BLUE_MED, linewidth=2, markersize=5)
ax5.fill_between(ns, np.array(lc_r2s)-np.array(lc_stds),
                  np.array(lc_r2s)+np.array(lc_stds), alpha=0.15, color=BLUE_MED)
ax5.axhline(0.614, color=ACCENT_RED, linestyle='--', linewidth=1, alpha=0.7)
ax5.text(900, 0.618, 'Ceiling', fontsize=8, color=ACCENT_RED)
ax5.set_xlabel('Training Samples'); ax5.set_ylabel('R²')
ax5.set_title('(e) Learning Curve', fontsize=11)

# Panel F: Error by HOMA range
ax6 = fig.add_subplot(gs[1, 2])
ranges = ['[0,1)', '[1,2)', '[2,3)', '[3,5)', '[5,8)', '[8+)']
bins_h = [0, 1, 2, 3, 5, 8, 100]
maes, biases, counts = [], [], []
for j in range(len(bins_h)-1):
    mask = (y >= bins_h[j]) & (y < bins_h[j+1])
    if mask.sum() > 0:
        maes.append(np.mean(np.abs(residuals[mask])))
        biases.append(np.mean(residuals[mask]))
        counts.append(mask.sum())

x_pos = np.arange(len(ranges))
ax6.bar(x_pos - 0.18, maes, 0.35, color=BLUE_MED, label='MAE', edgecolor='white', linewidth=0.3)
ax6.bar(x_pos + 0.18, biases, 0.35, color=BLUE_PALE, label='Bias', edgecolor='white', linewidth=0.3)
for j, (m, n) in enumerate(zip(maes, counts)):
    ax6.text(j, m+0.1, f'n={n}', ha='center', fontsize=6.5, color=ACCENT_GRAY)
ax6.axhline(0, color='black', linewidth=0.3)
ax6.set_xticks(x_pos); ax6.set_xticklabels(ranges, fontsize=8)
ax6.set_xlabel('True HOMA-IR Range'); ax6.set_ylabel('Error')
ax6.set_title('(f) Error by HOMA-IR Range', fontsize=11)
ax6.legend(fontsize=8, framealpha=0.9, edgecolor=ACCENT_LIGHT_GRAY)

plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_predictions_deep.png', dpi=300)
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_predictions_deep.pdf')
fprint("Saved fig_predictions_deep")
