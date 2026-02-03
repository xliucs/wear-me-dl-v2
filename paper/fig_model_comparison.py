"""
Figure 5: Model A vs Model B + Feature Group Ablation, blue palette.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2/paper')
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from style import *
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from eval_framework import (load_data, get_feature_sets, engineer_all_features,
                            engineer_dw_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS)
import xgboost as xgb

setup_style()

X_raw, y_raw, feat_names = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_raw)
y = y_raw; log_y = np.log1p(y); w = np.sqrt(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

xgb_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017,
    subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
    reg_alpha=2.8, reg_lambda=0.045, random_state=42, n_jobs=-1)

def eval_fs(X_df, name):
    oof = np.zeros(len(y))
    for tr, te in cv.split(X_df, y_binned):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_df.iloc[tr], log_y[tr], sample_weight=w[tr])
        oof[te] = np.expm1(m.predict(X_df.iloc[te]))
    r2 = 1 - np.sum((y - oof)**2) / np.sum((y - y.mean())**2)
    fprint(f"  {name}: R²={r2:.4f}")
    return r2, oof

fprint("Feature group ablation...")
X_raw_df_all = pd.DataFrame(X_all_raw, columns=all_cols)
X_raw_df_dw = pd.DataFrame(X_dw_raw, columns=dw_cols)

X_A = engineer_all_features(X_raw_df_all, all_cols)
r2_A, oof_A = eval_fs(X_A, "All (Model A)")
X_B = engineer_dw_features(X_raw_df_dw, dw_cols)
r2_B, oof_B = eval_fs(X_B, "Demo+Wear (Model B)")

X_g = X_raw_df_all[['glucose']].copy()
X_g['glucose_sq'] = X_g['glucose'] ** 2
r2_g, _ = eval_fs(X_g, "Glucose only")

X_d = X_raw_df_all[DEMOGRAPHICS].copy()
X_d['bmi_sq'] = X_d['bmi'] ** 2
r2_d, _ = eval_fs(X_d, "Demo only")

bc = [c for c in BLOOD_BIOMARKERS if c in X_raw_df_all.columns]
X_bl = X_raw_df_all[bc].copy()
X_bl['tyg'] = np.log(X_bl['triglycerides'].clip(1) * X_bl['glucose'].clip(1) / 2)
X_bl['trig_hdl'] = X_bl['triglycerides'] / X_bl['hdl'].clip(1)
r2_bl, _ = eval_fs(X_bl, "Blood only")

wc = [c for c in WEARABLES if c in X_raw_df_all.columns]
X_w = X_raw_df_all[wc].copy()
r2_w, _ = eval_fs(X_w, "Wear only")

dbc = [c for c in DEMOGRAPHICS + BLOOD_BIOMARKERS if c in X_raw_df_all.columns]
X_db = X_raw_df_all[dbc].copy()
X_db['glucose_bmi'] = X_db['glucose'] * X_db['bmi']
X_db['tyg'] = np.log(X_db['triglycerides'].clip(1) * X_db['glucose'].clip(1) / 2)
X_db['ir_proxy'] = X_db['glucose'] * X_db['bmi'] * X_db['triglycerides'] / (X_db['hdl'].clip(1) * 100)
r2_db, oof_db = eval_fs(X_db, "Demo+Blood")

# ============================================================
fig = plt.figure(figsize=(12, 5.5))
gs = gridspec.GridSpec(1, 3, wspace=0.35)

# Panel A: Feature group R²
ax1 = fig.add_subplot(gs[0, 0])
groups = ['Wear\nonly', 'Demo\nonly', 'Glucose\nonly', 'Demo+\nWear', 'Blood\nonly', 'Demo+\nBlood', 'All']
r2s = [r2_w, r2_d, r2_g, r2_B, r2_bl, r2_db, r2_A]
blues = [BLUES_SEQ[i] for i in range(len(groups))]

ax1.bar(range(len(groups)), r2s, color=blues, edgecolor='white', linewidth=0.5, width=0.7)
for i, r in enumerate(r2s):
    ax1.text(i, r + 0.008, f'{r:.3f}', ha='center', fontsize=8, fontweight='bold', color=BLUE_DARK)
ax1.axhline(0.614, color=ACCENT_RED, linestyle='--', linewidth=1, alpha=0.5)
ax1.text(6.5, 0.62, 'Ceiling', fontsize=7.5, color=ACCENT_RED)
ax1.set_xticks(range(len(groups))); ax1.set_xticklabels(groups, fontsize=8)
ax1.set_ylabel('R²', fontsize=10)
ax1.set_title('(a) Feature Group Contribution', fontsize=11)
ax1.set_ylim(0, 0.68)

# Panel B: Marginal contribution
ax2 = fig.add_subplot(gs[0, 1])
marg_w = r2_A - r2_db
marg_b = r2_A - r2_B
labels_m = ['Wearables\nadded to\nDemo+Blood', 'Blood\nadded to\nDemo+Wear']
deltas = [marg_w, marg_b]
ax2.bar(range(2), deltas, color=[BLUE_LIGHT, BLUE_DARK], edgecolor='white',
        linewidth=0.5, width=0.5)
for i, d in enumerate(deltas):
    ax2.text(i, d + 0.005, f'+{d:.3f}', ha='center', fontsize=10, fontweight='bold', color=BLUE_DARK)
ax2.set_xticks(range(2)); ax2.set_xticklabels(labels_m, fontsize=9)
ax2.set_ylabel('ΔR²', fontsize=10)
ax2.set_title('(b) Marginal Contribution', fontsize=11)
# Add ratio annotation
ratio = marg_b / marg_w if marg_w > 0 else float('inf')
ax2.text(0.5, max(deltas)*0.6, f'{ratio:.0f}× more\ninformation\nfrom blood', ha='center',
         fontsize=10, color=ACCENT_RED, fontweight='bold', transform=ax2.get_xaxis_transform())

# Panel C: Model A vs B scatter
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(oof_A, oof_B, c=BLUE_MED, s=8, alpha=0.3, edgecolors='none')
ax3.plot([0, 12], [0, 12], '--', color=ACCENT_GRAY, linewidth=1)
ax3.set_xlabel(f'Model A (R²={r2_A:.3f})', fontsize=10)
ax3.set_ylabel(f'Model B (R²={r2_B:.3f})', fontsize=10)
ax3.set_title('(c) Model A vs B Predictions', fontsize=11)
ax3.set_xlim(0, 12); ax3.set_ylim(0, 12)
ax3.set_aspect('equal', adjustable='box')
rho = np.corrcoef(oof_A, oof_B)[0, 1]
ax3.text(0.05, 0.92, f'ρ = {rho:.3f}', transform=ax3.transAxes, fontsize=10,
    bbox=dict(boxstyle='round,pad=0.3', facecolor=BLUE_BG, edgecolor=BLUE_PALE, linewidth=0.5))

plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_model_comparison.png', dpi=300)
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_model_comparison.pdf')
fprint("Saved fig_model_comparison")
