"""
Figure: Model A vs Model B + Feature Group Ablation — What information do wearables add?
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
from eval_framework import load_data, get_feature_sets, engineer_all_features, engineer_dw_features, DEMOGRAPHICS, WEARABLES, BLOOD_BIOMARKERS
import xgboost as xgb
from sklearn.linear_model import ElasticNet

def fprint(*a, **k):
    print(*a, **k, flush=True)

# Load data
X_raw, y_raw, feat_names = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_raw)
y = y_raw
log_y = np.log1p(y)
w = np.sqrt(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

xgb_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017,
    subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
    reg_alpha=2.8, reg_lambda=0.045, random_state=42, n_jobs=-1)

def eval_feature_set(X_df, name):
    """Quick 5-fold R² for a feature set."""
    oof = np.zeros(len(y))
    for tr, te in cv.split(X_df, y_binned):
        m = xgb.XGBRegressor(**xgb_params)
        m.fit(X_df.iloc[tr], log_y[tr], sample_weight=w[tr])
        oof[te] = np.expm1(m.predict(X_df.iloc[te]))
    r2 = 1 - np.sum((y - oof)**2) / np.sum((y - y.mean())**2)
    fprint(f"  {name}: R²={r2:.4f}")
    return r2, oof

# Feature group combinations
X_raw_df_all = pd.DataFrame(X_all_raw, columns=all_cols)
X_raw_df_dw = pd.DataFrame(X_dw_raw, columns=dw_cols)

# Model A: all features (engineered)
fprint("Evaluating feature group ablations...")
X_A = engineer_all_features(X_raw_df_all, all_cols)
r2_A, oof_A = eval_feature_set(X_A, "Model A (all)")

# Model B: demographics + wearables only (engineered)
X_B = engineer_dw_features(X_raw_df_dw, dw_cols)
r2_B, oof_B = eval_feature_set(X_B, "Model B (demo+wear)")

# Glucose only
X_glucose = X_raw_df_all[['glucose']].copy()
X_glucose['glucose_sq'] = X_glucose['glucose'] ** 2
X_glucose['glucose_log'] = np.log1p(X_glucose['glucose'])
r2_glucose, _ = eval_feature_set(X_glucose, "Glucose only")

# Demographics only
X_demo = X_raw_df_all[DEMOGRAPHICS].copy()
X_demo['bmi_sq'] = X_demo['bmi'] ** 2
X_demo['bmi_age'] = X_demo['bmi'] * X_demo['age']
r2_demo, _ = eval_feature_set(X_demo, "Demographics only")

# Blood only (no wearables, no demographics)
blood_cols_present = [c for c in BLOOD_BIOMARKERS if c in X_raw_df_all.columns]
X_blood = X_raw_df_all[blood_cols_present].copy()
X_blood['tyg'] = np.log(X_blood['triglycerides'].clip(1) * X_blood['glucose'].clip(1) / 2)
X_blood['trig_hdl'] = X_blood['triglycerides'] / X_blood['hdl'].clip(1)
X_blood['glucose_sq'] = X_blood['glucose'] ** 2
r2_blood, _ = eval_feature_set(X_blood, "Blood only")

# Wearables only (no demographics, no blood)
wear_cols_present = [c for c in WEARABLES if c in X_raw_df_all.columns]
X_wear = X_raw_df_all[wear_cols_present].copy()
r2_wear, _ = eval_feature_set(X_wear, "Wearables only")

# Demo + Blood (no wearables)
demo_blood_cols = [c for c in DEMOGRAPHICS + BLOOD_BIOMARKERS if c in X_raw_df_all.columns]
X_db = X_raw_df_all[demo_blood_cols].copy()
X_db['glucose_bmi'] = X_db['glucose'] * X_db['bmi']
X_db['tyg'] = np.log(X_db['triglycerides'].clip(1) * X_db['glucose'].clip(1) / 2)
X_db['trig_hdl'] = X_db['triglycerides'] / X_db['hdl'].clip(1)
X_db['ir_proxy'] = X_db['glucose'] * X_db['bmi'] * X_db['triglycerides'] / (X_db['hdl'].clip(1) * 100)
X_db['glucose_sq'] = X_db['glucose'] ** 2
X_db['bmi_sq'] = X_db['bmi'] ** 2
r2_db, oof_db = eval_feature_set(X_db, "Demo + Blood (no wearables)")

# ============================================================
# Create figure
# ============================================================
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Panel A: Feature group R² comparison (bar chart)
ax1 = fig.add_subplot(gs[0, 0])
groups = ['Wearables\nOnly', 'Demo\nOnly', 'Glucose\nOnly', 'Demo+\nWearables', 'Blood\nOnly', 'Demo+\nBlood', 'All\n(Model A)']
r2s = [r2_wear, r2_demo, r2_glucose, r2_B, r2_blood, r2_db, r2_A]
colors = ['#2E7D32', '#1565C0', '#D32F2F', '#00897B', '#F57F17', '#7B1FA2', '#333333']

bars = ax1.bar(range(len(groups)), r2s, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_xticks(range(len(groups))); ax1.set_xticklabels(groups, fontsize=9)
for i, r in enumerate(r2s):
    ax1.text(i, r + 0.005, f'{r:.3f}', ha='center', fontsize=9, fontweight='bold')
ax1.set_ylabel('R² (5-fold CV)', fontsize=10)
ax1.set_title('(A) Feature Group Contribution to HOMA-IR Prediction', fontsize=12, fontweight='bold')
ax1.axhline(0.614, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.5, label='Ceiling (0.614)')
ax1.set_ylim(0, 0.65)
ax1.legend(fontsize=9); ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B: Marginal contribution of wearables
ax2 = fig.add_subplot(gs[0, 1])
# What do wearables ADD to demographics + blood?
marginal_wear = r2_A - r2_db
# What does blood ADD to demographics + wearables?
marginal_blood = r2_A - r2_B
# What do demographics ADD to wearables?
marginal_demo_over_wear = r2_B - r2_wear

additions = ['Wearables\nadded to\nDemo+Blood', 'Blood\nadded to\nDemo+Wear', 'Demographics\nadded to\nWearables']
deltas = [marginal_wear, marginal_blood, marginal_demo_over_wear]
add_colors = ['#2E7D32', '#F57F17', '#1565C0']

bars2 = ax2.bar(range(len(additions)), deltas, color=add_colors, edgecolor='white')
for i, d in enumerate(deltas):
    ax2.text(i, d + 0.002, f'+{d:.3f}', ha='center', fontsize=10, fontweight='bold')
ax2.set_xticks(range(len(additions))); ax2.set_xticklabels(additions, fontsize=9)
ax2.set_ylabel('ΔR² (marginal gain)', fontsize=10)
ax2.set_title('(B) Marginal Information Contribution', fontsize=12, fontweight='bold')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# Panel C: Model A vs Model B predictions scatter
ax3 = fig.add_subplot(gs[1, 0])
sc = ax3.scatter(oof_A, oof_B, c=y, cmap='RdYlBu_r', s=15, alpha=0.6, vmin=0, vmax=10)
ax3.plot([0, 12], [0, 12], 'k--', linewidth=1, alpha=0.3)
ax3.set_xlabel(f'Model A Prediction (R²={r2_A:.3f})', fontsize=10)
ax3.set_ylabel(f'Model B Prediction (R²={r2_B:.3f})', fontsize=10)
ax3.set_title('(C) Model A vs Model B Predictions', fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=ax3, label='True HOMA-IR')
ax3.text(0.05, 0.95, f'Correlation: {np.corrcoef(oof_A, oof_B)[0,1]:.3f}',
         transform=ax3.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel D: Information hierarchy diagram
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

# Create a visual hierarchy
levels = [
    ('HOMA-IR = Glucose × Insulin / 405', 0.92, '#333333', 14),
    ('', 0.85, '', 0),
    ('Insulin (r=0.97) — NOT AVAILABLE', 0.78, '#D32F2F', 12),
    ('↓ Partial signal from proxy features', 0.72, '#666', 10),
    ('', 0.66, '', 0),
    (f'Wearables only: R²={r2_wear:.3f}', 0.60, '#2E7D32', 11),
    (f'Demographics only: R²={r2_demo:.3f}', 0.54, '#1565C0', 11),
    (f'Blood biomarkers only: R²={r2_blood:.3f}', 0.48, '#F57F17', 11),
    (f'All features combined: R²={r2_A:.3f}', 0.40, '#333333', 12),
    ('', 0.32, '', 0),
    (f'Theoretical ceiling: R²=0.614', 0.25, '#D32F2F', 12),
    (f'Remaining gap: {0.614-r2_A:.3f} (within estimation error)', 0.18, '#666', 10),
]

for text, y_pos, color, fontsize in levels:
    if text:
        ax4.text(0.5, y_pos, text, ha='center', va='center', fontsize=fontsize,
                color=color, fontweight='bold' if fontsize >= 12 else 'normal',
                transform=ax4.transAxes)

ax4.set_title('(D) Information Hierarchy for HOMA-IR', fontsize=12, fontweight='bold')

# Add boxes around key sections
from matplotlib.patches import FancyBboxPatch
rect1 = FancyBboxPatch((0.05, 0.35), 0.9, 0.32, boxstyle='round,pad=0.02',
                         facecolor='#E3F2FD', edgecolor='#1565C0', alpha=0.3,
                         transform=ax4.transAxes)
ax4.add_patch(rect1)

plt.suptitle('What Can Wearables Tell Us About Insulin Resistance?',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_model_comparison.pdf', bbox_inches='tight')
fprint("\nDone! Saved fig_model_comparison.png + .pdf")
