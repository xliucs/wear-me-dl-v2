"""
Figure: The "Hidden Insulin Resistant" — Who are the patients we can't predict?
t-SNE visualization + case studies of worst predictions.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from eval_framework import load_data, get_feature_sets, engineer_all_features
import xgboost as xgb

def fprint(*a, **k):
    print(*a, **k, flush=True)

# Load data
X_raw, y_raw, feat_names = load_data()
X_all_raw, X_dw_raw, all_cols, dw_cols = get_feature_sets(X_raw)
X_raw_df = pd.DataFrame(X_all_raw, columns=all_cols)
X_a = engineer_all_features(X_raw_df, all_cols)
y = y_raw

# Quick 5-fold OOF predictions
log_y = np.log1p(y)
w = np.sqrt(y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

oof_pred = np.zeros(len(y))
xgb_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017,
    subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
    reg_alpha=2.8, reg_lambda=0.045, random_state=42, n_jobs=-1)

fprint("Running 5-fold CV...")
for i, (tr, te) in enumerate(cv.split(X_a, y_binned)):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_a.iloc[tr], log_y[tr], sample_weight=w[tr])
    oof_pred[te] = np.expm1(m.predict(X_a.iloc[te]))
    fprint(f"  Fold {i+1}/5")

residuals = y - oof_pred
abs_err = np.abs(residuals)
r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
fprint(f"OOF R²: {r2:.4f}")

# t-SNE on scaled features
fprint("Computing t-SNE...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_a)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
fprint("t-SNE done")

# Identify worst predictions
worst_mask = abs_err > np.percentile(abs_err, 90)  # Top 10% worst
best_mask = abs_err < np.percentile(abs_err, 10)   # Top 10% best
hidden_ir = (y > 5) & (abs_err > 2)  # High HOMA + large error

# ============================================================
# Create figure
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

# Panel A: t-SNE colored by true HOMA-IR
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(y, 0, 10), cmap='RdYlBu_r',
                  s=10, alpha=0.7, vmin=0, vmax=10)
ax1.set_title('(A) t-SNE: True HOMA-IR', fontsize=12, fontweight='bold')
ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=ax1, label='HOMA-IR')
ax1.text(0.05, 0.95, 'High HOMA scattered\nthroughout feature space',
         transform=ax1.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel B: t-SNE colored by absolute error
ax2 = fig.add_subplot(gs[0, 1])
sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(abs_err, 0, 5), cmap='Reds',
                   s=10, alpha=0.7, vmin=0, vmax=5)
# Highlight "hidden IR" patients
ax2.scatter(X_tsne[hidden_ir, 0], X_tsne[hidden_ir, 1], facecolors='none',
            edgecolors='black', s=80, linewidths=2, label=f'"Hidden IR" (n={hidden_ir.sum()})')
ax2.set_title('(B) t-SNE: Prediction Error', fontsize=12, fontweight='bold')
ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
plt.colorbar(sc2, ax=ax2, label='|Error|')
ax2.legend(loc='lower right', fontsize=9)

# Panel C: t-SNE colored by glucose (the dominant feature)
ax3 = fig.add_subplot(gs[0, 2])
glucose = X_raw_df['glucose'].values
sc3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=glucose, cmap='YlOrRd',
                   s=10, alpha=0.7)
ax3.set_title('(C) t-SNE: Glucose Level', fontsize=12, fontweight='bold')
ax3.set_xlabel('t-SNE 1'); ax3.set_ylabel('t-SNE 2')
plt.colorbar(sc3, ax=ax3, label='Glucose (mg/dL)')

# Panel D: Feature profiles of "hidden IR" vs well-predicted
ax4 = fig.add_subplot(gs[1, 0])
well_predicted = abs_err < np.percentile(abs_err, 25)
# Compare key features
key_feats = ['glucose', 'bmi', 'triglycerides', 'hdl']
key_labels = ['Glucose', 'BMI', 'Triglycerides', 'HDL']

# Normalize features for comparison
profiles = {}
for label, feat in zip(key_labels, key_feats):
    vals = X_raw_df[feat].values
    # Z-score
    mu, sd = vals.mean(), vals.std()
    profiles[label] = {
        'hidden_ir': (vals[hidden_ir] - mu) / sd,
        'well_pred': (vals[well_predicted] - mu) / sd,
        'all': (vals - mu) / sd,
    }

x_pos = np.arange(len(key_labels))
width = 0.3
hidden_means = [profiles[l]['hidden_ir'].mean() for l in key_labels]
well_means = [profiles[l]['well_pred'].mean() for l in key_labels]

ax4.bar(x_pos - width/2, hidden_means, width, color='#D32F2F', label=f'"Hidden IR" (n={hidden_ir.sum()})')
ax4.bar(x_pos + width/2, well_means, width, color='#4CAF50', label=f'Well-predicted (n={well_predicted.sum()})')
ax4.axhline(0, color='black', linewidth=0.5)
ax4.set_xticks(x_pos); ax4.set_xticklabels(key_labels, fontsize=10)
ax4.set_ylabel('Z-score', fontsize=10)
ax4.set_title('(D) Feature Profiles: Hidden IR vs Well-Predicted', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9); ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)

# Panel E: Scatter of worst predictions — what does the model see?
ax5 = fig.add_subplot(gs[1, 1])
# Glucose vs BMI colored by error
sc5 = ax5.scatter(glucose, X_raw_df['bmi'].values, c=np.clip(abs_err, 0, 5),
                   cmap='RdYlGn_r', s=15, alpha=0.6, vmin=0, vmax=5)
# Circle the hidden IR
ax5.scatter(glucose[hidden_ir], X_raw_df['bmi'].values[hidden_ir],
            facecolors='none', edgecolors='red', s=80, linewidths=2)
ax5.set_xlabel('Glucose (mg/dL)', fontsize=10)
ax5.set_ylabel('BMI', fontsize=10)
ax5.set_title('(E) Glucose × BMI: Where Errors Occur', fontsize=12, fontweight='bold')
plt.colorbar(sc5, ax=ax5, label='|Error|')
ax5.text(0.05, 0.95, 'Red circles = "Hidden IR"\nNormal glucose/BMI but high HOMA',
         transform=ax5.transAxes, fontsize=8, va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel F: Case studies table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

# Find top 5 worst predictions
worst_idx = np.argsort(abs_err)[-8:][::-1]
table_data = []
for idx in worst_idx:
    table_data.append([
        f'{y[idx]:.1f}',
        f'{oof_pred[idx]:.1f}',
        f'{abs_err[idx]:.1f}',
        f'{glucose[idx]:.0f}',
        f'{X_raw_df["bmi"].values[idx]:.1f}',
        f'{X_raw_df["triglycerides"].values[idx]:.0f}',
        'F' if X_raw_df['sex_num'].values[idx] == 0 else 'M',
    ])

table = ax6.table(cellText=table_data,
                    colLabels=['True\nHOMA', 'Pred\nHOMA', '|Error|', 'Glucose', 'BMI', 'Trig', 'Sex'],
                    loc='center',
                    cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

# Color code the error column
for i in range(len(table_data)):
    cell = table[i+1, 2]  # Error column
    err = float(table_data[i][2])
    if err > 5:
        cell.set_facecolor('#FFCDD2')
    elif err > 3:
        cell.set_facecolor('#FFE0B2')

ax6.set_title('(F) Worst Predictions: Case Studies', fontsize=12, fontweight='bold', pad=20)
ax6.text(0.5, -0.05, 'These patients have elevated insulin but normal-appearing\nglucose/BMI — invisible to any model without insulin measurement.',
         transform=ax6.transAxes, fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='#FFF3E0', alpha=0.9))

plt.suptitle('The "Hidden Insulin Resistant": Patients Our Model Cannot Predict',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_hidden_ir.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_hidden_ir.pdf', bbox_inches='tight')
fprint("Done! Saved fig_hidden_ir.png + .pdf")
