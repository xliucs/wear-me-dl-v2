"""
Figure 4: Hidden Insulin Resistant — t-SNE + case studies, blue palette.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2/paper')
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2')
from style import *
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from eval_framework import load_data, get_feature_sets, engineer_all_features
import xgboost as xgb

setup_style()

X_raw, y_raw, feat_names = load_data()
X_all_raw, _, all_cols, _ = get_feature_sets(X_raw)
X_raw_df = pd.DataFrame(X_all_raw, columns=all_cols)
X_a = engineer_all_features(X_raw_df, all_cols)
y = y_raw; log_y = np.log1p(y); w = np.sqrt(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_binned = kbd.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

oof = np.zeros(len(y))
xgb_params = dict(n_estimators=612, max_depth=4, learning_rate=0.017,
    subsample=0.52, colsample_bytree=0.78, min_child_weight=29,
    reg_alpha=2.8, reg_lambda=0.045, random_state=42, n_jobs=-1)

fprint("Running 5-fold CV...")
for i, (tr, te) in enumerate(cv.split(X_a, y_binned)):
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(X_a.iloc[tr], log_y[tr], sample_weight=w[tr])
    oof[te] = np.expm1(m.predict(X_a.iloc[te]))
    fprint(f"  Fold {i+1}/5")

abs_err = np.abs(y - oof)
hidden_ir = (y > 5) & (abs_err > 2)
well_pred = abs_err < np.percentile(abs_err, 25)

fprint("Computing t-SNE...")
scaler = StandardScaler()
X_tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000).fit_transform(
    scaler.fit_transform(X_a))
fprint("t-SNE done")

glucose = X_raw_df['glucose'].values

# ============================================================
fig = plt.figure(figsize=(12, 11))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

# Use blue sequential colormap
from matplotlib.colors import LinearSegmentedColormap
blue_seq = LinearSegmentedColormap.from_list('bseq',
    ['#E3F2FD', '#64B5F6', '#1E88E5', '#0D47A1', '#1A237E'], N=256)
red_seq = LinearSegmentedColormap.from_list('rseq',
    ['#E3F2FD', '#90CAF9', '#FFAB91', '#E53935', '#B71C1C'], N=256)

# Panel A: t-SNE colored by HOMA-IR
ax1 = fig.add_subplot(gs[0, 0])
sc = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(y, 0, 10), cmap=blue_seq,
                  s=8, alpha=0.7, vmin=0, vmax=10, edgecolors='none')
ax1.set_title('(a) t-SNE: True HOMA-IR', fontsize=11)
ax1.set_xlabel('t-SNE 1'); ax1.set_ylabel('t-SNE 2')
cb = plt.colorbar(sc, ax=ax1, shrink=0.8, pad=0.02)
cb.ax.tick_params(labelsize=7)
ax1.set_aspect('equal', adjustable='datalim')

# Panel B: t-SNE colored by error
ax2 = fig.add_subplot(gs[0, 1])
sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.clip(abs_err, 0, 5), cmap=red_seq,
                   s=8, alpha=0.7, vmin=0, vmax=5, edgecolors='none')
ax2.scatter(X_tsne[hidden_ir, 0], X_tsne[hidden_ir, 1], facecolors='none',
            edgecolors=ACCENT_RED, s=50, linewidths=1.5,
            label=f'"Hidden IR" (n={hidden_ir.sum()})')
ax2.set_title('(b) t-SNE: Prediction Error', fontsize=11)
ax2.set_xlabel('t-SNE 1'); ax2.set_ylabel('t-SNE 2')
cb2 = plt.colorbar(sc2, ax=ax2, shrink=0.8, pad=0.02)
cb2.ax.tick_params(labelsize=7)
ax2.legend(fontsize=8, loc='lower left', framealpha=0.9, edgecolor=ACCENT_LIGHT_GRAY)
ax2.set_aspect('equal', adjustable='datalim')

# Panel C: Glucose × BMI with error
ax3 = fig.add_subplot(gs[0, 2])
bmi = X_raw_df['bmi'].values
sc3 = ax3.scatter(glucose, bmi, c=np.clip(abs_err, 0, 5), cmap=red_seq,
                   s=10, alpha=0.5, vmin=0, vmax=5, edgecolors='none')
ax3.scatter(glucose[hidden_ir], bmi[hidden_ir], facecolors='none',
            edgecolors=ACCENT_RED, s=50, linewidths=1.5)
ax3.set_xlabel('Glucose (mg/dL)'); ax3.set_ylabel('BMI')
ax3.set_title('(c) Glucose × BMI: Error Map', fontsize=11)
cb3 = plt.colorbar(sc3, ax=ax3, shrink=0.8, pad=0.02)
cb3.ax.tick_params(labelsize=7)
cb3.set_label('|Error|', fontsize=8)

# Panel D: Feature profiles
ax4 = fig.add_subplot(gs[1, 0])
key_feats = ['glucose', 'bmi', 'triglycerides', 'hdl']
key_labels = ['Glucose', 'BMI', 'Trig', 'HDL']
x_pos = np.arange(len(key_labels))
hidden_z, well_z = [], []
for feat in key_feats:
    vals = X_raw_df[feat].values
    mu, sd = vals.mean(), vals.std()
    hidden_z.append((vals[hidden_ir] - mu) / sd)
    well_z.append((vals[well_pred] - mu) / sd)

ax4.bar(x_pos - 0.18, [h.mean() for h in hidden_z], 0.35, color=ACCENT_RED, alpha=0.8,
        label=f'"Hidden IR" (n={hidden_ir.sum()})')
ax4.bar(x_pos + 0.18, [w_.mean() for w_ in well_z], 0.35, color=BLUE_MED, alpha=0.8,
        label=f'Well-predicted (n={well_pred.sum()})')
ax4.axhline(0, color='black', linewidth=0.3)
ax4.set_xticks(x_pos); ax4.set_xticklabels(key_labels)
ax4.set_ylabel('Z-score'); ax4.set_title('(d) Feature Profiles', fontsize=11)
ax4.legend(fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor=ACCENT_LIGHT_GRAY)

# Panel E: Predicted vs True for hidden IR only
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y[~hidden_ir], oof[~hidden_ir], c=BLUE_PALE, s=8, alpha=0.2, edgecolors='none', label='Other')
ax5.scatter(y[hidden_ir], oof[hidden_ir], c=ACCENT_RED, s=25, alpha=0.8, edgecolors='white',
            linewidth=0.3, label=f'"Hidden IR" (n={hidden_ir.sum()})')
ax5.plot([0, 15], [0, 15], '--', color=ACCENT_GRAY, linewidth=1)
ax5.set_xlabel('True HOMA-IR'); ax5.set_ylabel('Predicted HOMA-IR')
ax5.set_title('(e) Hidden IR: Pred vs True', fontsize=11)
ax5.set_xlim(0, 15); ax5.set_ylim(0, 12)
ax5.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor=ACCENT_LIGHT_GRAY)

# Panel F: Case studies table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

worst_idx = np.argsort(abs_err)[-6:][::-1]
table_data = []
for idx in worst_idx:
    table_data.append([
        f'{y[idx]:.1f}', f'{oof[idx]:.1f}', f'{abs_err[idx]:.1f}',
        f'{glucose[idx]:.0f}', f'{bmi[idx]:.1f}',
        'F' if X_raw_df['sex_num'].values[idx] == 0 else 'M',
    ])

table = ax6.table(cellText=table_data,
    colLabels=['True', 'Pred', '|Err|', 'Gluc', 'BMI', 'Sex'],
    loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.0)

# Style header
for j in range(6):
    table[0, j].set_facecolor(BLUE_DARK)
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(len(table_data)):
    for j in range(6):
        table[i+1, j].set_edgecolor(ACCENT_LIGHT_GRAY)
    err = float(table_data[i][2])
    if err > 8:
        table[i+1, 2].set_facecolor('#FFCDD2')
    elif err > 5:
        table[i+1, 2].set_facecolor('#FFE0B2')

ax6.set_title('(f) Worst Predictions', fontsize=11, pad=15)
ax6.text(0.5, 0.02, 'Normal glucose/BMI but\nelevated insulin (invisible)',
         transform=ax6.transAxes, fontsize=8, ha='center', color=ACCENT_GRAY, fontstyle='italic')

plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_hidden_ir.png', dpi=300)
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_hidden_ir.pdf')
fprint("Saved fig_hidden_ir")
