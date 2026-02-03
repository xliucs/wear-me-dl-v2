"""
Figure: The Hill-Climbing Journey — R² trajectory across 28 versions.
Shows how different approaches converge to the same ceiling.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Version data: (version, best_single, best_blend, category, label)
versions = [
    (1,  0.5110, None,    'baseline',    'Baseline (16 models)'),
    (2,  0.5110, 0.5276,  'ensemble',    'Dirichlet blend'),
    (3,  None,   0.5025,  'ensemble',    'Nested stacking (honest)'),
    (4,  0.5146, None,    'features',    '101 features'),
    (5,  0.4531, 0.5104,  'decomp',      'Insulin decomp'),
    (6,  0.5111, None,    'tuning',      'Optuna tuning'),
    (7,  0.5271, 0.5368,  'features',    'Log target + eng features'),
    (8,  0.5287, 0.5350,  'ensemble',    'Multi-seed'),
    (9,  0.5287, 0.5361,  'ensemble',    'Family mega-blend'),
    (10, 0.5287, 0.5365,  'analysis',    'Residual analysis'),
    (11, 0.5367, 0.5414,  'weighting',   'sqrt(y) weighting'),
    (12, 0.5367, 0.5414,  'weighting',   'Weight exponent search'),
    (13, 0.5406, 0.5452,  'tuning',      'Optuna weighted XGB'),
    (14, 0.5398, 0.5465,  'tuning',      'Optuna LGB + GBR'),
    (15, 0.5398, 0.5462,  'tuning',      'Optuna HGBR'),
    (16, 0.5398, 0.5463,  'features',    'Target encoding'),
    (17, 0.5398, 0.5466,  'transform',   'Power transforms'),
    (18, 0.5422, 0.546,   'ensemble',    'Nested stacking v2'),
    (19, 0.5398, 0.5465,  'models',      'KNN + Kernel Ridge'),
    (20, 0.5408, 0.5467,  'transform',   'QuantileTransformer'),
    (21, 0.5388, 0.5465,  'models',      'CatBoost + MAE'),
    (22, 0.5388, 0.5461,  'features',    'Pseudo-insulin + interactions'),
    (23, 0.5371, None,    'tuning',      'Optuna MAE'),
    (24, 0.5453, 0.5445,  'decomp',      'Target decomposition v2'),
    (25, 0.5365, None,    'analysis',    'Deep error analysis'),
    (26, 0.5392, None,    'calibration', 'Calibration + stretch'),
    (27, 0.5384, 0.5412,  'stratify',    'Sex/BMI/Glucose stratification'),
    (28, 0.5384, 0.5400,  'ensemble',    'Max diversity blend'),
]

# Category colors
cat_colors = {
    'baseline':    '#666666',
    'ensemble':    '#2196F3',
    'features':    '#4CAF50',
    'decomp':      '#F44336',
    'tuning':      '#FF9800',
    'weighting':   '#9C27B0',
    'analysis':    '#795548',
    'transform':   '#00BCD4',
    'models':      '#E91E63',
    'calibration': '#607D8B',
    'stratify':    '#CDDC39',
}

cat_labels = {
    'baseline':    'Baseline',
    'ensemble':    'Ensemble/Blending',
    'features':    'Feature Engineering',
    'decomp':      'Target Decomposition',
    'tuning':      'Hyperparameter Tuning',
    'weighting':   'Sample Weighting',
    'analysis':    'Error Analysis',
    'transform':   'Input/Target Transform',
    'models':      'Alternative Models',
    'calibration': 'Calibration',
    'stratify':    'Stratification',
}

fig, ax = plt.subplots(figsize=(14, 7))

# Plot ceiling
ax.axhline(y=0.614, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.7, zorder=1)
ax.text(28.5, 0.616, 'Theoretical Ceiling (R²=0.614)', fontsize=10, color='#D32F2F',
        ha='right', va='bottom', fontstyle='italic')

# Plot original paper baseline
ax.axhline(y=0.50, color='#1565C0', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
ax.text(28.5, 0.502, 'Prior Work (R²=0.50)', fontsize=9, color='#1565C0',
        ha='right', va='bottom', fontstyle='italic')

# Track running best for envelope
best_single_running = []
best_blend_running = []
curr_best_s = 0
curr_best_b = 0

for v, s, b, cat, lab in versions:
    if s is not None and s > curr_best_s:
        curr_best_s = s
    if b is not None and b > curr_best_b:
        curr_best_b = b
    best_single_running.append((v, curr_best_s))
    best_blend_running.append((v, curr_best_b))

# Running best envelope (blend)
bv = [x[0] for x in best_blend_running if x[1] > 0]
br = [x[1] for x in best_blend_running if x[1] > 0]
ax.plot(bv, br, color='#333333', linewidth=1.5, alpha=0.3, zorder=2)
ax.fill_between(bv, 0.48, br, alpha=0.05, color='#333333')

# Plot individual points
plotted_cats = set()
for v, s, b, cat, lab in versions:
    c = cat_colors[cat]
    if s is not None:
        ax.scatter(v, s, color=c, s=60, zorder=5, edgecolors='white', linewidth=0.5,
                   marker='o', label=cat_labels[cat] if cat not in plotted_cats else '')
    if b is not None:
        ax.scatter(v, b, color=c, s=90, zorder=5, edgecolors='black', linewidth=1,
                   marker='D', label=None)
    plotted_cats.add(cat)

# Annotate key milestones
annotations = [
    (7,  0.5368, 'V7: Log target\n+feature eng.', (-50, 25)),
    (11, 0.5414, 'V11: sqrt(y)\nweighting', (-60, 20)),
    (14, 0.5465, 'V14: Optuna\nLGB+GBR', (-55, 20)),
    (20, 0.5467, 'V20: Best\n(R²=0.547)', (15, 25)),
    (5,  0.4531, 'V5: Decomp\nfailure', (10, -30)),
    (25, 0.5365, 'V25: Ceiling\nanalysis', (10, -30)),
]
for v, r2, txt, offset in annotations:
    ax.annotate(txt, (v, r2), textcoords='offset points', xytext=offset,
                fontsize=8, ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#666', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#ccc', alpha=0.9))

# Legend
handles = [mpatches.Patch(facecolor=cat_colors[k], label=v) for k, v in cat_labels.items()]
# Add marker legend
handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Single Model'))
handles.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=8, label='Blend'))
ax.legend(handles=handles, loc='lower right', fontsize=8, ncol=2, framealpha=0.9)

ax.set_xlabel('Version', fontsize=12)
ax.set_ylabel('R² (25-fold CV)', fontsize=12)
ax.set_title('The Hill-Climbing Journey: 28 Approaches to HOMA-IR Prediction', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 29)
ax.set_ylim(0.44, 0.64)
ax.set_xticks(range(1, 29))
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_journey.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_journey.pdf', bbox_inches='tight')
print("Saved fig_journey.png + .pdf")
