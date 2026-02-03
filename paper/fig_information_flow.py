"""
Figure: Information Flow Diagram — Where does HOMA-IR information come from?
Quantifies the contribution of each feature group.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as FancyBboxPatch
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 11))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.3)

# ============================================================
# Panel A: Feature Group Importance (Drop-one analysis)
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])

groups = ['Glucose', 'BMI', 'Resting HR', 'HRV Features', 'Steps', 'Triglycerides',
          'HDL', 'Sleep', 'AZM', 'Age', 'Sex', 'LDL', 'Total Chol']
importance = [-0.096, -0.011, -0.004, -0.004, -0.002, -0.001,
              -0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.002]

colors = []
for g in groups:
    if g == 'Glucose':
        colors.append('#D32F2F')
    elif g in ['BMI', 'Age', 'Sex']:
        colors.append('#1565C0')
    elif g in ['Resting HR', 'HRV Features', 'Steps', 'Sleep', 'AZM']:
        colors.append('#2E7D32')
    else:
        colors.append('#F57F17')

bars = ax1.barh(range(len(groups)), importance, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(groups)))
ax1.set_yticklabels(groups, fontsize=9)
ax1.set_xlabel('ΔR² when feature group removed', fontsize=10)
ax1.set_title('(A) Feature Group Importance', fontsize=12, fontweight='bold')
ax1.axvline(0, color='black', linewidth=0.5)
ax1.invert_yaxis()

# Add value labels
for i, (v, g) in enumerate(zip(importance, groups)):
    if v < -0.002:
        ax1.text(v - 0.002, i, f'{v:.3f}', va='center', ha='right', fontsize=8, fontweight='bold')
    else:
        ax1.text(max(v, 0) + 0.001, i, f'{v:+.3f}', va='center', ha='left', fontsize=8, color='#666')

# Color legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#D32F2F', label='Blood: Glucose'),
    Patch(facecolor='#F57F17', label='Blood: Lipids'),
    Patch(facecolor='#1565C0', label='Demographics'),
    Patch(facecolor='#2E7D32', label='Wearable'),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ============================================================
# Panel B: Information Ceiling Analysis
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])

# Stacked bar showing variance decomposition
categories = ['Total\nVariance', 'Model\nCaptures', 'Achievable\nGap', 'Irreducible\nNoise']
values = [1.0, 0.547, 0.067, 0.386]
colors_bar = ['#E0E0E0', '#2196F3', '#FF9800', '#F44336']
bottoms = [0, 0, 0.547, 0.614]

# Single stacked bar
bar_width = 0.5
for i, (cat, val, col, bot) in enumerate(zip(categories, values, colors_bar, bottoms)):
    if cat == 'Total\nVariance':
        continue
    ax2.bar(0, val, bar_width, bottom=bot, color=col, edgecolor='white', linewidth=1, label=cat)

ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(0, 1.05)
ax2.set_xticks([])
ax2.set_ylabel('Proportion of Variance', fontsize=10)
ax2.set_title('(B) Variance Decomposition', fontsize=12, fontweight='bold')

# Annotations
ax2.annotate(f'Model captures\nR²=0.547 (55%)', xy=(0.25, 0.27), fontsize=10,
             fontweight='bold', color='white', ha='center')
ax2.annotate(f'Gap\n0.067', xy=(0.25, 0.58), fontsize=9,
             color='#333', ha='center')
ax2.annotate(f'Irreducible noise\n38.6%\n(missing insulin)', xy=(0.25, 0.82), fontsize=10,
             fontweight='bold', color='white', ha='center')

# Ceiling line
ax2.axhline(y=0.614, color='#D32F2F', linestyle='--', linewidth=2, xmin=0.1, xmax=0.9)
ax2.text(0.35, 0.625, 'Ceiling = 0.614', fontsize=9, color='#D32F2F', fontweight='bold')

ax2.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

# ============================================================
# Panel C: Error Correlation Matrix
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])

model_names = ['XGB\n(d=3)', 'XGB\n(Optuna)', 'LGB', 'LGB\n(QT)', 'GBR', 'ElasticNet']
error_corr = np.array([
    [1.000, 0.999, 0.991, 0.990, 0.995, 0.878],
    [0.999, 1.000, 0.989, 0.989, 0.994, 0.881],
    [0.991, 0.989, 1.000, 0.999, 0.993, 0.886],
    [0.990, 0.989, 0.999, 1.000, 0.992, 0.887],
    [0.995, 0.994, 0.993, 0.992, 1.000, 0.878],
    [0.878, 0.881, 0.886, 0.887, 0.878, 1.000],
])

im = ax3.imshow(error_corr, cmap='RdYlBu_r', vmin=0.85, vmax=1.0, aspect='auto')
ax3.set_xticks(range(6))
ax3.set_yticks(range(6))
ax3.set_xticklabels(model_names, fontsize=8)
ax3.set_yticklabels(model_names, fontsize=8)
ax3.set_title('(C) Error Correlation Between Models', fontsize=12, fontweight='bold')

# Add text annotations
for i in range(6):
    for j in range(6):
        color = 'white' if error_corr[i, j] > 0.97 else 'black'
        ax3.text(j, i, f'{error_corr[i, j]:.3f}', ha='center', va='center',
                fontsize=8, color=color, fontweight='bold' if i == 5 or j == 5 else 'normal')

plt.colorbar(im, ax=ax3, shrink=0.8, label='Pearson Correlation of Errors')

# Box around ElasticNet row/col
rect = plt.Rectangle((4.5, -0.5), 1.1, 6.1, linewidth=2, edgecolor='#D32F2F', facecolor='none', linestyle='--')
ax3.add_patch(rect)
rect2 = plt.Rectangle((-0.5, 4.5), 6.1, 1.1, linewidth=2, edgecolor='#D32F2F', facecolor='none', linestyle='--')
ax3.add_patch(rect2)
ax3.text(6.1, 2.5, 'Only source\nof diversity', fontsize=8, color='#D32F2F', fontweight='bold',
         ha='left')

# ============================================================
# Panel D: The Insulin Bottleneck
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])

# HOMA = glucose × insulin / 405
# Show correlation waterfall
factors = ['Insulin\n(r=0.97)', 'Glucose\n(r=0.57)', 'BMI\n(r=0.43)', 'Trig\n(r=0.41)',
           'HDL\n(r=-0.30)', 'RHR\n(r=0.19)', 'Steps\n(r=-0.12)', 'HRV\n(r=-0.10)']
corrs = [0.969, 0.574, 0.43, 0.41, -0.30, 0.19, -0.12, -0.10]
abs_corrs = [abs(c) for c in corrs]

bar_colors = ['#D32F2F' if f.startswith('Insulin') else
              '#F57F17' if any(f.startswith(x) for x in ['Glucose', 'Trig', 'HDL']) else
              '#1565C0' if f.startswith('BMI') else
              '#2E7D32' for f in factors]

bars = ax4.bar(range(len(factors)), abs_corrs, color=bar_colors, edgecolor='white', linewidth=0.5)
ax4.set_xticks(range(len(factors)))
ax4.set_xticklabels(factors, fontsize=8)
ax4.set_ylabel('|Correlation| with HOMA-IR', fontsize=10)
ax4.set_title('(D) The Insulin Bottleneck', fontsize=12, fontweight='bold')

# Highlight insulin bar
bars[0].set_edgecolor('#D32F2F')
bars[0].set_linewidth(2)
bars[0].set_hatch('///')

# Available vs unavailable
ax4.axvline(x=0.5, color='#D32F2F', linestyle='--', linewidth=2, alpha=0.7)
ax4.text(0, 1.02, 'NOT\nAVAILABLE', ha='center', fontsize=9, color='#D32F2F',
         fontweight='bold', transform=ax4.get_xaxis_transform())
ax4.text(4, 1.02, 'Available features', ha='center', fontsize=9, color='#2E7D32',
         fontweight='bold', transform=ax4.get_xaxis_transform())

ax4.set_ylim(0, 1.1)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.suptitle('Information-Theoretic Analysis of HOMA-IR Prediction', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_information_flow.png', dpi=300, bbox_inches='tight')
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_information_flow.pdf', bbox_inches='tight')
print("Saved fig_information_flow.png + .pdf")
