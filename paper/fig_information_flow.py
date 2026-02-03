"""
Figure 2: Information-Theoretic Analysis — 4 panels, clean blue palette.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2/paper')
from style import *
import numpy as np
import matplotlib.gridspec as gridspec

setup_style()

fig = plt.figure(figsize=(12, 11))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.35)

# ============================================================
# Panel A: Feature Group Importance
# ============================================================
ax1 = fig.add_subplot(gs[0, 0])

groups = ['Glucose', 'BMI', 'Resting HR', 'HRV', 'Steps', 'Triglycerides',
          'HDL', 'Sleep', 'AZM', 'Age', 'Sex', 'LDL', 'Total Chol']
importance = [-0.096, -0.011, -0.004, -0.004, -0.002, -0.001,
              -0.001, 0.000, 0.000, 0.001, 0.001, 0.001, 0.002]

bar_colors = []
for g in groups:
    if g == 'Glucose':
        bar_colors.append(ACCENT_RED)
    elif g in ['BMI', 'Age', 'Sex']:
        bar_colors.append(BLUE_DARK)
    elif g in ['Resting HR', 'HRV', 'Steps', 'Sleep', 'AZM']:
        bar_colors.append(ACCENT_GREEN)
    else:
        bar_colors.append(ACCENT_ORANGE)

ax1.barh(range(len(groups)), importance, color=bar_colors, edgecolor='white', linewidth=0.3, height=0.7)
ax1.set_yticks(range(len(groups)))
ax1.set_yticklabels(groups, fontsize=9)
ax1.set_xlabel('ΔR² when removed', fontsize=9)
ax1.set_title('(a) Feature Group Importance', fontsize=11)
ax1.axvline(0, color='black', linewidth=0.3)
ax1.invert_yaxis()
ax1.set_xlim(-0.11, 0.01)

for i, v in enumerate(importance):
    if v < -0.002:
        ax1.text(v - 0.002, i, f'{v:.3f}', va='center', ha='right', fontsize=7.5, fontweight='bold')

# Compact legend
from matplotlib.patches import Patch
leg = [Patch(facecolor=ACCENT_RED, label='Glucose'),
       Patch(facecolor=ACCENT_ORANGE, label='Lipids'),
       Patch(facecolor=BLUE_DARK, label='Demo'),
       Patch(facecolor=ACCENT_GREEN, label='Wearable')]
ax1.legend(handles=leg, loc='lower right', fontsize=7.5, framealpha=0.9,
           edgecolor=ACCENT_LIGHT_GRAY, handlelength=1, handletextpad=0.4)

# ============================================================
# Panel B: Variance Decomposition (pie chart — cleaner)
# ============================================================
ax2 = fig.add_subplot(gs[0, 1])

sizes = [0.547, 0.067, 0.386]
labels_pie = ['Model captures\n(R²=0.547)', 'Achievable gap\n(ΔR²=0.067)',
              'Irreducible noise\n(missing insulin)']
colors_pie = [BLUE_MED, BLUE_PALE, ACCENT_RED]
explode = (0.02, 0.04, 0.02)

wedges, texts, autotexts = ax2.pie(sizes, labels=None, autopct='%1.0f%%',
    colors=colors_pie, explode=explode, startangle=90,
    textprops={'fontsize': 10, 'fontweight': 'bold'},
    pctdistance=0.65, wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'})

# Color the percentage text
for at, c in zip(autotexts, ['white', BLUE_DARK, 'white']):
    at.set_color(c)

ax2.legend(wedges, labels_pie, loc='lower center', fontsize=8, framealpha=0.9,
           edgecolor=ACCENT_LIGHT_GRAY, bbox_to_anchor=(0.5, -0.12), ncol=1)
ax2.set_title('(b) HOMA-IR Variance Decomposition', fontsize=11, pad=12)

# ============================================================
# Panel C: Error Correlation Matrix
# ============================================================
ax3 = fig.add_subplot(gs[1, 0])

model_names = ['XGB\n(d=3)', 'XGB\n(Opt.)', 'LGB', 'LGB\n(QT)', 'GBR', 'Elastic\nNet']
error_corr = np.array([
    [1.000, 0.999, 0.991, 0.990, 0.995, 0.878],
    [0.999, 1.000, 0.989, 0.989, 0.994, 0.881],
    [0.991, 0.989, 1.000, 0.999, 0.993, 0.886],
    [0.990, 0.989, 0.999, 1.000, 0.992, 0.887],
    [0.995, 0.994, 0.993, 0.992, 1.000, 0.878],
    [0.878, 0.881, 0.886, 0.887, 0.878, 1.000],
])

# Use blue colormap
from matplotlib.colors import LinearSegmentedColormap
blue_cmap = LinearSegmentedColormap.from_list('blue_corr',
    ['#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1'], N=256)

im = ax3.imshow(error_corr, cmap=blue_cmap, vmin=0.85, vmax=1.0, aspect='equal')
ax3.set_xticks(range(6))
ax3.set_yticks(range(6))
ax3.set_xticklabels(model_names, fontsize=8)
ax3.set_yticklabels(model_names, fontsize=8)
ax3.set_title('(c) Error Correlation Between Models', fontsize=11)

for i in range(6):
    for j in range(6):
        color = 'white' if error_corr[i, j] > 0.96 else BLUE_DARK
        weight = 'bold' if (i == 5 or j == 5) else 'normal'
        ax3.text(j, i, f'{error_corr[i, j]:.3f}', ha='center', va='center',
                fontsize=7.5, color=color, fontweight=weight)

cb = plt.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
cb.ax.tick_params(labelsize=8)
cb.set_label('Error Correlation', fontsize=8)

# Highlight ElasticNet
for i in range(6):
    ax3.add_patch(plt.Rectangle((4.5, i-0.5), 1, 1, linewidth=1.5,
                  edgecolor=ACCENT_RED, facecolor='none', linestyle='--'))
    ax3.add_patch(plt.Rectangle((i-0.5, 4.5), 1, 1, linewidth=1.5,
                  edgecolor=ACCENT_RED, facecolor='none', linestyle='--'))

# ============================================================
# Panel D: The Insulin Bottleneck
# ============================================================
ax4 = fig.add_subplot(gs[1, 1])

factors = ['Insulin', 'Glucose', 'BMI', 'Trig', 'HDL', 'RHR', 'Steps', 'HRV']
corrs = [0.969, 0.574, 0.43, 0.41, 0.30, 0.19, 0.12, 0.10]

bar_colors_d = [ACCENT_RED] + [BLUE_DARK]*7
bar_hatches = ['///'] + ['']*7

bars = ax4.bar(range(len(factors)), corrs, color=bar_colors_d, edgecolor='white',
               linewidth=0.5, width=0.7)
bars[0].set_hatch('///')
bars[0].set_edgecolor(ACCENT_RED)

ax4.set_xticks(range(len(factors)))
ax4.set_xticklabels(factors, fontsize=9)
ax4.set_ylabel('|Correlation| with HOMA-IR', fontsize=9)
ax4.set_title('(d) The Insulin Bottleneck', fontsize=11)
ax4.set_ylim(0, 1.15)

# Divider between unavailable and available
ax4.axvline(x=0.5, color=ACCENT_RED, linestyle='--', linewidth=1.5, alpha=0.6)
ax4.text(0, 1.05, 'Unavailable', ha='center', fontsize=8, color=ACCENT_RED, fontweight='bold')
ax4.text(4, 1.05, 'Available features', ha='center', fontsize=8, color=BLUE_DARK, fontweight='bold')

# Value labels on bars
for i, c in enumerate(corrs):
    ax4.text(i, c + 0.02, f'{c:.2f}', ha='center', fontsize=7.5, color=ACCENT_GRAY)

plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_information_flow.png', dpi=300)
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_information_flow.pdf')
fprint("Saved fig_information_flow")
