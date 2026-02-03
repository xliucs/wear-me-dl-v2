"""
Figure 1: The Hill-Climbing Journey — R² trajectory across 28 versions.
Clean, compact, blue-dominant.
"""
import sys
sys.path.insert(0, '/Users/jarvis/clawd/wear-me-dl-v2/paper')
from style import *
import numpy as np

setup_style()

# Version data: (version, best_single, best_blend, category)
versions = [
    (1,  0.5110, None,    'baseline'),
    (2,  0.5110, 0.5276,  'ensemble'),
    (3,  None,   0.5025,  'ensemble'),
    (4,  0.5146, None,    'features'),
    (5,  0.4531, 0.5104,  'decomp'),
    (6,  0.5111, None,    'tuning'),
    (7,  0.5271, 0.5368,  'features'),
    (8,  0.5287, 0.5350,  'ensemble'),
    (9,  0.5287, 0.5361,  'ensemble'),
    (10, 0.5287, 0.5365,  'analysis'),
    (11, 0.5367, 0.5414,  'weighting'),
    (12, 0.5367, 0.5414,  'weighting'),
    (13, 0.5406, 0.5452,  'tuning'),
    (14, 0.5398, 0.5465,  'tuning'),
    (15, 0.5398, 0.5462,  'tuning'),
    (16, 0.5398, 0.5463,  'features'),
    (17, 0.5398, 0.5466,  'transform'),
    (18, 0.5422, 0.546,   'ensemble'),
    (19, 0.5398, 0.5465,  'models'),
    (20, 0.5408, 0.5467,  'transform'),
    (21, 0.5388, 0.5465,  'models'),
    (22, 0.5388, 0.5461,  'features'),
    (23, 0.5371, None,    'tuning'),
    (24, 0.5453, 0.5445,  'decomp'),
    (25, 0.5365, None,    'analysis'),
    (26, 0.5392, None,    'calibration'),
    (27, 0.5384, 0.5412,  'stratify'),
    (28, 0.5384, 0.5400,  'ensemble'),
]

fig, ax = plt.subplots(figsize=(10, 6))

# Ceiling and prior work
ax.axhspan(0.610, 0.618, color=ACCENT_RED, alpha=0.08)
ax.axhline(y=0.614, color=ACCENT_RED, linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
ax.text(28.3, 0.614, 'Ceiling\n(R²=0.614)', fontsize=8, color=ACCENT_RED,
        ha='left', va='center', fontstyle='italic')

ax.axhline(y=0.50, color=ACCENT_GRAY, linestyle=':', linewidth=1, alpha=0.5, zorder=1)
ax.text(28.3, 0.50, 'Prior work\n(R²=0.50)', fontsize=8, color=ACCENT_GRAY,
        ha='left', va='center', fontstyle='italic')

# Running best envelope (blend)
curr_best_b = 0
bv, br = [], []
for v, s, b, cat in versions:
    if b is not None and b > curr_best_b:
        curr_best_b = b
    if curr_best_b > 0:
        bv.append(v); br.append(curr_best_b)
ax.fill_between(bv, 0.49, br, alpha=0.04, color=BLUE_DARK)
ax.plot(bv, br, color=BLUE_DARK, linewidth=1, alpha=0.2, zorder=2)

# Plot points
for v, s, b, cat in versions:
    c = CAT_COLORS.get(cat, ACCENT_GRAY)
    if s is not None:
        ax.scatter(v, s, color=c, s=35, zorder=5, edgecolors='white', linewidth=0.3,
                   marker='o', alpha=0.8)
    if b is not None:
        ax.scatter(v, b, color=c, s=60, zorder=6, edgecolors=c, linewidth=1,
                   marker='D', alpha=0.9, facecolors=c)

# Annotate key milestones only
annotations = [
    (7,  0.5368, 'Log target\n+features', (-40, 22)),
    (11, 0.5414, 'sqrt(y)\nweighting', (-45, 18)),
    (14, 0.5465, 'Optuna\nLGB+GBR', (15, 18)),
    (20, 0.5467, 'Best blend\nR²=0.547', (15, 18)),
]
for v, r2, txt, offset in annotations:
    ax.annotate(txt, (v, r2), textcoords='offset points', xytext=offset,
                fontsize=8, ha='center', color=BLUE_DARK,
                arrowprops=dict(arrowstyle='->', color=BLUE_MED, lw=0.8),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=BLUE_PALE, alpha=0.9, linewidth=0.5))

# Simplified legend — just single vs blend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE_MED,
           markersize=6, label='Single model', linewidth=0),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=BLUE_MED,
           markeredgecolor=BLUE_DARK, markersize=7, label='Blend', linewidth=0),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9,
          edgecolor=ACCENT_LIGHT_GRAY)

ax.set_xlabel('Experiment Version', fontsize=11)
ax.set_ylabel('R² (25-fold CV)', fontsize=11)
ax.set_title('28 Approaches to HOMA-IR Prediction', fontsize=13, fontweight='bold', pad=10)
ax.set_xlim(0, 29.5)
ax.set_ylim(0.44, 0.64)
ax.set_xticks([1, 5, 10, 15, 20, 25, 28])

plt.tight_layout()
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_journey.png', dpi=300)
plt.savefig('/Users/jarvis/clawd/wear-me-dl-v2/paper/fig_journey.pdf')
fprint("Saved fig_journey")
