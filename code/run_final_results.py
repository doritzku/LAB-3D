"""
run_final_results.py
====================
Single script — runs ALL methods on LAB-3D dataset,
generates per-sample errors, metrics, and all figures.

Uses original method implementations from:
  - comparison_methods.py  (IDW, RBF-Direct, APLM, M-PLM, RBF-Network, Kriging, PL-RBF)
  - ablation_study.py      (AblationPLRBF: C0 → C4)
  - plrbf_enhanced.py      (PathLossRBF_Enhanced: C4 full method)

Outputs (./results_final/):
  per_sample.csv            — per-position H/V/3D errors, all methods
  per_sample_ablation.csv   — per-position errors, ablation C0-C4
  metrics.csv               — summary metrics (mean, P90, RMSE)
  plot_cdf.png              — empirical CDF, all methods
  plot_boxplot.png          — box plot, all methods
  plot_bars.png             — bar chart, all methods
  plot_ablation.png         — ablation progression bars

Requirements:
  pip install numpy pandas scipy scikit-learn matplotlib seaborn
  Scripts needed in same folder:
    comparison_methods.py, ablation_study.py, plrbf_enhanced.py
  Data files:
    Lab_3D_Train_combined_clean.csv
    Lab_3D_Test_combined_clean.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths — edit these if needed ──────────────────────────────
TRAIN_PATH = 'Lab_3D_Train_combined_clean.csv'
TEST_PATH  = 'Lab_3D_Test_combined_clean.csv'
OUT_DIR    = './results_final'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────
RSSI_COLS = [f'RSSI-{i}' for i in range(1, 7)]
FILL_VAL  = -100.0

# ── Colors ────────────────────────────────────────────────────
C_E     = '#C0392B'   # PL-RBF Enhanced (red)
C_B     = '#2980B9'   # PL-RBF Baseline (blue)
C_K     = '#7F8C8D'   # Wt. k-NN (grey)
C_PUB   = ['#607D8B', '#FF9800', '#2196F3',
           '#4CAF50', '#9C27B0', '#795548']

METHOD_ORDER = ['IDW', 'RBF-Direct', 'APLM', 'M-PLM',
                'RBF-Network', 'Kriging',
                'PL-RBF Baseline', 'Wt. k-NN (C0)', 'PL-RBF Enhanced']
SHORT_LABELS = ['IDW', 'RBF-D', 'APLM', 'M-PLM', 'RBF-N', 'Krig.',
                'PL-RBF\nBase', 'Wt.\nkNN', 'PL-RBF\nEnhc.']
COLORS       = C_PUB + [C_B, C_K, C_E]

ABL_ORDER  = ['C0: Wt. k-NN', 'C1: +Path-Loss', 'C2: +Gauss 3\u03c3',
              'C3: +Smooth 0.3', 'C4: +W-3NN']
ABL_LABELS = ['C0\nWt. k-NN', 'C1\n+PL', 'C2\n+Gauss',
              'C3\n+Smooth', 'C4\n+W-3NN']

# ══════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════
print('Loading data ...')
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
for df in [train, test]:
    for c in RSSI_COLS:
        df[c] = df[c].replace(FILL_VAL, np.nan)

train_med = train.groupby(['X', 'Y', 'Z'])[RSSI_COLS].median().reset_index()
test_med  = test.groupby(['X', 'Y', 'Z'])[RSSI_COLS].median().reset_index()
train_med[RSSI_COLS] = train_med[RSSI_COLS].fillna(FILL_VAL)
test_med[RSSI_COLS]  = test_med[RSSI_COLS].fillna(FILL_VAL)
y_te = test_med[['X', 'Y', 'Z']].values

print(f'  Train RPs: {len(train_med)} | Test positions: {len(test_med)}')

# ══════════════════════════════════════════════════════════════
# 2.  IMPORT ORIGINAL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comparison_methods import (
    IDWInterpolator,
    RBFDirectInterpolator,
    APLMInterpolator,
    MPLMInterpolator,
    RBFNetworkInterpolator,
    KrigingInterpolator,
    PathLossRBF,
)
from ablation_study import AblationPLRBF
from plrbf_enhanced import PathLossRBF_Enhanced

# ══════════════════════════════════════════════════════════════
# 3.  ERROR HELPER
# ══════════════════════════════════════════════════════════════
def errs(pred, true):
    eh  = np.sqrt((pred[:, 0] - true[:, 0])**2 +
                  (pred[:, 1] - true[:, 1])**2)
    ev  = np.abs(pred[:, 2] - true[:, 2])
    e3d = np.sqrt(eh**2 + ev**2)
    return eh, ev, e3d

def met(eh, ev, e3d, name):
    d = {'Method': name}
    for nm, x in [('H', eh), ('V', ev), ('3D', e3d)]:
        d.update({
            f'{nm}_Mean':   round(float(x.mean()),   4),
            f'{nm}_Median': round(float(np.median(x)), 4),
            f'{nm}_P75':    round(float(np.percentile(x, 75)), 4),
            f'{nm}_P90':    round(float(np.percentile(x, 90)), 4),
            f'{nm}_RMSE':   round(float(np.sqrt((x**2).mean())), 4),
        })
    return d

# ══════════════════════════════════════════════════════════════
# 4.  RUN ALL COMPARISON METHODS
# ══════════════════════════════════════════════════════════════
print('\n=== Comparison methods ===')
results    = []
per_sample = []

def run(name, pred):
    eh, ev, e3 = errs(pred, y_te)
    results.append(met(eh, ev, e3, name))
    for i in range(len(eh)):
        per_sample.append({'Method': name,
                           'eH': eh[i], 'eV': ev[i], 'e3D': e3[i]})
    print(f'  {name:28s}  H={eh.mean():.4f}  V={ev.mean():.4f}')

# Published baselines — use exactly the original fit() calls
m = IDWInterpolator();           m.fit(train_med)
run('IDW',         m.predict_position(test_med))

m = RBFDirectInterpolator();     m.fit(raw_train=train, train_med=train_med)
run('RBF-Direct',  m.predict_position(test_med))

m = APLMInterpolator();          m.fit(train_med)
run('APLM',        m.predict_position(test_med))

m = MPLMInterpolator();          m.fit(train_med)
run('M-PLM',       m.predict_position(test_med))

m = RBFNetworkInterpolator();    m.fit(train_med)
run('RBF-Network', m.predict_position(test_med))

m = KrigingInterpolator();       m.fit(train_med)
run('Kriging',     m.predict_position(test_med))

m = PathLossRBF();               m.fit(train_med)
run('PL-RBF Baseline', m.predict_position(test_med))

# Proposed
m = PathLossRBF_Enhanced(sigma_thresh=3.0, smoothing=0.3, k=3)
m.fit(train)
run('PL-RBF Enhanced', m.predict_position(test_med))

# Weighted k-NN baseline (C0)
m = AblationPLRBF(use_rbf=False, use_pl=False, use_gauss=False,
                  smoothing=1.0, k=5, use_wknn=True)
m.fit(train)
run('Wt. k-NN (C0)', m.predict_position(test_med))

# ══════════════════════════════════════════════════════════════
# 5.  RUN ABLATION  C0 → C4
# ══════════════════════════════════════════════════════════════
print('\n=== Ablation C0 → C4 ===')
abl_configs = [
    ('C0: Wt. k-NN',
     dict(use_rbf=False, use_pl=False, use_gauss=False,
          smoothing=1.0,  k=5, use_wknn=True)),
    ('C1: +Path-Loss',
     dict(use_rbf=True,  use_pl=True,  use_gauss=False,
          smoothing=1.0,  k=1, use_wknn=False)),
    ('C2: +Gauss 3\u03c3',
     dict(use_rbf=True,  use_pl=True,  use_gauss=True,
          smoothing=1.0,  k=1, use_wknn=False)),
    ('C3: +Smooth 0.3',
     dict(use_rbf=True,  use_pl=True,  use_gauss=True,
          smoothing=0.3,  k=1, use_wknn=False)),
    ('C4: +W-3NN',
     dict(use_rbf=True,  use_pl=True,  use_gauss=True,
          smoothing=0.3,  k=3, use_wknn=True)),
]

abl_per = []
for nm, cfg in abl_configs:
    m = AblationPLRBF(**cfg)
    m.fit(train)
    pred = m.predict_position(test_med)
    eh, ev, e3 = errs(pred, y_te)
    print(f'  {nm:28s}  H={eh.mean():.4f}  V={ev.mean():.4f}')
    for i in range(len(eh)):
        abl_per.append({'Config': nm,
                        'eH': eh[i], 'eV': ev[i], 'e3D': e3[i]})

# ══════════════════════════════════════════════════════════════
# 6.  SAVE CSVs
# ══════════════════════════════════════════════════════════════
df_per     = pd.DataFrame(per_sample)
df_abl     = pd.DataFrame(abl_per)
df_metrics = pd.DataFrame(results)

df_per.to_csv(    f'{OUT_DIR}/per_sample.csv',           index=False)
df_abl.to_csv(    f'{OUT_DIR}/per_sample_ablation.csv',  index=False)
df_metrics.to_csv(f'{OUT_DIR}/metrics.csv',              index=False)
print(f'\nCSVs saved to {OUT_DIR}/')

# ══════════════════════════════════════════════════════════════
# 7.  FIGURES
# ══════════════════════════════════════════════════════════════

# ── Fig 1: CDF ────────────────────────────────────────────────
print('Plotting CDF ...')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(wspace=0.38)
lss = ['--', '-.', '--', '-.', '--', '-.', '--', '--', '-']

for ax, metric, title in [
    (axes[0], 'eH', 'Horizontal Error $\\varepsilon_H$'),
    (axes[1], 'eV', 'Vertical Error $\\varepsilon_V$'),
]:
    for i, m in enumerate(METHOD_ORDER):
        v = np.sort(df_per[df_per['Method'] == m][metric].values)
        if len(v) == 0:
            continue
        lw = 2.5 if m == 'PL-RBF Enhanced' else 1.2
        ls = '-'  if m == 'PL-RBF Enhanced' else lss[i % len(lss)]
        ax.plot(v, np.linspace(0, 1, len(v)),
                color=COLORS[i], lw=lw, ls=ls, label=SHORT_LABELS[i],
                zorder=5 if m == 'PL-RBF Enhanced' else 2)
    ax.axhline(0.9, color='k', lw=0.7, ls=':', alpha=0.4)
    ax.set_xlabel(f'{title} (m)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, ls=':')
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)

axes[0].legend(fontsize=7.5, loc='lower right', framealpha=0.9, ncol=2)
fig.suptitle('Empirical CDF — D2 Test-Combined (199 positions)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/plot_cdf.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 2: Box Plot ───────────────────────────────────────────
print('Plotting box plot ...')
fig, axes = plt.subplots(2, 1, figsize=(12, 9))
fig.subplots_adjust(hspace=0.45)

for ax, metric, ylabel, title in [
    (axes[0], 'eH', '$\\varepsilon_H$ (m)', 'Horizontal Positioning Error'),
    (axes[1], 'eV', '$\\varepsilon_V$ (m)', 'Vertical Positioning Error'),
]:
    data = [df_per[df_per['Method'] == m][metric].values
            for m in METHOD_ORDER]
    x = np.arange(1, 10)
    bp = ax.boxplot(
        data, positions=x, widths=0.55, patch_artist=True,
        showfliers=True,
        flierprops=dict(marker='o', markersize=2.5, alpha=0.3,
                        linestyle='none'),
        medianprops=dict(color='white', linewidth=2.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.5),
        boxprops=dict(linewidth=1.2),
    )
    for patch, col in zip(bp['boxes'], COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for flier, col in zip(bp['fliers'], COLORS):
        flier.set_markerfacecolor(col)
        flier.set_markeredgecolor(col)
    for i, d in enumerate(data):
        if len(d) == 0:
            continue
        med = np.median(d)
        ax.text(x[i], med + 0.02, f'{med:.2f}',
                ha='center', va='bottom', fontsize=7,
                color='white', fontweight='bold', zorder=6)

    ax.axvspan(7.5, 9.5, alpha=0.06, color=C_E, zorder=0)
    ax.axvline(7.5, color='gray', lw=0.8, ls='--', alpha=0.4)
    ax.text(8.5, ax.get_ylim()[1] * 0.92,
            'Proposed', ha='center', fontsize=8.5,
            color=C_E, style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls=':', zorder=1)
    ax.set_xlim(0.3, 9.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    from matplotlib.patches import Patch
    leg = [
        Patch(color=C_PUB[0], label='Published baselines (6)'),
        Patch(color=C_B,      label='PL-RBF Baseline'),
        Patch(color=C_K,      label='Wt. k-NN'),
        Patch(color=C_E,      label='PL-RBF Enhanced'),
    ]
    ax.legend(handles=leg, fontsize=8, loc='upper right',
              framealpha=0.9, ncol=2)

fig.suptitle(
    'Box Plot — D2 Test-Combined\n'
    'White = median; box = IQR; whiskers = 1.5\u00d7IQR; dots = outliers',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/plot_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 3: Bar Chart ──────────────────────────────────────────
print('Plotting bar chart ...')
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.subplots_adjust(wspace=0.42)

for ax, metric, ylabel, title in [
    (axes[0], 'eH',  '$\\varepsilon_H$ Mean (m)', 'Horizontal'),
    (axes[1], 'eV',  '$\\varepsilon_V$ Mean (m)', 'Vertical'),
    (axes[2], 'e3D', '$\\varepsilon_{3D}$ Mean (m)', '3-D Combined'),
]:
    vals = [df_per[df_per['Method'] == m][metric].mean()
            for m in METHOD_ORDER]
    x = np.arange(9)
    bars = ax.bar(x, vals, color=COLORS, edgecolor='white',
                  linewidth=0.5, width=0.7, zorder=2)
    bi = int(np.argmin(vals))
    bars[bi].set_edgecolor('gold')
    bars[bi].set_linewidth(2.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.text(x[bi], vals[bi] + 0.07, '\u2605',
            ha='center', fontsize=13, color='#F39C12', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, fontsize=8,
                       rotation=30, ha='right')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls=':', zorder=1)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

from matplotlib.patches import Patch
leg = [
    Patch(color=C_PUB[0], label='Published baselines'),
    Patch(color=C_B,       label='PL-RBF Baseline'),
    Patch(color=C_K,       label='Wt. k-NN'),
    Patch(color=C_E,       label='PL-RBF Enhanced (proposed)'),
]
fig.legend(handles=leg, loc='upper center', ncol=4,
           fontsize=8.5, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.04))
fig.suptitle('Method Comparison — D2 Test-Combined (199 positions)',
             fontsize=12, fontweight='bold', y=1.10)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/plot_bars.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Fig 4: Ablation Bars ──────────────────────────────────────
print('Plotting ablation ...')
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.42)

for ax, metric, col, ylabel, title in [
    (axes[0], 'eH', '#2980B9', '$\\varepsilon_H$ Mean (m)', 'Horizontal'),
    (axes[1], 'eV', '#E67E22', '$\\varepsilon_V$ Mean (m)', 'Vertical'),
]:
    vals = [df_abl[df_abl['Config'] == c][metric].mean()
            for c in ABL_ORDER]
    x = np.arange(5)
    bars = ax.bar(x, vals, color=[col] * 4 + [C_E],
                  edgecolor='white', linewidth=0.5,
                  width=0.65, zorder=2)
    bars[-1].set_edgecolor(C_E)
    bars[-1].set_linewidth(2)

    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=8.5,
                fontweight='bold' if i == 4 else 'normal')

    for i in range(1, 5):
        d = (vals[i] - vals[i - 1]) / vals[i - 1] * 100
        yy = max(vals[i - 1], vals[i]) + 0.05
        ax.annotate(
            f'{d:+.1f}%',
            xy=(x[i], vals[i] + 0.015), xytext=(x[i], yy),
            fontsize=7.5, ha='center', color=C_E,
            arrowprops=dict(arrowstyle='->', color=C_E, lw=0.9),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ABL_LABELS, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, ls=':', zorder=1)
    ax.set_ylim(0, max(vals) * 1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Ablation Study — D2 Test-Combined',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/plot_ablation.png', dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════
# 8.  PRINT FINAL TABLE
# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 65)
print(f"{'Method':<25} {'H_Mean':>8} {'V_Mean':>8} {'3D_Mean':>9}")
print('-' * 65)
for _, row in df_metrics.iterrows():
    print(f"{row['Method']:<25} {row['H_Mean']:8.4f} "
          f"{row['V_Mean']:8.4f} {row['3D_Mean']:9.4f}")
print('=' * 65)
print(f'\n\u2705  All outputs saved to: {OUT_DIR}/')
