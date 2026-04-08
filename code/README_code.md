# LAB-3D Code — Usage Guide

## Requirements

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

Tested with Python 3.9–3.13.

---

## Files

| File | Description |
|------|-------------|
| `run_all_results.py` | **Master script** — runs all 9 methods, saves CSVs + figures |
| `plrbf_enhanced.py` | Enhanced PL-RBF + Baseline PL-RBF classes |
| `comparison_methods.py` | 6 published baseline methods |
| `ablation_study.py` | Ablation configs C0–C4 (AblationPLRBF class) |

---

## Run Everything

```bash
# Place all scripts + data CSVs in same folder, then:
python run_all_results.py
```

Outputs saved to `./results_final/`:

```
results_final/
├── per_sample.csv          ← per-position errors (all 9 methods)
├── per_sample_ablation.csv ← per-position errors (C0–C4)
├── metrics.csv             ← summary: mean, P90, RMSE
├── plot_cdf.png            ← empirical CDF
├── plot_boxplot.png        ← box plot (εH)
├── plot_bars.png           ← bar chart H/V/3D
└── plot_ablation.png       ← ablation progression
```

---

## Use Enhanced PL-RBF in Your Project

```python
import pandas as pd
import numpy as np
from plrbf_enhanced import PathLossRBF_Enhanced

# ── Load data ─────────────────────────────────────────────────
train = pd.read_csv('D1_Train.csv')
test  = pd.read_csv('D2_Test_Combined.csv')

AP_COLS = ['RSSI-1','RSSI-2','RSSI-3','RSSI-4','RSSI-5','RSSI-6']

# ── Aggregate test to one row per position ────────────────────
test_med = test.groupby(['X','Y','Z'])[AP_COLS].median().reset_index()
test_med[AP_COLS] = test_med[AP_COLS].fillna(-100)

# ── Train ─────────────────────────────────────────────────────
model = PathLossRBF_Enhanced(
    sigma_thresh = 3.0,   # Gaussian 3σ outlier threshold
    smoothing    = 0.3,   # TPS-RBF regularisation
    k            = 3      # weighted k-NN neighbours
)
model.fit(train)   # pass raw training DataFrame

# ── Predict ───────────────────────────────────────────────────
pred_pos = model.predict_position(test_med)
# pred_pos: numpy array (N, 3) → columns = [x_hat, y_hat, z_hat]

# ── Evaluate ──────────────────────────────────────────────────
true_pos = test_med[['X','Y','Z']].values
eH  = np.sqrt((pred_pos[:,0]-true_pos[:,0])**2 +
               (pred_pos[:,1]-true_pos[:,1])**2)
eV  = np.abs(pred_pos[:,2] - true_pos[:,2])
e3D = np.sqrt(eH**2 + eV**2)

print(f"εH Mean : {eH.mean():.4f} m")
print(f"εV Mean : {eV.mean():.4f} m")
print(f"ε3D Mean: {e3D.mean():.4f} m")
```

---

## Run Only Baseline Methods

```python
from comparison_methods import (
    IDWInterpolator, RBFDirectInterpolator,
    APLMInterpolator, MPLMInterpolator,
    RBFNetworkInterpolator, KrigingInterpolator,
    PathLossRBF
)
import pandas as pd

train = pd.read_csv('D1_Train.csv')
test  = pd.read_csv('D2_Test_Combined.csv')

AP_COLS = ['RSSI-1','RSSI-2','RSSI-3','RSSI-4','RSSI-5','RSSI-6']

# Aggregate to fingerprints
def make_fp(df):
    fp = df.groupby(['X','Y','Z'])[AP_COLS].median().reset_index()
    fp[AP_COLS] = fp[AP_COLS].fillna(-100)
    return fp

train_fp = make_fp(train)
test_fp  = make_fp(test)

# Run Kriging (replace with any method above)
model = KrigingInterpolator()
model.fit(train_fp)
pred = model.predict_position(test_fp)
```

---

## Ablation Study

```python
from ablation_study import AblationPLRBF
import pandas as pd

train = pd.read_csv('D1_Train.csv')
test_fp = ...   # aggregated test fingerprints (see above)

configs = {
    'C0: Wt. k-NN':
        dict(use_rbf=False, use_pl=False, use_gauss=False,
             smoothing=1.0, k=5, use_wknn=True),
    'C1: +Path-Loss':
        dict(use_rbf=True, use_pl=True, use_gauss=False,
             smoothing=1.0, k=1, use_wknn=False),
    'C2: +Gauss 3σ':
        dict(use_rbf=True, use_pl=True, use_gauss=True,
             smoothing=1.0, k=1, use_wknn=False),
    'C3: +Smooth 0.3':
        dict(use_rbf=True, use_pl=True, use_gauss=True,
             smoothing=0.3, k=1, use_wknn=False),
    'C4: +W-3NN (Full)':
        dict(use_rbf=True, use_pl=True, use_gauss=True,
             smoothing=0.3, k=3, use_wknn=True),
}

for name, cfg in configs.items():
    model = AblationPLRBF(**cfg)
    model.fit(train)
    pred = model.predict_position(test_fp)
    print(f"{name}: pred shape = {pred.shape}")
```

---

## Expected Results (D2, 199 positions)

```
Method               εH Mean    εV Mean    ε3D Mean
----------------------------------------------------
IDW                   2.2731     0.4146     2.3572
RBF-Direct            1.6752     0.3543     1.7610
APLM                  2.1840     0.4523     2.2744
M-PLM                 2.3325     0.4925     2.4478
RBF-Network           2.3972     0.2563     2.4415
Kriging               1.8440     0.3442     1.9164
PL-RBF Baseline       1.5975     0.3492     1.6822
PL-RBF Enhanced       1.3439     0.2750     1.4036
```

If your results differ by > 0.01 m, check:
1. Sentinel handling: `−100` must be excluded, not used as RSSI
2. Test aggregation: median per (X, Y, Z) group
3. Random seed: `np.random.seed(42)` at top of script

---

## Reproducing Paper Figures

All figures in the paper were generated by `run_all_results.py`.
The script uses **only the original method classes** listed above —
no synthetic or approximated data.
