# LAB-3D: A Three-Dimensional Wi-Fi RSSI Fingerprint Dataset

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset DOI](https://img.shields.io/badge/DOI-pending-blue)](#)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Sensors%20Journal%20(under%20review)-red)](#)

---

## Overview

**LAB-3D** is the first publicly available Wi-Fi RSSI fingerprint
dataset spanning **three controlled height levels**
(z ∈ {1.0, 1.5, 2.0} m) within a single laboratory floor.
It is designed to benchmark **height-aware (3-D) indoor
positioning** algorithms that go beyond the conventional
horizontal-only assumption.

| Property | Value |
|----------|-------|
| Environment | CORDIoT Lab, IIIT Allahabad, India |
| Room size | 11 m × 14 m (154 m²) |
| Access points | 6 × IEEE 802.11n @ 2.4 GHz |
| Height levels | z ∈ {1.0, 1.5, 2.0} m |
| Device | ESP32 on height-adjustable tripod |
| Training samples | 19,866 raw samples @ 88 unique 3-D RPs |
| Test D2 positions | 199 positions (Test-Combined) |
| Test D3 positions | 197 positions (Test-Distinct, 0 overlap) |
| Collection time | ~26 hours (day + night sessions) |

---

## Repository Structure

```
LAB-3D/
│
├── README.md                   ← This file
├── LICENSE                     ← CC BY 4.0
├── CITATION.cff                ← Citation metadata
│
├── data/
│   ├── README_data.md          ← Detailed data description
│   ├── D1_Train.csv            ← 19,866 training samples (88 RPs)
│   ├── D2_Test_Combined.csv    ← 28,112 test samples (199 positions)
│   └── D3_Test_Distinct.csv    ← 27,884 test samples (197 positions)
│
├── code/
│   ├── README_code.md          ← Code usage instructions
│   ├── plrbf_enhanced.py       ← Enhanced PL-RBF algorithm
│   ├── comparison_methods.py   ← 6 baseline methods
│   ├── ablation_study.py       ← Ablation configs C0–C4
│   └── run_final_results.py    ← Master script (all methods + figures)
│
└── figures/
    ├── fig_environment.png     ← Floor plan + lab photo
    ├── fig_dataset_3d.png      ← 3-D scatter of D1/D2/D3
    └── fig_radiomap.png        ← Interpolated RSSI radio maps
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### 2. Run all methods and generate figures
```bash
cd code/
python run_final_results.py
```

Results saved to `./results_final/`:
- `per_sample.csv` — per-position H/V/3D errors
- `metrics.csv`    — summary table
- `plot_cdf.png`, `plot_boxplot.png`, `plot_bars.png`, `plot_ablation.png`

### 3. Use Enhanced PL-RBF in your own code
```python
import pandas as pd
from plrbf_enhanced import PathLossRBF_Enhanced

train = pd.read_csv('data/D1_Train.csv')
test  = pd.read_csv('data/D2_Test_Combined.csv')

model = PathLossRBF_Enhanced(sigma_thresh=3.0, smoothing=0.3, k=3)
model.fit(train)

test_med = test.groupby(['X','Y','Z'])[[
    'RSSI-1','RSSI-2','RSSI-3','RSSI-4','RSSI-5','RSSI-6'
]].median().reset_index()
test_med[['RSSI-1','RSSI-2','RSSI-3',
          'RSSI-4','RSSI-5','RSSI-6']] = \
    test_med[['RSSI-1','RSSI-2','RSSI-3',
              'RSSI-4','RSSI-5','RSSI-6']].fillna(-100)

predictions = model.predict_position(test_med)
# predictions: (N, 3) array of [x, y, z] estimates
```

---

## Key Results (D2, 199 positions)

| Method | εH Mean (m) | εV Mean (m) | ε3D Mean (m) |
|--------|------------|------------|-------------|
| IDW | 2.273 | 0.415 | 2.357 |
| APLM | 2.184 | 0.452 | 2.274 |
| M-PLM | 2.332 | 0.492 | 2.448 |
| RBF-Network | 2.397 | 0.256 | 2.441 |
| Kriging | 1.844 | 0.344 | 1.916 |
| RBF-Direct | 1.675 | 0.354 | 1.761 |
| Wt. k-NN (C0) | 1.980 | 0.344 | 2.036 |
| PL-RBF Baseline | 1.598 | 0.349 | 1.682 |
| **PL-RBF Enhanced** | **1.344** | **0.275** | **1.404** |

Results on D3 differ by < 0.02 m from D2 on every metric.

---

## Dataset Description

### Column Format
All three CSV files share the same column structure:

| Column | Type | Description |
|--------|------|-------------|
| `X` | float | X-coordinate (metres, from lab origin) |
| `Y` | float | Y-coordinate (metres, from lab origin) |
| `Z` | float | Height level: 1.0, 1.5, or 2.0 m |
| `Split` | string | Role: `train`, `test`, or `test_distinct` |
| `RSSI-1` | float | RSSI from AP-1 (dBm); −100 = no signal |
| `RSSI-2` | float | RSSI from AP-2 (dBm) |
| `RSSI-3` | float | RSSI from AP-3 (dBm) |
| `RSSI-4` | float | RSSI from AP-4 (dBm) |
| `RSSI-5` | float | RSSI from AP-5 (dBm) |
| `RSSI-6` | float | RSSI from AP-6 (dBm) |

### File Summary

| File | Rows | Unique Positions | Description |
|------|------|-----------------|-------------|
| `D1_Train.csv` | 19,866 | 88 | Training / reference fingerprints |
| `D2_Test_Combined.csv` | 28,112 | 199 | Test set (2 positions overlap D1) |
| `D3_Test_Distinct.csv` | 27,884 | 197 | Test set (0 overlap with D1) |

### Sentinel Value
`RSSI = −100 dBm` indicates **no signal detected** (hardware
sentinel). These should be excluded before any computation.
Missing-AP fraction: D1 = 0.10%, D2 = 0.00%, D3 = 0.00%.

---

## Environment Details

- **Location:** CORDIoT Laboratory, IIIT Allahabad, India
- **Dimensions:** 11 m × 14 m (154 m²)
- **Features:** Furniture, computers, glass partition
  (divides room into two halves — creates asymmetric multipath)
- **Access Points:** 6 × IEEE 802.11n @ 2.4 GHz
  (including 1 central Aruba router)
- **Device:** ESP32 development board on adjustable tripod
- **Orientations:** 4 per position (Up, Down, Left, Right)
- **Training duration:** 5 min per position per height
- **Test duration:** 2 min per position per height
- **Sessions:** Multiple day + night sessions
- **Total effort:** ~26 hours

---

## Citation

If you use LAB-3D in your research, please cite:

```bibtex
@article{kumar2024lab3d,
  author  = {Kumar, Ritesh and Chaurasiya, Vijay Kumar},
  title   = {{LAB-3D}: Measurement, Characterization, and
             Height-Aware Interpolation of a Three-Dimensional
             {Wi-Fi} {RSSI} Fingerprint Dataset},
  journal = {Under Review only},
  year    = {2024},
  note    = {Under review}
}
```

---

## License

This dataset and code are released under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
You are free to share and adapt the material for any purpose,
provided you give appropriate credit.

See [LICENSE](LICENSE) for full terms.

---

## Contact

**Dr. Ritesh Kumar**
Department of Electronics and Communication Engineering,
Indian Institute of Technology Roorkee, India
✉ dr.riteshk94@gmail.com

**Prof. Vijay Kumar Chaurasiya**
Indian Institute of Information Technology Allahabad, India
✉ vijayk@iiita.ac.in
