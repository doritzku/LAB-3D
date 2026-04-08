# LAB-3D Data Files — Detailed Description

## Files

| File | Rows | Unique 3-D Positions | Role |
|------|------|---------------------|------|
| `D1_Train.csv` | 19,866 | 88 | Training / reference set |
| `D2_Test_Combined.csv` | 28,112 | 199 | Evaluation (2 overlap with D1) |
| `D3_Test_Distinct.csv` | 27,884 | 197 | Evaluation (0 overlap with D1) |

---

## Column Description

```
X       float   X-coordinate in metres from laboratory origin
Y       float   Y-coordinate in metres from laboratory origin
Z       float   Device height: one of {1.0, 1.5, 2.0} metres
Split   string  File role identifier
                  "train"         → D1_Train.csv
                  "test"          → D2_Test_Combined.csv
                  "test_distinct" → D3_Test_Distinct.csv
RSSI-1  float   Received Signal Strength (dBm) from Access Point 1
RSSI-2  float   Received Signal Strength (dBm) from Access Point 2
RSSI-3  float   Received Signal Strength (dBm) from Access Point 3
RSSI-4  float   Received Signal Strength (dBm) from Access Point 4
RSSI-5  float   Received Signal Strength (dBm) from Access Point 5
RSSI-6  float   Received Signal Strength (dBm) from Access Point 6
```

**Sentinel value:** `RSSI = −100 dBm` means the access point
was not detected during that scan. Exclude these before any
mathematical operation (mean, interpolation, distance matching).

---

## Coordinate System

```
Origin: South-West corner of CORDIoT laboratory
X-axis: Points East  (along 11 m wall)
Y-axis: Points North (along 14 m wall)
Z-axis: Points Up    (height above floor)

Room bounds:
  X ∈ [0, 11] m
  Y ∈ [0, 14] m
  Z ∈ {1.0, 1.5, 2.0} m
```

---

## Per-File Statistics

### D1 — Training Set

| Height (m) | Unique RPs | Raw Samples | Samples/RP |
|-----------|-----------|------------|-----------|
| 1.0 | 31 | 8,507 | 274 ± 16 |
| 1.5 | 30 | 3,733 | 124 ± 33 |
| 2.0 | 27 | 7,626 | 282 ± 24 |
| **Total** | **88** | **19,866** | **226 ± 78** |

### D2 — Test-Combined

| Height (m) | Unique TPs | Raw Samples | Samples/TP |
|-----------|-----------|------------|-----------|
| 1.0 | 86 | 10,790 | 126 ± 18 |
| 1.5 | 86 | 9,693 | 113 ± 9 |
| 2.0 | 27 | 7,629 | 283 ± 24 |
| **Total** | **199** | **28,112** | **141 ± 59** |

### D3 — Test-Distinct (Zero Overlap with D1)

| Height (m) | Unique TPs | Raw Samples | Samples/TP |
|-----------|-----------|------------|-----------|
| **Total** | **197** | **27,884** | **142 ± 59** |

---

## Per-AP Signal Statistics (D1 Training Fingerprints)

| AP | Min (dBm) | Max (dBm) | Mean (dBm) | STD (dBm) | STD/RP | Missing % |
|----|----------|----------|-----------|----------|--------|----------|
| AP-1 | −65 | −38 | −53.8 | 5.9 | 4.93 | 0.16 |
| AP-2 | −81 | −47 | −67.6 | 6.9 | 4.08 | 0.80 |
| AP-3 | −84 | −53 | −65.0 | 6.5 | 4.06 | 0.03 |
| AP-4 | −87 | −50 | −68.7 | 7.7 | 4.08 | 0.35 |
| AP-5 | −81 | −48 | −66.6 | 7.2 | 3.57 | 0.25 |
| AP-6 | −80 | −50 | −65.3 | 6.8 | 3.88 | 0.31 |

STD/RP = mean within-position standard deviation (temporal noise).

---

## Data Collection Protocol

1. ESP32 board fixed on adjustable tripod at target height
2. RSSI recorded continuously for **5 minutes** per height
   (training) or **2 minutes** (test)
3. Device held in **4 orientations**: Up, Down, Left, Right
4. Sessions conducted on **multiple days and nights** to
   capture temporal channel variability
5. All measurements at a single (X, Y, Z) position form
   one group in the CSV

---

## Preprocessing Notes

The distributed CSV files contain **raw per-scan RSSI values**
(one row = one scan). For fingerprinting, aggregate per position:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('D1_Train.csv')
df[df == -100] = np.nan          # mark no-signal as NaN

# Simple median fingerprint per 3-D position
fp = df.groupby(['X','Y','Z'])[[
    'RSSI-1','RSSI-2','RSSI-3',
    'RSSI-4','RSSI-5','RSSI-6'
]].median().reset_index()

fp = fp.fillna(-100)             # restore sentinel for matching
```

For the **Enhanced PL-RBF** preprocessing (Gaussian 3σ filter +
median), see `code/plrbf_enhanced.py → _aggregate()`.

---

## Overlap Between Files

| Pair | Overlapping positions |
|------|-----------------------|
| D1 ∩ D2 | 2 positions |
| D1 ∩ D3 | 0 positions (strictly disjoint) |
| D2 ∩ D3 | Not checked (independent grids) |

D3 is therefore a **strictly independent** generalisation test.
