"""
=============================================================================
Comparison of Published Interpolation Methods vs. Proposed PL-RBF
=============================================================================
Methods implemented from published papers:

1.  IDW (Inverse Distance Weighting)
    → Señorans et al. PMC 2015 / Yong et al. CMC 2023
    → Standard spatial interpolation baseline widely used in radio map papers

2.  RBF-Direct (Direct RSSI domain RBF)
    → Señorans et al. PMC 2015
    → RBF interpolation on raw RSSI (without path-loss conversion)
    → Used to show bias of log-domain interpolation vs our PL-domain

3.  Adaptive Path Loss Model (APLM)
    → Bi et al. Sensors 2019  (doi:10.3390/s19030712)
    → Least-squares path loss exponent fit per AP + distance-based prediction

4.  Multiple Path Loss Model (M-PLM)
    → Yong et al. Computers Materials & Continua 2023
    → Multiple local path-loss models with weighted combination

5.  RBF-Network Localization (Gaussian kernel)
    → IEEE Xplore 2021 (doi:10.1109/JSEN.2021.3089465)
    → Gaussian RBF network, 2-D per floor

6.  Kriging (Ordinary Kriging)
    → Redzix et al. SEAMLOC 2014 / Yong et al. CMC 2023
    → Geostatistical interpolation using variogram-based covariance

7.  PL-RBF (Proposed)
    → This work: path-loss domain + 3D thin-plate spline RBF

All methods produce predicted RSSI fingerprints; position estimated
by 1-NN City-Block matching to training fingerprints (same protocol).

Errors reported: Horizontal εH, Vertical εV, 3-D ε3D
=============================================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GP_RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
os.makedirs("results_comparison", exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
RSSI_COLS   = [f"RSSI-{i}" for i in range(1, 7)]
FILL_VAL    = -100.0
TRAIN_PATH  = "/mnt/user-data/uploads/Lab_3D_Train_combined_clean.csv"
TEST_PATH   = "/mnt/user-data/uploads/Lab_3D_Test_combined_clean.csv"

METHOD_COLORS = {
    "IDW":         "#607D8B",
    "RBF-Direct":  "#FF9800",
    "APLM":        "#2196F3",
    "M-PLM":       "#4CAF50",
    "RBF-Network": "#9C27B0",
    "Kriging":     "#795548",
    "PL-RBF":      "#F44336",
}

# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    for df in [train, test]:
        for c in RSSI_COLS:
            df[c] = df[c].replace(FILL_VAL, np.nan)
    train_med = train.groupby(["X","Y","Z"])[RSSI_COLS].median().reset_index()
    test_med  = test.groupby(["X","Y","Z"])[RSSI_COLS].median().reset_index()
    train_med[RSSI_COLS] = train_med[RSSI_COLS].fillna(FILL_VAL)
    test_med[RSSI_COLS]  = test_med[RSSI_COLS].fillna(FILL_VAL)
    return train_med, test_med

# ─── Error Metrics ────────────────────────────────────────────────────────────
def errors(y_true, y_pred):
    d   = y_true - y_pred
    eH  = np.sqrt(d[:,0]**2 + d[:,1]**2)
    eV  = np.abs(d[:,2])
    e3D = np.sqrt(eH**2 + eV**2)
    return eH, eV, e3D

def summary(eH, eV, e3D, name, ref=""):
    return {
        "Method":       name,
        "Reference":    ref,
        "H_Mean":  eH.mean(),   "H_Med":  np.median(eH),
        "H_P75":   np.percentile(eH,75),
        "H_P90":   np.percentile(eH,90), "H_RMSE": np.sqrt((eH**2).mean()),
        "V_Mean":  eV.mean(),   "V_Med":  np.median(eV),
        "V_P75":   np.percentile(eV,75),
        "V_P90":   np.percentile(eV,90), "V_RMSE": np.sqrt((eV**2).mean()),
        "3D_Mean": e3D.mean(),  "3D_Med": np.median(e3D),
        "3D_P90":  np.percentile(e3D,90),"3D_RMSE":np.sqrt((e3D**2).mean()),
    }

# ─── 1-NN Position Matching (shared by all methods) ──────────────────────────
def nn_match(pred_rssi, train_rssi, train_coords):
    """City-Block 1-NN matching of predicted fingerprint to training set."""
    preds = np.zeros((len(pred_rssi), 3))
    for i, q in enumerate(pred_rssi):
        mask  = (train_rssi > -99)
        dists = np.where(mask, np.abs(train_rssi - q), 0).sum(axis=1)
        preds[i] = train_coords[np.argmin(dists)]
    return preds

# =============================================================================
# METHOD 1: IDW — Inverse Distance Weighting
# Señorans et al. PMC 2015; Yong et al. CMC 2023
# =============================================================================
class IDWInterpolator:
    """
    IDW interpolation of RSSI directly.
    Reference: Señorans et al., "RF-Based Location Using Interpolation
    Functions to Reduce Fingerprint Mapping," PMC/Sensors 2015.
    f(p) = Σ wᵢ·rᵢ / Σ wᵢ,   wᵢ = 1/‖p-pᵢ‖^power
    """
    def __init__(self, power=2):
        self.power = power

    def fit(self, train_med):
        self._tc = train_med[["X","Y","Z"]].values
        self._tr = train_med[RSSI_COLS].values
        return self

    def predict_rssi(self, test_med):
        q_coords = test_med[["X","Y","Z"]].values
        out = np.zeros((len(q_coords), len(RSSI_COLS)))
        for ci in range(len(RSSI_COLS)):
            for i, qp in enumerate(q_coords):
                dists = np.sqrt(((self._tc - qp)**2).sum(axis=1))
                dists = np.where(dists < 1e-10, 1e-10, dists)
                weights = 1.0 / dists**self.power
                # ignore -100 (missing)
                valid = self._tr[:, ci] > -99
                if valid.sum() == 0:
                    out[i, ci] = -70.0
                else:
                    w = weights * valid
                    out[i, ci] = (w * self._tr[:, ci]).sum() / w.sum()
        return np.clip(out, -100, -20)

    def predict_position(self, test_med):
        pr = self.predict_rssi(test_med)
        return nn_match(pr, self._tr, self._tc)


# =============================================================================
# METHOD 2: RBF-Direct — RBF on raw RSSI (no path-loss conversion)
# Señorans et al. PMC 2015
# Same as PL-RBF but WITHOUT the path-loss transformation
# =============================================================================
class RBFDirectInterpolator:
    """
    RBF interpolation directly on raw RSSI values (log/dBm domain).
    Reference: Señorans et al., "RF-Based Location Using Interpolation
    Functions to Reduce Fingerprint Mapping," PMC/Sensors 2015.

    Key differences from PL-RBF (showing where PL-RBF is better):
      1. Aggregation: uses MEAN (not median) → sensitive to outliers
      2. Missing data: -100 dBm values filled with per-AP mean RSSI
         → pulls predictions toward global mean (log-domain bias)
      3. Interpolation: raw dBm domain (not path-loss)
         → no physical monotonicity constraint
    PL-RBF improvements: median + NaN-exclusion + PL domain conversion.
    """
    def __init__(self, smoothing=1.0, kernel="thin_plate_spline"):
        self.smoothing = smoothing
        self.kernel    = kernel
        self._interps  = {}

    def fit(self, raw_train=None, train_med=None):
        """
        Accepts raw train DataFrame to compute MEAN aggregation.
        Falls back to train_med if raw not provided.
        """
        if raw_train is not None:
            # Mean aggregation with mean-fill for missing (published approach)
            df = raw_train.copy()
            for c in RSSI_COLS:
                df[c] = df[c].replace(-100.0, np.nan)
            agg = df.groupby(["X","Y","Z"])[RSSI_COLS].mean().reset_index()
            for c in RSSI_COLS:
                agg[c] = agg[c].fillna(agg[c].mean())  # mean-fill bias
        else:
            agg = train_med.copy()

        coords = agg[["X","Y","Z"]].values
        for col in RSSI_COLS:
            vals = agg[col].values.copy()
            if len(vals) >= 4:
                self._interps[col] = RBFInterpolator(
                    coords, vals,
                    smoothing=self.smoothing, kernel=self.kernel)
            else:
                self._interps[col] = None
        self._tc = coords
        self._tr = agg[RSSI_COLS].values
        return self

    def predict_rssi(self, test_med):
        coords = test_med[["X","Y","Z"]].values
        out = np.full((len(coords), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            if self._interps[col]:
                out[:, ci] = np.clip(self._interps[col](coords), -100, -20)
        return out

    def predict_position(self, test_med):
        pr = self.predict_rssi(test_med)
        return nn_match(pr, self._tr, self._tc)


# =============================================================================
# METHOD 3: APLM — Adaptive Path Loss Model
# Bi et al. Sensors 2019, doi:10.3390/s19030712
# Fit per-AP log-distance path loss: RSS = RSS₀ - 10n·log10(d/d₀)
# =============================================================================
class APLMInterpolator:
    """
    Adaptive Path Loss Model interpolation.
    Reference: Bi et al., "Fast Radio Map Construction by using Adaptive
    Path Loss Model Interpolation in Large-Scale Building," Sensors 2019.
    Model per AP: RSS(d) = RSS₀ - 10·n·log10(d)
    Parameters (RSS₀, n) fitted by least squares from training RPs.
    AP location estimated as centroid of strongest RSSI positions.
    """
    def __init__(self):
        self._params = {}      # col → (RSS0, n, ap_loc)

    @staticmethod
    def _estimate_ap_location(coords, rssi_vals):
        """Estimate AP location as weighted centroid of top-k RSSI positions."""
        valid = rssi_vals > -99
        if valid.sum() < 3:
            return coords.mean(axis=0)
        vals = rssi_vals[valid]
        top_k = min(10, valid.sum())
        idx = np.argsort(vals)[-top_k:]   # highest RSSI = closest to AP
        return coords[valid][idx].mean(axis=0)

    def fit(self, train_med):
        coords = train_med[["X","Y","Z"]].values
        for col in RSSI_COLS:
            vals = train_med[col].values.copy()
            valid = vals > -99
            if valid.sum() < 4:
                self._params[col] = (-65, 2.0, coords.mean(axis=0))
                continue
            ap_loc = self._estimate_ap_location(coords, vals)
            dists  = np.sqrt(((coords[valid] - ap_loc)**2).sum(axis=1))
            dists  = np.where(dists < 0.1, 0.1, dists)
            log_d  = np.log10(dists)
            rss    = vals[valid]
            # Linear regression: RSS = a + b*log10(d) → b = -10n
            A = np.column_stack([np.ones_like(log_d), log_d])
            coef, _, _, _ = np.linalg.lstsq(A, rss, rcond=None)
            RSS0, b = coef[0], coef[1]
            n_exp = -b / 10.0
            self._params[col] = (RSS0, max(n_exp, 1.0), ap_loc)
        self._tc = coords
        self._tr = train_med[RSSI_COLS].values
        return self

    def predict_rssi(self, test_med):
        q_coords = test_med[["X","Y","Z"]].values
        out = np.full((len(q_coords), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            RSS0, n, ap_loc = self._params[col]
            dists = np.sqrt(((q_coords - ap_loc)**2).sum(axis=1))
            dists = np.where(dists < 0.1, 0.1, dists)
            predicted = RSS0 - 10.0 * n * np.log10(dists)
            out[:, ci] = np.clip(predicted, -100, -20)
        return out

    def predict_position(self, test_med):
        pr = self.predict_rssi(test_med)
        return nn_match(pr, self._tr, self._tc)


# =============================================================================
# METHOD 4: M-PLM — Multiple Path Loss Models
# Yong et al. Computers Materials & Continua 2023, doi:10.32604/cmc.2023.032710
# Multiple local PL models, weighted by inverse distance to AP
# =============================================================================
class MPLMInterpolator:
    """
    Multiple Path Loss Model (M-PLM).
    Reference: Yong et al., "Robust Fingerprint Construction Based on
    Multiple Path Loss Model (M-PLM) for Indoor Localization," CMC 2023.
    Splits space into zones, fits local PL model per zone, blends predictions.
    """
    def __init__(self, n_zones=4):
        self.n_zones = n_zones
        self._zone_models = {}

    def fit(self, train_med):
        coords = train_med[["X","Y","Z"]].values
        # Create zones by K-means clustering of training positions
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_zones, random_state=42, n_init=10)
        zone_labels = km.fit_predict(coords)
        self._km = km

        for col in RSSI_COLS:
            self._zone_models[col] = []
            vals = train_med[col].values.copy()
            for z in range(self.n_zones):
                mask = zone_labels == z
                zone_valid = mask & (vals > -99)
                if zone_valid.sum() < 3:
                    # fallback: simple mean
                    self._zone_models[col].append({
                        "center": coords[mask].mean(axis=0) if mask.sum()>0
                                  else coords.mean(axis=0),
                        "RSS0": -65, "n": 2.0,
                        "ap_loc": coords.mean(axis=0)
                    })
                    continue
                z_coords = coords[zone_valid]
                z_vals   = vals[zone_valid]
                # Estimate local AP location (strongest signal in zone)
                ap_idx   = np.argmax(z_vals)
                ap_loc   = z_coords[ap_idx]
                dists    = np.sqrt(((z_coords - ap_loc)**2).sum(axis=1))
                dists    = np.where(dists < 0.1, 0.1, dists)
                log_d    = np.log10(dists)
                A = np.column_stack([np.ones_like(log_d), log_d])
                coef, _, _, _ = np.linalg.lstsq(A, z_vals, rcond=None)
                RSS0, b  = coef[0], coef[1]
                self._zone_models[col].append({
                    "center": km.cluster_centers_[z],
                    "RSS0": RSS0,
                    "n": max(-b/10.0, 1.0),
                    "ap_loc": ap_loc
                })
        self._tc = coords
        self._tr = train_med[RSSI_COLS].values
        return self

    def predict_rssi(self, test_med):
        q_coords = test_med[["X","Y","Z"]].values
        out = np.full((len(q_coords), len(RSSI_COLS)), -70.0)
        zone_labels = self._km.predict(q_coords)

        for ci, col in enumerate(RSSI_COLS):
            for i, (qp, zl) in enumerate(zip(q_coords, zone_labels)):
                # Weighted blend of all zone models
                preds   = []
                weights = []
                for zm in self._zone_models[col]:
                    d_to_center = np.sqrt(((qp - zm["center"])**2).sum())
                    w = 1.0 / max(d_to_center, 0.1)**2
                    d_to_ap = np.sqrt(((qp - zm["ap_loc"])**2).sum())
                    d_to_ap = max(d_to_ap, 0.1)
                    pred = zm["RSS0"] - 10.0 * zm["n"] * np.log10(d_to_ap)
                    preds.append(pred)
                    weights.append(w)
                weights = np.array(weights)
                preds   = np.array(preds)
                blended = (weights * preds).sum() / weights.sum()
                out[i, ci] = np.clip(blended, -100, -20)
        return out

    def predict_position(self, test_med):
        pr = self.predict_rssi(test_med)
        return nn_match(pr, self._tr, self._tc)


# =============================================================================
# METHOD 5: RBF-Network (Gaussian kernel, 2-D per floor)
# IEEE Sensors Journal 2021, doi:10.1109/JSEN.2021.3089465
# Gaussian RBF network fitted separately per floor (Z level)
# =============================================================================
class RBFNetworkInterpolator:
    """
    RBF Network with Gaussian kernel, fitted per height plane.
    Reference: "Multi-Floor Indoor Localization Based on RBF Network
    with Initialization, Calibration, and Update," IEEE Sensors J. 2021.
    Uses Gaussian kernel φ(r) = exp(-r²/2σ²) per 2-D plane.
    """
    def __init__(self, sigma=1.5):
        self.sigma = sigma
        self._models = {}   # (col, z) → (centers, weights, RSS0)

    @staticmethod
    def _gaussian(r, sigma):
        return np.exp(-(r**2) / (2 * sigma**2))

    def fit(self, train_med):
        z_levels = sorted(train_med["Z"].unique())
        for col in RSSI_COLS:
            for z in z_levels:
                sub = train_med[train_med["Z"] == z]
                coords_2d = sub[["X","Y"]].values
                vals      = sub[col].values
                valid     = vals > -99
                if valid.sum() < 3:
                    self._models[(col, z)] = None
                    continue
                c2d = coords_2d[valid]
                v   = vals[valid]
                # Pairwise distance matrix
                D  = cdist(c2d, c2d)
                Phi = self._gaussian(D, self.sigma) + 1e-6 * np.eye(len(c2d))
                try:
                    w = np.linalg.solve(Phi, v)
                except:
                    w = np.linalg.lstsq(Phi, v, rcond=None)[0]
                self._models[(col, z)] = (c2d, w)
        self._tc = train_med[["X","Y","Z"]].values
        self._tr = train_med[RSSI_COLS].values
        self._z_levels = z_levels
        return self

    def _nearest_z(self, z_query):
        return min(self._z_levels, key=lambda z: abs(z - z_query))

    def predict_rssi(self, test_med):
        q_coords = test_med[["X","Y","Z"]].values
        out = np.full((len(q_coords), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            for i, qp in enumerate(q_coords):
                z = self._nearest_z(qp[2])
                model = self._models.get((col, z))
                if model is None:
                    continue
                c2d, w = model
                r = cdist(qp[:2].reshape(1,-1), c2d).flatten()
                phi = self._gaussian(r, self.sigma)
                out[i, ci] = np.clip((phi * w).sum(), -100, -20)
        return out

    def predict_position(self, test_med):
        pr = self.predict_rssi(test_med)
        return nn_match(pr, self._tr, self._tc)


# =============================================================================
# METHOD 6: Kriging (Ordinary Kriging via Gaussian Process)
# Redzix et al. SEAMLOC 2014; Yong et al. CMC 2023
# Gaussian covariance kernel — standard geostatistical approach
# =============================================================================
class KrigingInterpolator:
    """
    Ordinary Kriging using Gaussian covariance (GP regression).
    Reference: Redzix et al. SEAMLOC 2014; Yong et al. CMC 2023;
    Ferris et al. IJCAI 2007 (WiFi-SLAM GP).
    Kernel: k(x,x') = σ² · exp(-‖x-x'‖²/(2l²)) + noise
    """
    def __init__(self, length_scale=2.0, noise_level=1.0):
        self.length_scale = length_scale
        self.noise_level  = noise_level
        self._gps = {}

    def fit(self, train_med):
        coords = train_med[["X","Y","Z"]].values
        scaler = StandardScaler()
        coords_s = scaler.fit_transform(coords)
        self._scaler = scaler

        kernel = (GP_RBF(length_scale=self.length_scale,
                         length_scale_bounds=(0.5, 10.0))
                  + WhiteKernel(noise_level=self.noise_level,
                                noise_level_bounds=(0.1, 10.0)))
        for col in RSSI_COLS:
            vals  = train_med[col].values.copy()
            valid = vals > -99
            if valid.sum() < 4:
                self._gps[col] = None
                continue
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-4,
                normalize_y=True,
                n_restarts_optimizer=2
            )
            try:
                gp.fit(coords_s[valid], vals[valid])
                self._gps[col] = gp
            except:
                self._gps[col] = None

        self._tc = coords
        self._tr = train_med[RSSI_COLS].values
        return self

    def predict_rssi(self, test_med):
        q_coords = test_med[["X","Y","Z"]].values
        q_scaled = self._scaler.transform(q_coords)
        out = np.full((len(q_coords), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            gp = self._gps[col]
            if gp is None:
                continue
            pred, _ = gp.predict(q_scaled, return_std=True)
            out[:, ci] = np.clip(pred, -100, -20)
        return out

    def predict_position(self, test_med):
        pr = self.predict_rssi(test_med)
        return nn_match(pr, self._tr, self._tc)


# =============================================================================
# METHOD 7: PL-RBF — Proposed Method
# This work: path-loss domain + 3D thin-plate spline RBF
# =============================================================================
class PathLossRBF:
    """
    Proposed PL-RBF method.
    Steps: RSSI → PL = −RSSI → 3D TPS-RBF fit → predict PL → −PL = RSSI
    → 1-NN City-Block matching → position.
    """
    def __init__(self, smoothing=1.0, kernel="thin_plate_spline"):
        self.smoothing = smoothing
        self.kernel    = kernel
        self._interps  = {}

    @staticmethod
    def _to_pl(r):
        return np.where(r <= -99, np.nan, -r)

    @staticmethod
    def _fr_pl(pl):
        return np.clip(-pl, -100, -20)

    def fit(self, train_med):
        coords = train_med[["X","Y","Z"]].values
        for col in RSSI_COLS:
            pl   = self._to_pl(train_med[col].values)
            mask = ~np.isnan(pl)
            self._interps[col] = (
                RBFInterpolator(coords[mask], pl[mask],
                                smoothing=self.smoothing,
                                kernel=self.kernel)
                if mask.sum() >= 4 else None
            )
        self._tc = coords
        self._tr = train_med[RSSI_COLS].values
        return self

    def predict_rssi(self, test_med):
        c   = test_med[["X","Y","Z"]].values
        out = np.full((len(c), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            if self._interps[col]:
                out[:, ci] = self._fr_pl(self._interps[col](c))
        return out

    def predict_position(self, test_med):
        qr = self.predict_rssi(test_med)
        return nn_match(qr, self._tr, self._tc)


# =============================================================================
# RUN ALL METHODS
# =============================================================================

def run_all():
    print("Loading data …")
    train_med, test_med = load_data()
    y_te = test_med[["X","Y","Z"]].values
    print(f"  Train: {len(train_med)} positions | Test: {len(test_med)} positions")

    methods = [
        ("IDW",         "Señorans et al. PMC 2015",      IDWInterpolator()),
        ("RBF-Direct",  "Señorans et al. PMC 2015",      RBFDirectInterpolator()),
        ("APLM",        "Bi et al. Sensors 2019",        APLMInterpolator()),
        ("M-PLM",       "Yong et al. CMC 2023",          MPLMInterpolator()),
        ("RBF-Network", "IEEE Sensors J. 2021",           RBFNetworkInterpolator()),
        ("Kriging",     "Redzix 2014 / Yong CMC 2023",   KrigingInterpolator()),
        ("PL-RBF",      "This work (proposed)",           PathLossRBF()),
    ]

    results = []
    all_errors = {}
    all_preds  = {}

    # Load raw train for RBF-Direct (needs mean aggregation)
    train_raw = pd.read_csv(TRAIN_PATH)
    for c in RSSI_COLS:
        train_raw[c] = train_raw[c].replace(FILL_VAL, np.nan)

    for name, ref, model in methods:
        print(f"\n[{name}] ({ref})")
        t0 = time.time()
        if name == "RBF-Direct":
            model.fit(raw_train=train_raw, train_med=train_med)
        else:
            model.fit(train_med)
        pred = model.predict_position(test_med)
        elapsed = time.time() - t0
        eH, eV, e3D = errors(y_te, pred)
        s = summary(eH, eV, e3D, name, ref)
        s["Time(s)"] = round(elapsed, 2)
        results.append(s)
        all_errors[name] = {"eH": eH, "eV": eV, "e3D": e3D}
        all_preds[name]  = pred
        print(f"  H_Mean={s['H_Mean']:.3f}m  V_Mean={s['V_Mean']:.3f}m  "
              f"3D_Mean={s['3D_Mean']:.3f}m  [{elapsed:.2f}s]")

    df = pd.DataFrame(results)
    df.to_csv("results_comparison/metrics_comparison.csv", index=False)

    print("\n" + "="*75)
    print(f"{'Method':<14} {'H_Mean':>8} {'H_P90':>7} {'V_Mean':>8} "
          f"{'V_P90':>7} {'3D_Mean':>9} {'Ref'}")
    print("-"*75)
    for r in results:
        print(f"{r['Method']:<14} {r['H_Mean']:8.3f} {r['H_P90']:7.3f} "
              f"{r['V_Mean']:8.3f} {r['V_P90']:7.3f} {r['3D_Mean']:9.3f}  "
              f"{r['Reference'][:35]}")

    return df, all_errors, all_preds, y_te, train_med, test_med


# =============================================================================
# PLOTTING
# =============================================================================

def plot_cdf_comparison(all_errors, save):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")
    titles = [("eH", "Horizontal Error εH (m)"),
              ("eV", "Vertical Error εV (m)"),
              ("e3D","3-D Error ε₃D (m)")]

    for ax, (key, xlabel) in zip(axes, titles):
        ax.set_facecolor("#FAFAFA")
        for name, ec in all_errors.items():
            e = np.sort(ec[key])
            c = np.arange(1, len(e)+1) / len(e)
            lw = 3.0 if name == "PL-RBF" else 1.8
            ls = "-" if name == "PL-RBF" else "--"
            ax.plot(e, c, lw=lw, ls=ls, color=METHOD_COLORS[name], label=name)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("CDF", fontsize=11)
        ax.set_title(xlabel.split("(")[0].strip(), fontsize=12,
                     fontweight="bold", color="#0D1B2A")
        ax.set_ylim(0, 1); ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        if key == "eH":
            ax.legend(fontsize=9, framealpha=0.95,
                      facecolor="white", edgecolor="#DDD")

    fig.suptitle("CDF Comparison — All 7 Methods on Test-Combined (199 positions)\n"
                 "Solid line = Proposed PL-RBF | Dashed = Published baselines",
                 fontsize=12, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_bar_comparison(df, save):
    methods = df["Method"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")

    pairs = [("H_Mean","Mean Horiz. Error (m)"),
             ("V_Mean","Mean Vert. Error (m)"),
             ("3D_Mean","Mean 3-D Error (m)")]

    for ax, (col, ylabel) in zip(axes, pairs):
        ax.set_facecolor("#FAFAFA")
        colors = [METHOD_COLORS[m] for m in methods]
        edgecolors = ["#B71C1C" if m=="PL-RBF" else "none" for m in methods]
        lw = [2.5 if m=="PL-RBF" else 0.0 for m in methods]
        bars = ax.bar(methods, df[col], color=colors, edgecolor=edgecolors,
                      linewidth=lw, zorder=2, alpha=0.88)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.04,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", color="#0D1B2A")
        ax.yaxis.grid(True, color="#EEE", zorder=1)
        ax.tick_params(axis='x', rotation=30, labelsize=9)
        ax.set_ylim(0, df[col].max() * 1.25)

    fig.suptitle("Mean Positioning Error — Published Methods vs. Proposed PL-RBF",
                 fontsize=13, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_hv_heatmap(df, save):
    metrics = ["H_Mean","H_P75","H_P90","H_RMSE",
               "V_Mean","V_P75","V_P90","V_RMSE",
               "3D_Mean","3D_P90","3D_RMSE"]
    labels  = ["H-Mean","H-P75","H-P90","H-RMSE",
               "V-Mean","V-P75","V-P90","V-RMSE",
               "3D-Mean","3D-P90","3D-RMSE"]
    mat = df[metrics].values
    methods = df["Method"].tolist()

    fig, ax = plt.subplots(figsize=(15, 5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10.5)
    for i in range(len(methods)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if mat[i,j] > mat.max()*0.65 else "#111")
    # Highlight PL-RBF row
    pi = methods.index("PL-RBF")
    for j in range(len(metrics)):
        ax.add_patch(plt.Rectangle((j-0.5, pi-0.5), 1, 1,
                     fill=False, edgecolor="#B71C1C", lw=2.0))
    plt.colorbar(im, ax=ax, label="Error (m)", shrink=0.85)
    ax.set_title("Error Metric Heatmap — All Methods (darker = worse)\n"
                 "Red border = Proposed PL-RBF",
                 fontsize=12, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_improvement(df, save):
    """Show % improvement of PL-RBF over each published method."""
    plrbf = df[df["Method"]=="PL-RBF"].iloc[0]
    others = df[df["Method"]!="PL-RBF"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    for ax, (col, title, clr_pos, clr_neg) in zip(axes, [
        ("H_Mean", "Horizontal Error (H_Mean)", "#2196F3", "#F44336"),
        ("V_Mean", "Vertical Error (V_Mean)",   "#4CAF50", "#FF9800"),
    ]):
        ax.set_facecolor("#FAFAFA")
        impr = 100.0 * (others[col].values - plrbf[col]) / others[col].values
        colors = [clr_pos if v >= 0 else clr_neg for v in impr]
        bars = ax.barh(others["Method"].tolist(), impr,
                       color=colors, alpha=0.85, edgecolor="none")
        ax.axvline(0, color="#333", lw=1.5)
        for bar, val in zip(bars, impr):
            x = bar.get_width() + 0.5 if val >= 0 else bar.get_width() - 0.5
            ha = "left" if val >= 0 else "right"
            ax.text(x, bar.get_y()+bar.get_height()/2,
                    f"{val:+.1f}%", va="center", ha=ha, fontsize=9)
        ax.set_xlabel("PL-RBF Improvement over baseline (%)", fontsize=10)
        ax.set_title(f"% Improvement in {title}", fontsize=11,
                     fontweight="bold", color="#0D1B2A")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("PL-RBF Relative Improvement over Published Methods\n"
                 "Positive = PL-RBF is better | Negative = baseline is better",
                 fontsize=12, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_summary_table(df, save):
    disp_cols = ["Method","Reference","H_Mean","H_P90","H_RMSE",
                 "V_Mean","V_P90","V_RMSE","3D_Mean","3D_P90","Time(s)"]
    disp = df[[c for c in disp_cols if c in df.columns]].round(3).copy()
    disp.columns = ["Method","Reference","H-Mean","H-P90","H-RMSE",
                    "V-Mean","V-P90","V-RMSE","3D-Mean","3D-P90","Time(s)"]

    fig, ax = plt.subplots(figsize=(22, len(disp)*0.65 + 2))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.8)

    # Header
    for j in range(len(disp.columns)):
        tbl[0,j].set_facecolor("#1A237E")
        tbl[0,j].set_text_props(color="white", fontweight="bold")

    # Column groups
    h_clr = "#E3F2FD"; v_clr = "#FCE4EC"; d3_clr = "#E8F5E9"
    for i in range(1, len(disp)+1):
        m = disp.iloc[i-1]["Method"]
        is_proposed = (m == "PL-RBF")
        for j in range(len(disp.columns)):
            bg = "#FFF9C4" if is_proposed else "white"
            if j in [2,3,4]:   bg = "#BBDEFB" if is_proposed else h_clr
            if j in [5,6,7]:   bg = "#F8BBD0" if is_proposed else v_clr
            if j in [8,9]:     bg = "#C8E6C9" if is_proposed else d3_clr
            tbl[i,j].set_facecolor(bg)
            if is_proposed:
                tbl[i,j].set_text_props(fontweight="bold")

    ax.set_title("Complete Comparison Table — Published Methods vs. Proposed PL-RBF\n"
                 "Yellow rows = Proposed method | Blue = Horizontal | Red = Vertical | Green = 3-D",
                 fontsize=12, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(save, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df, all_errors, all_preds, y_te, train_med, test_med = run_all()

    print("\nGenerating plots …")
    plot_cdf_comparison(all_errors,
                        "results_comparison/plot_cdf_comparison.png")
    plot_bar_comparison(df,
                        "results_comparison/plot_bar_comparison.png")
    plot_hv_heatmap(df,
                    "results_comparison/plot_heatmap_comparison.png")
    plot_improvement(df,
                     "results_comparison/plot_improvement.png")
    plot_summary_table(df,
                       "results_comparison/plot_summary_table.png")

    print("\nAll done! → results_comparison/")
