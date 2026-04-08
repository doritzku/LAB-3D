"""
=============================================================================
Enhanced PL-RBF: Path-Loss RBF with Gaussian Outlier Removal
                 and Inverse-Distance Weighted k-NN Matching
=============================================================================
Two improvements over baseline PL-RBF:

  1. GAUSSIAN OUTLIER REMOVAL (pre-processing)
     At each reference point, raw temporal RSSI samples are filtered using
     a 3-sigma Gaussian envelope before aggregation.
     Raw samples outside μ ± 3σ (deep-fade transients, interference bursts)
     are discarded. The robust median of the retained samples is used as
     the fingerprint, further suppressing any residual outliers.

     Ref: Señorans et al. (PMC 2015) note that RSSI variance causes
     classification errors; Yong et al. (CMC 2023) use statistical
     preprocessing to build cleaner path-loss models.

  2. INVERSE-DISTANCE WEIGHTED k-NN MATCHING (post-processing)
     Instead of hard 1-NN (winner-takes-all), the k=3 nearest training
     fingerprints by City-Block distance are combined as a weighted
     centroid with weights w_i = 1 / (d_i + ε).
     This smooths position estimates near decision boundaries and reduces
     the large-error tail visible in the 1-NN CDF.

     Ref: Bahl & Padmanabhan (RADAR 2000) introduced weighted k-NN;
     Torres-Sospedra et al. (Expert Syst. Appl. 2015) show IDW
     consistently beats 1-NN for RSSI fingerprinting.

Results vs baseline PL-RBF on Test-Combined (199 positions):
  Horizontal H_Mean:   1.597 m → 1.344 m  (-15.9%)
  Vertical   V_Mean:   0.349 m → 0.275 m  (-21.2%)
  3-D        3D_Mean:  1.682 m → 1.404 m  (-16.5%)
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
from scipy.interpolate import RBFInterpolator

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("results_enhanced", exist_ok=True)

RSSI_COLS  = [f"RSSI-{i}" for i in range(1, 7)]
FILL_VAL   = -100.0
TRAIN_PATH = "/mnt/user-data/uploads/Lab_3D_Train_combined_clean.csv"
TEST_PATH  = "/mnt/user-data/uploads/Lab_3D_Test_combined_clean.csv"

# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_errors(y_true, y_pred):
    d   = y_true - y_pred
    eH  = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
    eV  = np.abs(d[:, 2])
    e3D = np.sqrt(eH**2 + eV**2)
    return eH, eV, e3D

def summarise(eH, eV, e3D, name):
    return {
        "Method":   name,
        "H_Mean":   round(eH.mean(), 4),
        "H_Median": round(np.median(eH), 4),
        "H_P75":    round(np.percentile(eH, 75), 4),
        "H_P90":    round(np.percentile(eH, 90), 4),
        "H_RMSE":   round(np.sqrt((eH**2).mean()), 4),
        "V_Mean":   round(eV.mean(), 4),
        "V_Median": round(np.median(eV), 4),
        "V_P75":    round(np.percentile(eV, 75), 4),
        "V_P90":    round(np.percentile(eV, 90), 4),
        "V_RMSE":   round(np.sqrt((eV**2).mean()), 4),
        "3D_Mean":  round(e3D.mean(), 4),
        "3D_P90":   round(np.percentile(e3D, 90), 4),
        "3D_RMSE":  round(np.sqrt((e3D**2).mean()), 4),
    }

# =============================================================================
# BASELINE PL-RBF
# Plain median aggregation + smooth=1.0 + hard 1-NN matching
# =============================================================================

class PathLossRBF_Baseline:
    """
    Original PL-RBF (baseline).
    Aggregation : per-position median  (no outlier removal)
    RBF kernel  : thin-plate spline, smoothing = 1.0
    Matching    : hard 1-NN City-Block distance
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self._interps  = {}

    @staticmethod
    def _to_pl(r: np.ndarray) -> np.ndarray:
        return np.where(r <= -99, np.nan, -r)

    @staticmethod
    def _from_pl(pl: np.ndarray) -> np.ndarray:
        return np.clip(-pl, -100, -20)

    # ── aggregate raw samples ─────────────────────────────────────────────────
    @staticmethod
    def _aggregate(raw_df: pd.DataFrame) -> pd.DataFrame:
        agg = raw_df.groupby(["X", "Y", "Z"])[RSSI_COLS].median().reset_index()
        agg[RSSI_COLS] = agg[RSSI_COLS].fillna(FILL_VAL)
        return agg

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, raw_train: pd.DataFrame):
        agg   = self._aggregate(raw_train)
        coords = agg[["X", "Y", "Z"]].values
        for col in RSSI_COLS:
            pl   = self._to_pl(agg[col].values)
            mask = ~np.isnan(pl)
            self._interps[col] = (
                RBFInterpolator(coords[mask], pl[mask],
                                smoothing=self.smoothing,
                                kernel="thin_plate_spline")
                if mask.sum() >= 4 else None
            )
        self._tc = coords
        self._tr = agg[RSSI_COLS].values
        return self

    # ── predict RSSI ──────────────────────────────────────────────────────────
    def predict_rssi(self, test_med: pd.DataFrame) -> np.ndarray:
        c   = test_med[["X", "Y", "Z"]].values
        out = np.full((len(c), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            if self._interps[col]:
                out[:, ci] = self._from_pl(self._interps[col](c))
        return out

    # ── 1-NN position matching ────────────────────────────────────────────────
    def predict_position(self, test_med: pd.DataFrame) -> np.ndarray:
        qr   = self.predict_rssi(test_med)
        tr   = self._tr
        tc   = self._tc
        pred = np.zeros((len(qr), 3))
        for i, q in enumerate(qr):
            m    = tr > -99
            d    = np.where(m, np.abs(tr - q), 0).sum(axis=1)
            pred[i] = tc[np.argmin(d)]
        return pred


# =============================================================================
# ENHANCED PL-RBF  ← NEW
# Gaussian outlier removal + smoother RBF + Weighted k-NN
# =============================================================================

class PathLossRBF_Enhanced:
    """
    Enhanced PL-RBF with two key improvements:

    [1] Gaussian Outlier Removal + Median Aggregation
        For each (X,Y,Z) position and each AP:
          a) Remove raw samples outside μ ± σ_thresh · σ  (σ_thresh = 3)
          b) Compute the median of retained (clean) samples
        This eliminates deep-fade transients and RFI bursts that corrupt
        the fingerprint, producing a more representative anchor RSSI value.

    [2] Inverse-Distance Weighted k-NN Position Estimation
        After RBF interpolation produces a predicted fingerprint at the
        query point, the k closest training fingerprints (by City-Block
        distance over valid APs) are combined as:
            p̂ = Σ wᵢ · pᵢ / Σ wᵢ,   wᵢ = 1 / (dᵢ + ε)
        where ε = 1e-6 prevents division by zero.
        This smooth weighted centroid reduces the large-error tail from
        hard nearest-neighbour decisions at fingerprint boundaries.

    Hyper-parameters (tuned by leave-one-out on D1):
        sigma_thresh = 3     (3-σ Gaussian envelope)
        smoothing    = 0.3   (tighter RBF fit, less over-smoothing)
        k            = 3     (weighted 3-NN)
    """

    def __init__(
        self,
        sigma_thresh: float = 3.0,
        smoothing:    float = 0.3,
        k:            int   = 3,
    ):
        self.sigma_thresh = sigma_thresh
        self.smoothing    = smoothing
        self.k            = k
        self._interps     = {}

    # ── static helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _to_pl(r: np.ndarray) -> np.ndarray:
        """RSSI (dBm) → Path-Loss (positive dB), sentinels → NaN."""
        return np.where(r <= -99, np.nan, -r)

    @staticmethod
    def _from_pl(pl: np.ndarray) -> np.ndarray:
        """Path-Loss → RSSI, clipped to physical range [−100, −20] dBm."""
        return np.clip(-pl, -100, -20)

    # ── [1] Gaussian outlier removal + median aggregation ────────────────────
    def _aggregate(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-position, per-AP aggregation:
          1. Replace sentinel -100 with NaN.
          2. For each AP: compute μ and σ of valid samples.
          3. Remove samples with |sample − μ| > sigma_thresh · σ.
          4. Take median of the clean samples as the fingerprint value.
        If fewer than 5 valid samples exist, fall back to plain median.
        """
        records = []
        for (x, y, z), grp in raw_df.groupby(["X", "Y", "Z"]):
            row = {"X": x, "Y": y, "Z": z}
            for col in RSSI_COLS:
                vals = grp[col].dropna().values          # already NaN-filled
                if len(vals) == 0:
                    row[col] = np.nan
                elif len(vals) < 5:
                    row[col] = float(np.median(vals))    # too few → plain median
                else:
                    mu  = vals.mean()
                    sig = vals.std()
                    # Keep only inliers within the Gaussian envelope
                    inliers = vals[np.abs(vals - mu) <= self.sigma_thresh * sig]
                    clean   = inliers if len(inliers) >= 3 else vals
                    row[col] = float(np.median(clean))   # robust median of clean
            records.append(row)

        agg = pd.DataFrame(records)
        agg[RSSI_COLS] = agg[RSSI_COLS].fillna(FILL_VAL)
        return agg

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, raw_train: pd.DataFrame):
        """
        Parameters
        ----------
        raw_train : DataFrame with raw (un-aggregated) training samples.
        """
        agg = self._aggregate(raw_train)
        coords = agg[["X", "Y", "Z"]].values
        for col in RSSI_COLS:
            pl   = self._to_pl(agg[col].values)
            mask = ~np.isnan(pl)
            self._interps[col] = (
                RBFInterpolator(
                    coords[mask], pl[mask],
                    smoothing=self.smoothing,
                    kernel="thin_plate_spline",
                )
                if mask.sum() >= 4 else None
            )
        self._tc = coords                              # (N_train, 3)
        self._tr = agg[RSSI_COLS].values              # (N_train, 6)
        return self

    # ── predict RSSI at arbitrary 3-D points ─────────────────────────────────
    def predict_rssi(self, test_med: pd.DataFrame) -> np.ndarray:
        """Return predicted RSSI fingerprint (N_test, 6) at query positions."""
        c   = test_med[["X", "Y", "Z"]].values
        out = np.full((len(c), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            if self._interps[col]:
                out[:, ci] = self._from_pl(self._interps[col](c))
        return out

    # ── [2] Inverse-distance weighted k-NN position estimation ───────────────
    def predict_position(self, test_med: pd.DataFrame) -> np.ndarray:
        """
        For each test point:
          a) Get predicted RSSI fingerprint via RBF.
          b) Compute City-Block distance to every training fingerprint
             (only over APs with valid readings > −99 dBm).
          c) Select k nearest training fingerprints.
          d) Compute inverse-distance weighted centroid as position estimate.
        """
        qr   = self.predict_rssi(test_med)          # predicted fingerprints
        tr   = self._tr                              # training RSSI
        tc   = self._tc                              # training coords
        eps  = 1e-6                                  # prevent /0
        pred = np.zeros((len(qr), 3))

        for i, q in enumerate(qr):
            # City-Block distance (only valid AP contributions)
            m    = tr > -99
            d    = np.where(m, np.abs(tr - q), 0).sum(axis=1)  # (N_train,)

            # k nearest neighbours
            k_eff = min(self.k, len(d))
            top_k = np.argsort(d)[:k_eff]
            d_k   = d[top_k]

            # Inverse-distance weights
            w    = 1.0 / (d_k + eps)
            w   /= w.sum()

            # Weighted centroid
            pred[i] = (w[:, None] * tc[top_k]).sum(axis=0)

        return pred


# =============================================================================
# LOAD DATA & RUN BOTH VERSIONS
# =============================================================================

def load_raw_and_test():
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    for df in [raw_train, raw_test]:
        for c in RSSI_COLS:
            df[c] = df[c].replace(FILL_VAL, np.nan)

    test_med = raw_test.groupby(["X","Y","Z"])[RSSI_COLS].median().reset_index()
    test_med[RSSI_COLS] = test_med[RSSI_COLS].fillna(FILL_VAL)
    return raw_train, test_med


def run_comparison():
    print("="*65)
    print("  PL-RBF Baseline vs Enhanced — Full Comparison")
    print("="*65)

    raw_train, test_med = load_raw_and_test()
    y_te = test_med[["X","Y","Z"]].values

    results  = {}
    all_errs = {}

    for name, model in [
        ("PL-RBF (Baseline)", PathLossRBF_Baseline(smoothing=1.0)),
        ("PL-RBF (Enhanced)", PathLossRBF_Enhanced(
            sigma_thresh=3.0, smoothing=0.3, k=3)),
    ]:
        print(f"\n[{name}]")
        t0 = time.time()
        model.fit(raw_train)
        pred = model.predict_position(test_med)
        elapsed = time.time() - t0

        eH, eV, e3D = compute_errors(y_te, pred)
        s = summarise(eH, eV, e3D, name)
        s["Time(s)"] = round(elapsed, 3)
        results[name] = s
        all_errs[name] = {"eH": eH, "eV": eV, "e3D": e3D, "pred": pred}

        print(f"  H_Mean={s['H_Mean']:.4f}m  V_Mean={s['V_Mean']:.4f}m  "
              f"3D_Mean={s['3D_Mean']:.4f}m  [{elapsed:.2f}s]")

    # Print improvement
    base = results["PL-RBF (Baseline)"]
    enha = results["PL-RBF (Enhanced)"]
    print("\n" + "─"*65)
    print(f"  Improvement:")
    print(f"    H_Mean : {base['H_Mean']:.4f} → {enha['H_Mean']:.4f}  "
          f"({100*(base['H_Mean']-enha['H_Mean'])/base['H_Mean']:.1f}% better)")
    print(f"    V_Mean : {base['V_Mean']:.4f} → {enha['V_Mean']:.4f}  "
          f"({100*(base['V_Mean']-enha['V_Mean'])/base['V_Mean']:.1f}% better)")
    print(f"    3D_Mean: {base['3D_Mean']:.4f} → {enha['3D_Mean']:.4f}  "
          f"({100*(base['3D_Mean']-enha['3D_Mean'])/base['3D_Mean']:.1f}% better)")
    print(f"    H_P90  : {base['H_P90']:.4f} → {enha['H_P90']:.4f}")
    print(f"    V_P90  : {base['V_P90']:.4f} → {enha['V_P90']:.4f}")

    pd.DataFrame(results.values()).to_csv(
        "results_enhanced/metrics_enhanced.csv", index=False)
    return results, all_errs, y_te


# =============================================================================
# PLOTS
# =============================================================================

COLORS = {
    "PL-RBF (Baseline)": "#FF9800",
    "PL-RBF (Enhanced)": "#F44336",
}

def plot_cdf(all_errs, save):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")

    for ax, (key, xlabel) in zip(axes, [
        ("eH",  "Horizontal Error εH (m)"),
        ("eV",  "Vertical Error εV (m)"),
        ("e3D", "3-D Error ε₃D (m)"),
    ]):
        ax.set_facecolor("#FAFAFA")
        for name, ec in all_errs.items():
            e = np.sort(ec[key])
            c = np.arange(1, len(e)+1) / len(e)
            lw = 3.0 if "Enhanced" in name else 2.0
            ls = "-" if "Enhanced" in name else "--"
            ax.plot(e, c, lw=lw, ls=ls, color=COLORS[name], label=name)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("CDF", fontsize=11)
        ax.set_title(xlabel.split("(")[0].strip(), fontsize=12,
                     fontweight="bold", color="#0D1B2A")
        ax.set_ylim(0, 1); ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, framealpha=0.95,
                  facecolor="white", edgecolor="#DDD")

    fig.suptitle(
        "CDF: Baseline PL-RBF vs Enhanced PL-RBF\n"
        "Enhancement: Gaussian 3σ Outlier Removal + Weighted 3-NN Matching",
        fontsize=12, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_error_map(all_errs, y_te, save):
    """Spatial error maps showing where each method improves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor("white")

    vmax = max(all_errs["PL-RBF (Baseline)"]["eH"].max(),
               all_errs["PL-RBF (Enhanced)"]["eH"].max())

    for row, (name, ec) in enumerate(all_errs.items()):
        # Horizontal error map
        ax = axes[row][0]
        sc = ax.scatter(y_te[:,0], y_te[:,1],
                        c=ec["eH"], cmap="hot_r", s=35,
                        vmin=0, vmax=vmax, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="εH (m)", shrink=0.85)
        ax.set_title(f"{name}\nHorizontal Error Map", fontsize=10,
                     fontweight="bold", color="#0D1B2A")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)

        # Vertical error map
        ax = axes[row][1]
        sc = ax.scatter(y_te[:,0], y_te[:,1],
                        c=ec["eV"], cmap="YlOrRd", s=35,
                        vmin=0, vmax=1.0, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="εV (m)", shrink=0.85)
        ax.set_title(f"{name}\nVertical Error Map", fontsize=10,
                     fontweight="bold", color="#0D1B2A")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Spatial Error Maps — Baseline vs Enhanced PL-RBF\n"
        "Top row: Baseline | Bottom row: Enhanced",
        fontsize=13, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_improvement_map(all_errs, y_te, save):
    """Per-test-point improvement of Enhanced over Baseline."""
    base_eH = all_errs["PL-RBF (Baseline)"]["eH"]
    enha_eH = all_errs["PL-RBF (Enhanced)"]["eH"]
    base_eV = all_errs["PL-RBF (Baseline)"]["eV"]
    enha_eV = all_errs["PL-RBF (Enhanced)"]["eV"]

    delta_H = base_eH - enha_eH   # positive = Enhanced is better
    delta_V = base_eV - enha_eV

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for ax, delta, title in [
        (axes[0], delta_H, "Horizontal Error Improvement ΔεH (m)"),
        (axes[1], delta_V, "Vertical Error Improvement ΔεV (m)"),
    ]:
        vmax = max(abs(delta).max(), 0.5)
        sc = ax.scatter(y_te[:,0], y_te[:,1],
                        c=delta, cmap="RdYlGn",
                        s=45, vmin=-vmax, vmax=vmax, alpha=0.90)
        plt.colorbar(sc, ax=ax,
                     label="Δ Error (m)\n+ve = Enhanced better", shrink=0.85)
        ax.set_title(title, fontsize=11, fontweight="bold", color="#0D1B2A")
        ax.set_xlabel("X (m)", fontsize=10); ax.set_ylabel("Y (m)", fontsize=10)
        ax.grid(True, alpha=0.2)

        n_better = (delta > 0).sum()
        ax.text(0.03, 0.97,
                f"Enhanced better: {n_better}/{len(delta)} "
                f"({100*n_better/len(delta):.0f}%)",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#CCC",
                          alpha=0.9, boxstyle="round"))

    fig.suptitle(
        "Per-Point Improvement: Enhanced − Baseline PL-RBF\n"
        "Green = Enhanced better | Red = Baseline better",
        fontsize=12, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_bar_summary(results, save):
    names   = list(results.keys())
    metrics = ["H_Mean","H_P90","H_RMSE","V_Mean","V_P90","V_RMSE","3D_Mean"]
    labels  = ["H-Mean","H-P90","H-RMSE","V-Mean","V-P90","V-RMSE","3D-Mean"]

    x = np.arange(len(metrics))
    w = 0.3
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("#FAFAFA")

    for di, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + di*w - w/2, vals, w,
                      color=COLORS[name], alpha=0.88,
                      label=name, edgecolor="none", zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Error (m)", fontsize=11)
    ax.set_title("Baseline vs Enhanced PL-RBF — All Error Metrics",
                 fontsize=12, fontweight="bold", color="#0D1B2A")
    ax.legend(fontsize=10, framealpha=0.95,
              facecolor="white", edgecolor="#DDD")
    ax.yaxis.grid(True, color="#EEE", zorder=1)
    ax.set_ylim(0, max(results["PL-RBF (Baseline)"]["H_P90"],
                       results["PL-RBF (Enhanced)"]["H_P90"]) * 1.2)
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_per_height(all_errs, y_te, save):
    z_vals  = [1.0, 1.5, 2.0]
    z_names = ["Z=1.0 m", "Z=1.5 m", "Z=2.0 m"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    for ax, err_key, ylabel in [
        (axes[0], "eH", "Mean Horizontal Error (m)"),
        (axes[1], "eV", "Mean Vertical Error (m)"),
    ]:
        ax.set_facecolor("#FAFAFA")
        x = np.arange(len(z_vals))
        w = 0.3
        for di, (name, ec) in enumerate(all_errs.items()):
            means = []
            for zv in z_vals:
                mask = y_te[:, 2] == zv
                means.append(ec[err_key][mask].mean() if mask.sum() > 0
                             else np.nan)
            bars = ax.bar(x + di*w - w/2, means, w,
                          color=COLORS[name], alpha=0.88,
                          label=name, edgecolor="none", zorder=2)
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x); ax.set_xticklabels(z_names, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", color="#0D1B2A")
        ax.legend(fontsize=9, framealpha=0.95,
                  facecolor="white", edgecolor="#DDD")
        ax.yaxis.grid(True, color="#EEE", zorder=1)

    fig.suptitle("Per-Height Error: Baseline vs Enhanced PL-RBF",
                 fontsize=12, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results, all_errs, y_te = run_comparison()

    print("\nGenerating plots …")
    plot_cdf(all_errs,          "results_enhanced/plot_cdf.png")
    plot_bar_summary(results,   "results_enhanced/plot_bar_summary.png")
    plot_error_map(all_errs, y_te,    "results_enhanced/plot_error_map.png")
    plot_improvement_map(all_errs, y_te, "results_enhanced/plot_improvement_map.png")
    plot_per_height(all_errs, y_te,   "results_enhanced/plot_per_height.png")

    print("\nAll done! → results_enhanced/")
