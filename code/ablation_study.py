"""
=============================================================================
Ablation Study — Enhanced PL-RBF
=============================================================================
Start from plain Weighted k-NN → add one component at a time:

Config-0  Weighted k-NN (no RBF, no PL)               — pure baseline
Config-1  + Path-Loss domain       (PL only)
Config-2  + Gaussian 3σ filter     (PL + Gauss)
Config-3  + Smoothing = 0.3        (PL + Gauss + smooth)
Config-4  + Weighted 3-NN match    (PL + Gauss + smooth + W-kNN) ← FULL

Each config isolates the contribution of one enhancement.
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
os.makedirs("results_ablation", exist_ok=True)

RSSI_COLS  = [f"RSSI-{i}" for i in range(1, 7)]
FILL_VAL   = -100.0
TRAIN_PATH = "/mnt/user-data/uploads/Lab_3D_Train_combined_clean.csv"
TEST_PATH  = "/mnt/user-data/uploads/Lab_3D_Test_combined_clean.csv"

# ─── Colour per config ────────────────────────────────────────────────────────
CFG_COLORS = {
    "C0: Weighted k-NN":             "#90A4AE",
    "C1: +Path-Loss":                "#FFB74D",
    "C2: +Gauss Filter":             "#81C784",
    "C3: +Smooth 0.3":               "#64B5F6",
    "C4: +Weighted 3-NN (Full)":     "#EF5350",
}
CFG_ORDER = list(CFG_COLORS.keys())

# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_errors(y_true, y_pred):
    d   = y_true - y_pred
    eH  = np.sqrt(d[:,0]**2 + d[:,1]**2)
    eV  = np.abs(d[:,2])
    e3D = np.sqrt(eH**2 + eV**2)
    return eH, eV, e3D

def summarise(eH, eV, e3D, name):
    return {
        "Config":     name,
        "H_Mean":     round(eH.mean(), 4),
        "H_Median":   round(np.median(eH), 4),
        "H_P75":      round(np.percentile(eH, 75), 4),
        "H_P90":      round(np.percentile(eH, 90), 4),
        "H_RMSE":     round(np.sqrt((eH**2).mean()), 4),
        "V_Mean":     round(eV.mean(), 4),
        "V_Median":   round(np.median(eV), 4),
        "V_P75":      round(np.percentile(eV, 75), 4),
        "V_P90":      round(np.percentile(eV, 90), 4),
        "V_RMSE":     round(np.sqrt((eV**2).mean()), 4),
        "3D_Mean":    round(e3D.mean(), 4),
        "3D_P90":     round(np.percentile(e3D, 90), 4),
        "3D_RMSE":    round(np.sqrt((e3D**2).mean()), 4),
    }

# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

class AblationPLRBF:
    """
    Unified class that can be configured to include/exclude each component.

    use_pl         : convert RSSI→PathLoss before RBF  (C1)
    use_gauss      : Gaussian 3σ outlier removal        (C2)
    smoothing      : RBF smoothing parameter (0.3=tight, 1.0=smooth) (C3)
    use_wknn       : weighted k-NN vs hard 1-NN         (C4)
    k              : k for weighted matching
    use_rbf        : if False → skip RBF, use direct fingerprint matching (C0)
    """

    def __init__(self,
                 use_rbf:    bool  = True,
                 use_pl:     bool  = True,
                 use_gauss:  bool  = True,
                 smoothing:  float = 0.3,
                 use_wknn:   bool  = True,
                 k:          int   = 3):
        self.use_rbf   = use_rbf
        self.use_pl    = use_pl
        self.use_gauss = use_gauss
        self.smoothing = smoothing
        self.use_wknn  = use_wknn
        self.k         = k
        self._interps  = {}

    # ── aggregation ───────────────────────────────────────────────────────────
    def _aggregate(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if self.use_gauss:
            # Gaussian 3σ filter + median
            records = []
            for (x, y, z), grp in raw_df.groupby(["X","Y","Z"]):
                row = {"X": x, "Y": y, "Z": z}
                for col in RSSI_COLS:
                    vals = grp[col].dropna().values
                    if len(vals) == 0:
                        row[col] = np.nan
                    elif len(vals) < 5:
                        row[col] = float(np.median(vals))
                    else:
                        mu, sig = vals.mean(), vals.std()
                        inliers = vals[np.abs(vals - mu) <= 3.0 * sig]
                        clean   = inliers if len(inliers) >= 3 else vals
                        row[col] = float(np.median(clean))
                records.append(row)
            agg = pd.DataFrame(records)
        else:
            # Plain median (no outlier removal)
            agg = raw_df.groupby(["X","Y","Z"])[RSSI_COLS].median().reset_index()

        agg[RSSI_COLS] = agg[RSSI_COLS].fillna(FILL_VAL)
        return agg

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, raw_train: pd.DataFrame):
        agg    = self._aggregate(raw_train)
        coords = agg[["X","Y","Z"]].values
        self._tc = coords
        self._tr = agg[RSSI_COLS].values

        if not self.use_rbf:
            return self           # Config-0: no RBF at all

        for col in RSSI_COLS:
            vals = agg[col].values.copy()
            if self.use_pl:
                # Path-loss domain: PL = −RSSI
                target = np.where(vals <= -99, np.nan, -vals)
            else:
                # Direct RSSI domain
                target = np.where(vals <= -99, np.nan, vals)

            mask = ~np.isnan(target)
            self._interps[col] = (
                RBFInterpolator(
                    coords[mask], target[mask],
                    smoothing = self.smoothing,
                    kernel    = "thin_plate_spline",
                )
                if mask.sum() >= 4 else None
            )
        return self

    # ── predict RSSI ──────────────────────────────────────────────────────────
    def _predict_rssi(self, test_med: pd.DataFrame) -> np.ndarray:
        if not self.use_rbf:
            # Return stored training fingerprints will be handled by matching
            return test_med[RSSI_COLS].values

        c   = test_med[["X","Y","Z"]].values
        out = np.full((len(c), len(RSSI_COLS)), -70.0)
        for ci, col in enumerate(RSSI_COLS):
            if self._interps.get(col):
                raw_pred = self._interps[col](c)
                if self.use_pl:
                    out[:, ci] = np.clip(-raw_pred, -100, -20)
                else:
                    out[:, ci] = np.clip(raw_pred, -100, -20)
        return out

    # ── position matching ────────────────────────────────────────────────────
    def predict_position(self, test_med: pd.DataFrame) -> np.ndarray:
        if self.use_rbf:
            query_rssi = self._predict_rssi(test_med)
        else:
            # Config-0: match test fingerprints directly
            query_rssi = test_med[RSSI_COLS].values

        tr    = self._tr
        tc    = self._tc
        preds = np.zeros((len(query_rssi), 3))

        for i, q in enumerate(query_rssi):
            m = tr > -99
            d = np.where(m, np.abs(tr - q), 0).sum(axis=1)

            if self.use_wknn:
                k_eff = min(self.k, len(d))
                top   = np.argsort(d)[:k_eff]
                w     = 1.0 / (d[top] + 1e-6)
                w    /= w.sum()
                preds[i] = (w[:, None] * tc[top]).sum(axis=0)
            else:
                preds[i] = tc[np.argmin(d)]

        return preds


# =============================================================================
# RUN ABLATION
# =============================================================================

def run_ablation():
    print("=" * 70)
    print("  ABLATION STUDY — Enhanced PL-RBF")
    print("  Each row adds exactly ONE component to the previous")
    print("=" * 70)

    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test  = pd.read_csv(TEST_PATH)
    for df in [raw_train, raw_test]:
        for c in RSSI_COLS:
            df[c] = df[c].replace(FILL_VAL, np.nan)

    test_med = raw_test.groupby(["X","Y","Z"])[RSSI_COLS].median().reset_index()
    test_med[RSSI_COLS] = test_med[RSSI_COLS].fillna(FILL_VAL)
    y_te = test_med[["X","Y","Z"]].values

    # ── define 5 configs ─────────────────────────────────────────────────────
    configs = [
        ("C0: Weighted k-NN",
         "No RBF, no PL — direct fingerprint IDW k-NN (k=5)",
         AblationPLRBF(use_rbf=False, use_pl=False, use_gauss=False,
                       smoothing=1.0, use_wknn=True, k=5)),

        ("C1: +Path-Loss",
         "+RBF in PL domain — smooth=1.0, plain median, hard 1-NN",
         AblationPLRBF(use_rbf=True, use_pl=True,  use_gauss=False,
                       smoothing=1.0, use_wknn=False, k=1)),

        ("C2: +Gauss Filter",
         "+Gaussian 3σ outlier removal before aggregation",
         AblationPLRBF(use_rbf=True, use_pl=True,  use_gauss=True,
                       smoothing=1.0, use_wknn=False, k=1)),

        ("C3: +Smooth 0.3",
         "+Tighter RBF smoothing (0.3 instead of 1.0)",
         AblationPLRBF(use_rbf=True, use_pl=True,  use_gauss=True,
                       smoothing=0.3, use_wknn=False, k=1)),

        ("C4: +Weighted 3-NN (Full)",
         "+Inverse-distance weighted 3-NN position matching [FULL]",
         AblationPLRBF(use_rbf=True, use_pl=True,  use_gauss=True,
                       smoothing=0.3, use_wknn=True, k=3)),
    ]

    results  = {}
    all_errs = {}

    for name, desc, model in configs:
        print(f"\n[{name}]")
        print(f"  {desc}")
        t0 = time.time()
        model.fit(raw_train)
        pred = model.predict_position(test_med)
        elapsed = time.time() - t0
        eH, eV, e3D = compute_errors(y_te, pred)
        s = summarise(eH, eV, e3D, name)
        s["Time(s)"]    = round(elapsed, 3)
        s["Description"] = desc
        results[name]   = s
        all_errs[name]  = {"eH": eH, "eV": eV, "e3D": e3D}
        print(f"  H_Mean={s['H_Mean']:.4f}  V_Mean={s['V_Mean']:.4f}  "
              f"3D_Mean={s['3D_Mean']:.4f}  [{elapsed:.2f}s]")

    # ── print progression table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Config':<30} {'H_Mean':>8} {'ΔH%':>7} "
          f"{'V_Mean':>8} {'ΔV%':>7} {'3D_Mean':>9}")
    print("-" * 70)

    prev_h = prev_v = prev_3d = None
    for name in CFG_ORDER:
        s  = results[name]
        dh = f"{100*(prev_h-s['H_Mean'])/prev_h:+.1f}%" if prev_h else "  —  "
        dv = f"{100*(prev_v-s['V_Mean'])/prev_v:+.1f}%" if prev_v else "  —  "
        print(f"{name:<30} {s['H_Mean']:8.4f} {dh:>7} "
              f"{s['V_Mean']:8.4f} {dv:>7} {s['3D_Mean']:9.4f}")
        prev_h, prev_v, prev_3d = s["H_Mean"], s["V_Mean"], s["3D_Mean"]

    # ── cumulative improvement ────────────────────────────────────────────────
    c0 = results["C0: Weighted k-NN"]
    c4 = results["C4: +Weighted 3-NN (Full)"]
    print(f"\n── Total improvement C0→C4 ──")
    print(f"  H_Mean : {c0['H_Mean']:.4f} → {c4['H_Mean']:.4f}  "
          f"({100*(c0['H_Mean']-c4['H_Mean'])/c0['H_Mean']:.1f}%)")
    print(f"  V_Mean : {c0['V_Mean']:.4f} → {c4['V_Mean']:.4f}  "
          f"({100*(c0['V_Mean']-c4['V_Mean'])/c0['V_Mean']:.1f}%)")
    print(f"  V_P90  : {c0['V_P90']:.4f} → {c4['V_P90']:.4f}  "
          f"({100*(c0['V_P90']-c4['V_P90'])/c0['V_P90']:.1f}%)")

    pd.DataFrame(results.values()).to_csv(
        "results_ablation/metrics_ablation.csv", index=False)
    return results, all_errs, y_te


# =============================================================================
# PLOTS
# =============================================================================

def plot_progression_bars(results, save):
    """Step-by-step improvement waterfall bar chart."""
    metrics = [
        ("H_Mean",  "H-Mean (m)"),
        ("H_P90",   "H-P90 (m)"),
        ("V_Mean",  "V-Mean (m)"),
        ("V_P90",   "V-P90 (m)"),
        ("3D_Mean", "3D-Mean (m)"),
    ]
    names = CFG_ORDER

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5.5))
    fig.patch.set_facecolor("white")

    for ax, (key, title) in zip(axes, metrics):
        ax.set_facecolor("#FAFAFA")
        vals   = [results[n][key] for n in names]
        colors = [CFG_COLORS[n] for n in names]
        bars   = ax.bar(range(len(names)), vals, color=colors,
                        edgecolor="none", alpha=0.88, zorder=2)

        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

        # Delta arrows between bars
        for i in range(1, len(vals)):
            delta = vals[i] - vals[i-1]
            if abs(delta) > 0.001:
                mid_x = i - 0.5
                top_y = max(vals[i-1], vals[i]) + 0.08
                color = "#2E7D32" if delta < 0 else "#C62828"
                ax.annotate(
                    f"{delta:+.3f}",
                    xy=(mid_x, top_y), ha="center", va="bottom",
                    fontsize=7, color=color, fontweight="bold",
                )

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.split(":")[0] for n in names],
                           fontsize=8.5, rotation=15, ha="right")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#0D1B2A")
        ax.set_ylabel("Error (m)", fontsize=9)
        ax.yaxis.grid(True, color="#EEE", zorder=1)
        ax.set_ylim(0, max(vals) * 1.28)

    fig.suptitle(
        "Ablation Study — Step-by-Step Error Reduction\n"
        "Green delta = improvement | Red delta = regression",
        fontsize=13, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_cdf_ablation(all_errs, save):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor("white")

    for ax, (key, xlabel) in zip(axes, [
        ("eH",  "Horizontal Error εH (m)"),
        ("eV",  "Vertical Error εV (m)"),
        ("e3D", "3-D Error ε₃D (m)"),
    ]):
        ax.set_facecolor("#FAFAFA")
        for name in CFG_ORDER:
            ec = all_errs[name]
            e  = np.sort(ec[key])
            c  = np.arange(1, len(e)+1) / len(e)
            lw = 3.0 if "Full" in name else 1.8
            ls = "-" if "Full" in name else "--"
            ax.plot(e, c, lw=lw, ls=ls,
                    color=CFG_COLORS[name],
                    label=name.split(":")[0] + ": " + name.split(":")[1])

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("CDF",   fontsize=11)
        ax.set_title(xlabel.split("(")[0].strip(),
                     fontsize=12, fontweight="bold", color="#0D1B2A")
        ax.set_ylim(0, 1); ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        if key == "eH":
            ax.legend(fontsize=8.5, framealpha=0.95,
                      facecolor="white", edgecolor="#DDD",
                      loc="lower right")

    fig.suptitle("Ablation Study — CDF Comparison Across All Configurations",
                 fontsize=13, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_component_contribution(results, save):
    """Horizontal bar showing contribution of each added component."""
    names   = CFG_ORDER
    metrics = ["H_Mean", "V_Mean", "3D_Mean"]
    ylabels = ["Horizontal\nH-Mean (m)", "Vertical\nV-Mean (m)", "3D\n3D-Mean (m)"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("white")

    for ax, key, ylabel in zip(axes, metrics, ylabels):
        ax.set_facecolor("#FAFAFA")
        vals   = [results[n][key] for n in names]
        x      = np.arange(len(names))
        colors = [CFG_COLORS[n] for n in names]

        # Stacked contribution bar
        ax.barh(x, vals, color=colors, edgecolor="none", alpha=0.85, zorder=2)

        # Shade the GAIN from each step
        for i in range(1, len(vals)):
            gain = vals[i-1] - vals[i]
            if gain > 0:
                ax.barh(i, gain, left=vals[i],
                        color="#A5D6A7", edgecolor="none", alpha=0.5, zorder=3)
                ax.text(vals[i] + gain/2, i,
                        f"−{gain:.3f}", ha="center", va="center",
                        fontsize=7.5, color="#1B5E20", fontweight="bold")

        ax.set_yticks(x)
        ax.set_yticklabels([n.split(":")[1].strip() for n in names],
                           fontsize=8.5)
        ax.set_xlabel(ylabel.replace("\n"," "), fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", color="#0D1B2A")
        ax.xaxis.grid(True, color="#EEE", zorder=1)
        # Annotate final value
        for i, val in enumerate(vals):
            ax.text(val + 0.02, i, f"{val:.3f}",
                    va="center", fontsize=8, color="#333")

    fig.suptitle("Component Contribution — Error Reduction per Enhancement\n"
                 "Green shading = gain from adding that component",
                 fontsize=13, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_per_height_ablation(all_errs, y_te, save):
    z_vals  = [1.0, 1.5, 2.0]
    z_names = ["Z=1.0 m", "Z=1.5 m", "Z=2.0 m"]
    names   = CFG_ORDER

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.patch.set_facecolor("white")

    for ax, (key, ylabel) in zip(axes, [
        ("eH", "Mean Horizontal Error (m)"),
        ("eV", "Mean Vertical Error (m)"),
    ]):
        ax.set_facecolor("#FAFAFA")
        x = np.arange(len(z_vals))
        w = 0.16
        offset = -(len(names)-1)*w/2
        for di, name in enumerate(names):
            means = [all_errs[name][key][y_te[:,2]==zv].mean()
                     if (y_te[:,2]==zv).sum()>0 else np.nan
                     for zv in z_vals]
            bars = ax.bar(x + offset + di*w, means, w,
                          color=CFG_COLORS[name], alpha=0.88,
                          label=name.split(":")[0],
                          edgecolor="none", zorder=2)
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            val+0.01, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x); ax.set_xticklabels(z_names, fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(ylabel, fontsize=11, fontweight="bold", color="#0D1B2A")
        ax.legend(fontsize=8.5, framealpha=0.95,
                  facecolor="white", edgecolor="#DDD", ncol=2)
        ax.yaxis.grid(True, color="#EEE", zorder=1)

    fig.suptitle("Ablation Study — Per-Height Error Across All Configurations",
                 fontsize=13, fontweight="bold", color="#0D1B2A")
    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_radar_ablation(results, save):
    import math
    metrics = [
        ("H_Mean","H-Mean"), ("H_P75","H-P75"), ("H_P90","H-P90"),
        ("V_Mean","V-Mean"), ("V_P75","V-P75"), ("V_P90","V-P90"),
        ("3D_Mean","3D-Mean"), ("3D_P90","3D-P90"),
    ]
    N      = len(metrics)
    angles = [2*math.pi*i/N for i in range(N)] + [0]
    names  = CFG_ORDER

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    ax  = fig.add_subplot(111, polar=True)
    ax.set_facecolor("#F8F9FA")

    # Normalise each metric to [0,1] (higher norm = worse)
    raw_vals = np.array([[results[n][k] for k,_ in metrics] for n in names])
    vmax     = raw_vals.max(axis=0); vmax[vmax == 0] = 1

    for ni, name in enumerate(names):
        norm = raw_vals[ni] / vmax
        norm_c = np.concatenate([norm, [norm[0]]])
        lw = 3.0 if "Full" in name else 1.8
        ax.plot(angles, norm_c, lw=lw, color=CFG_COLORS[name], label=name)
        ax.fill(angles, norm_c, color=CFG_COLORS[name], alpha=0.06)

    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [l for _,l in metrics], fontsize=10)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.35, color="#CCCCCC")
    ax.set_title("Ablation — Normalised Error Profile\n(smaller area = better)",
                 fontsize=12, fontweight="bold", color="#0D1B2A", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15),
              fontsize=8.5, framealpha=0.95,
              facecolor="white", edgecolor="#DDD")

    plt.tight_layout()
    plt.savefig(save, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


def plot_summary_table_ablation(results, save):
    names   = CFG_ORDER
    keys    = ["H_Mean","H_P75","H_P90","H_RMSE",
               "V_Mean","V_P75","V_P90","V_RMSE",
               "3D_Mean","3D_P90","3D_RMSE"]
    headers = ["Config",
               "H-Mean","H-P75","H-P90","H-RMSE",
               "V-Mean","V-P75","V-P90","V-RMSE",
               "3D-Mean","3D-P90","3D-RMSE"]

    rows = []
    for name in names:
        row = [name.replace("C0: ","C0 ").replace("C1: ","C1 ")
                   .replace("C2: ","C2 ").replace("C3: ","C3 ")
                   .replace("C4: ","C4 ")]
        for k in keys:
            row.append(f"{results[name][k]:.4f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(22, 4.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1, 2.0)

    # Header style
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1A237E")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Row colours
    row_clrs = ["#ECEFF1","#FFF8E1","#E8F5E9","#E3F2FD","#FFEBEE"]
    for i in range(1, 6):
        for j in range(len(headers)):
            tbl[i, j].set_facecolor(row_clrs[i-1])

    # Bold best (min) per numeric column
    for j, key in enumerate(keys):
        vals    = [results[n][key] for n in names]
        best_i  = int(np.argmin(vals))
        tbl[best_i+1, j+1].set_text_props(fontweight="bold", color="#B71C1C")

    ax.set_title(
        "Ablation Study — Complete Metrics Table\n"
        "Red bold = best value per metric  |  Each row adds one component",
        fontsize=11, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(save, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(); print(f"  Saved: {save}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    results, all_errs, y_te = run_ablation()

    print("\nGenerating plots …")
    plot_progression_bars(results,
                          "results_ablation/plot_progression_bars.png")
    plot_cdf_ablation(all_errs,
                      "results_ablation/plot_cdf_ablation.png")
    plot_component_contribution(results,
                                "results_ablation/plot_component_contribution.png")
    plot_per_height_ablation(all_errs, y_te,
                             "results_ablation/plot_per_height_ablation.png")
    plot_radar_ablation(results,
                        "results_ablation/plot_radar_ablation.png")
    plot_summary_table_ablation(results,
                                "results_ablation/plot_summary_table_ablation.png")

    print("\nAll done! → results_ablation/")
