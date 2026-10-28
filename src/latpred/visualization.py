"""All figures. Every function draws one figure, saves it as PNG in ``fig_dir`` and returns it.

Style: one validated categorical palette used in fixed order, thin marks, recessive grid,
text in neutral ink (never in series colour), direct labels where they help.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical palette (fixed order) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
STATUS_COLORS = {"known_ps1_station": SERIES[0], "new_in_region": SERIES[1],
                 "new_out_of_region": SERIES[2], "insufficient_data": "#9a9893"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.size": 9.5,
    "lines.linewidth": 2,
})


def _save(fig, fig_dir, name):
    """Save to PNG, then close so figures never pile up (a closed figure still renders with display())."""
    if fig_dir is not None:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / name)
    plt.close(fig)
    return fig


# ------------------------------------------------------------------ data understanding
def plot_missingness(miss: pd.DataFrame, fig_dir=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(miss))
    ax.barh(y - 0.2, miss.iloc[:, 0], height=0.38, color=SERIES[0], label=miss.columns[0])
    ax.barh(y + 0.2, miss.iloc[:, 1], height=0.38, color=SERIES[1], label=miss.columns[1])
    ax.set_yticks(y, miss.index)
    ax.invert_yaxis()
    ax.set_xlabel("% missing after cleaning")
    ax.set_title("Missingness by variable")
    ax.legend(loc="lower right")
    fig.text(0.01, -0.02, "Radiation (gbrd) is not reported at night, so ~55% is missing even in PS1.",
             color=INK2, fontsize=8.5)
    return _save(fig, fig_dir, "01_missingness.png")


def plot_ps2_coverage(ps2: pd.DataFrame, fig_dir=None):
    cov = (ps2.assign(ok=ps2.temp.notna()).groupby("file").ok.mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(cov)), cov.values, color=SERIES[0], width=0.8)
    ax.axhline(10, color=INK2, lw=1, ls="--")
    ax.text(-0.5, 12, "10% valid", color=INK2, ha="left", fontsize=8.5)
    ax.set_xlabel("PS2 sensor (sorted by coverage)")
    ax.set_ylabel("% rows with valid temperature")
    ax.set_title("PS2 data coverage per sensor")
    ax.set_xticks([])
    return _save(fig, fig_dir, "02_ps2_coverage.png")


def plot_seasonal_cycles(f1: pd.DataFrame, f2: pd.DataFrame, station_lat: pd.Series, fig_dir=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    norm = plt.Normalize(station_lat.min(), station_lat.max())
    for s, g in f1.groupby("station_id"):
        m = g.groupby("month").t_mean.mean()
        ax[0].plot(m.index, m.values, color=BLUE_RAMP(norm(station_lat[s])), lw=1.2, alpha=0.9)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=BLUE_RAMP)
    fig.colorbar(sm, ax=ax[0], label="station latitude (deg)", fraction=0.05, pad=0.02)
    ax[0].set_title("PS1 monthly mean temperature")
    for s, g in f2.groupby("file"):
        m = g.groupby("month").t_mean.mean()
        cold = m.min() < 20
        ax[1].plot(m.index, m.values, color=SERIES[1] if cold else "#b9b7b0", lw=2 if cold else 1.1, zorder=3 if cold else 1)
    ax[1].set_title("PS2 monthly mean temperature (latitude unknown)")
    ax[1].text(1.0, 16.0, "highlighted: austral-winter cooling,\na southern station", color=INK2, fontsize=8.5)
    for a in ax:
        a.set_xticks(range(1, 13))
        a.set_xlabel("month")
    ax[0].set_ylabel("deg C")
    fig.tight_layout()
    return _save(fig, fig_dir, "03_seasonal_cycles.png")


def plot_feature_vs_latitude(f1: pd.DataFrame, station_lat: pd.Series, features=("t_amp", "rad20_rel", "wind_v", "p_s2"),
                             fig_dir=None):
    st = f1.groupby("station_id")[list(features)].median().join(station_lat.rename("lat"))
    fig, axes = plt.subplots(1, len(features), figsize=(3.2 * len(features), 3.2), sharey=True)
    for a, c in zip(axes, features):
        a.scatter(st[c], st.lat, s=28, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        r = st[[c, "lat"]].corr().iloc[0, 1]
        a.set_title(f"{c}  (r = {r:+.2f})", fontsize=10)
        a.set_xlabel(c)
    axes[0].set_ylabel("latitude (deg)")
    fig.suptitle("Station-level features vs latitude (PS1)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, fig_dir, "04_features_vs_latitude.png")


# ------------------------------------------------------------------ modeling & evaluation
def plot_model_comparison(cv: pd.DataFrame, fig_dir=None):
    cv = cv.sort_values("station MAE (deg)")
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = [SERIES[0] if i == 0 else "#b9b7b0" for i in range(len(cv))]
    ax.barh(cv.index, cv["station MAE (deg)"], color=colors, height=0.6)
    for i, (v, lift) in enumerate(zip(cv["station MAE (deg)"], cv["lift vs baseline"])):
        ax.text(v + 0.02, i, f"{v:.2f} deg  ({lift:+.0%} vs baseline)", va="center", color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, cv["station MAE (deg)"].max() * 1.45)
    ax.set_xlabel("station-level MAE, leave-one-station-out (deg)")
    ax.set_title("Model comparison")
    return _save(fig, fig_dir, "05_model_comparison.png")


def plot_cv_predictions(s: pd.DataFrame, model_name: str, fig_dir=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].scatter(s.lat, s.pred, s=34, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    lim = [s.lat.min() - 0.5, s.lat.max() + 0.5]
    ax[0].plot(lim, lim, color=INK2, lw=1, ls="--")
    ax[0].text(lim[1], lim[1] - 0.3, "perfect", color=INK2, ha="right", fontsize=8.5)
    ax[0].set_xlabel("true latitude (deg)")
    ax[0].set_ylabel("predicted latitude, held-out station (deg)")
    ax[0].set_title(f"{model_name}: leave-one-station-out")
    err = s.pred - s.lat
    ax[1].hist(err, bins=15, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    ax[1].axvline(0, color=INK2, lw=1)
    ax[1].set_xlabel("prediction error (deg)")
    ax[1].set_ylabel("stations")
    ax[1].set_title(f"Error distribution (MAE {err.abs().mean():.2f} deg)")
    fig.tight_layout()
    return _save(fig, fig_dir, "06_cv_predicted_vs_true.png")


def plot_feature_importance(imp: pd.Series, top: int = 12, fig_dir=None):
    imp = imp.tail(top)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.6)
    ax.set_xlabel("increase in MAE when permuted (deg)")
    ax.set_title("Permutation feature importance (gradient boosting)")
    return _save(fig, fig_dir, "07_feature_importance.png")


def plot_extrapolation(ext: pd.DataFrame, fig_dir=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(ext))
    ax.scatter(ext.true_lat, y, s=70, color=SERIES[0], label="true latitude", zorder=3)
    ax.scatter(ext.ml_pred, y, s=70, color=SERIES[1], label="ML prediction", zorder=3)
    for i, (a, b) in enumerate(zip(ext.true_lat, ext.ml_pred)):
        ax.plot([a, b], [i, i], color=GRID, lw=2, zorder=1)
    ax.axvline(ext.max_train_lat.iloc[0], color=INK2, ls="--", lw=1)
    ax.text(ext.max_train_lat.iloc[0] + 0.08, -0.45, "max training latitude", color=INK2, fontsize=8.5)
    ax.set_yticks(y, ext.index)
    ax.set_xlabel("latitude (deg)")
    ax.set_ylim(-0.7, len(ext) - 0.5)
    ax.set_title("Extrapolation test: trees cannot predict beyond the training range")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    return _save(fig, fig_dir, "08_extrapolation_test.png")


def plot_physics_validation(phys: pd.DataFrame, coords: pd.DataFrame, fig_dir=None):
    t = phys.join(coords[["lat", "lon"]])
    ok = t[t.phys_ok]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].scatter(ok.lat, ok.phys_lat, s=34, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax[0].plot([-5, 5], [-5, 5], color=INK2, lw=1, ls="--")
    ax[0].set_xlabel("true latitude (deg)")
    ax[0].set_ylabel("physics latitude (deg)")
    ax[0].set_title(f"Physics latitude (MAE {(ok.phys_lat - ok.lat).abs().mean():.2f} deg, no training)")
    ax[1].scatter(ok.lon, ok.phys_lon, s=34, color=SERIES[2], edgecolor=SURFACE, linewidth=1)
    lo = [ok.lon.min() - 1, ok.lon.max() + 1]
    ax[1].plot(lo, lo, color=INK2, lw=1, ls="--")
    ax[1].set_xlabel("true longitude (deg)")
    ax[1].set_ylabel("physics longitude (deg)")
    ax[1].set_title(f"Physics longitude (MAE {(ok.phys_lon - ok.lon).abs().mean():.2f} deg)")
    fig.tight_layout()
    return _save(fig, fig_dir, "09_physics_validation.png")


def plot_matching_calibration(cal, fig_dir=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.arange(0, 2, 0.05)
    ax[0].hist(cal.d_true, bins=bins, color=SERIES[0], alpha=0.75, label="same station, 1-3 years later")
    ax[0].hist(cal.d_imp, bins=bins, color=SERIES[1], alpha=0.65, label="closest other station (simulated new)")
    ax[0].axvline(cal.threshold, color=INK, ls="--", lw=1)
    ax[0].text(cal.threshold + 0.03, 0.55, f"threshold {cal.threshold:.2f}", fontsize=8.5, transform=ax[0].get_xaxis_transform())
    ax[0].set_xlabel("fingerprint distance")
    ax[0].set_ylabel("PS1 stations")
    ax[0].set_title("Re-identification calibration on PS1")
    ax[0].legend(loc="upper right", fontsize=8.5)
    ax[1].plot(cal.grid, cal.ba_curve, color=SERIES[0])
    ax[1].axvline(cal.threshold, color=INK, ls="--", lw=1)
    ax[1].set_xlabel("distance threshold")
    ax[1].set_ylabel("balanced accuracy")
    ax[1].set_title(f"Threshold choice (max balanced accuracy {cal.balanced_accuracy:.0%})")
    fig.tight_layout()
    return _save(fig, fig_dir, "10_matching_calibration.png")


def plot_novelty(null: pd.Series, ps2_scores: pd.Series, threshold: float, fig_dir=None):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.linspace(0, max(ps2_scores.max(), null.max()) * 1.05, 30)
    ax.hist(null, bins=bins, color=SERIES[0], alpha=0.75, label="PS1 stations (leave-one-out)")
    ax.hist(ps2_scores.dropna(), bins=bins, color=SERIES[1], alpha=0.65, label="PS2 sensors")
    ax.axvline(threshold, color=INK, ls="--", lw=1)
    ax.text(threshold + 0.05, ax.get_ylim()[1] * 0.85, "max PS1 score", fontsize=8.5)
    ax.set_xlabel("kNN climate-novelty score")
    ax.set_ylabel("count")
    ax.set_title("Climate novelty: is the sensor's climate seen in PS1?")
    ax.legend()
    return _save(fig, fig_dir, "11_novelty_scores.png")


def plot_simulation(summary: pd.DataFrame, fig_dir=None):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["MAE"], width=0.34, color=SERIES[0], label="decision rule")
    ax.bar(x + 0.18, summary["ML-only MAE"], width=0.34, color="#b9b7b0", label="ML only")
    for i, (a, b) in enumerate(zip(summary["MAE"], summary["ML-only MAE"])):
        ax.text(i - 0.18, a + 0.03, f"{a:.2f}", ha="center", fontsize=9)
        ax.text(i + 0.18, b + 0.03, f"{b:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, summary.index)
    ax.set_ylabel("MAE (deg)")
    ax.set_ylim(0, max(summary["MAE"].max(), summary["ML-only MAE"].max()) * 1.35)
    ax.set_title("Simulated end-to-end validation on PS1")
    ax.legend(loc="upper left", ncol=2)
    return _save(fig, fig_dir, "12_simulated_validation.png")


# ------------------------------------------------------------------ results
def plot_ps2_map(res: pd.DataFrame, coords: pd.DataFrame, fig_dir=None):
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.axhspan(coords.lat.min(), coords.lat.max(), color=SERIES[0], alpha=0.06)
    ax.text(coords.lon.max() + 6, coords.lat.min() - 1.4, "PS1 latitude band (shaded)", color=INK2, fontsize=8.5, ha="right")
    ax.scatter(coords.lon, coords.lat, marker="^", s=46, facecolors="none", edgecolors=INK2, label="PS1 stations (true)")
    lon_plot = res.phys_lon.fillna(res.match_station.map(coords.lon))
    for status, color in STATUS_COLORS.items():
        g_all = res[res.status == status]
        g = g_all[lon_plot.loc[g_all.index].notna()]
        if len(g):
            ax.scatter(lon_plot.loc[g.index], g.predicted_latitude, s=52, color=color, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3, label=f"PS2 {status.replace('_', ' ')} ({len(g)} of {len(g_all)} placed)")
    ax.set_xlabel("longitude (PS1 true; PS2 physics or matched estimate)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("PS2 predicted positions vs PS1 stations")
    ax.legend(loc="lower left", fontsize=8.5)
    return _save(fig, fig_dir, "13_ps2_predicted_positions.png")


def plot_ps2_status(res: pd.DataFrame, fig_dir=None):
    order = [s for s in STATUS_COLORS if s in set(res.status)]
    counts = res.status.value_counts().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh([s.replace("_", " ") for s in order], counts.values, color=[STATUS_COLORS[s] for s in order], height=0.6)
    for i, v in enumerate(counts.values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("PS2 sensors")
    ax.set_title(f"PS2 sensor classification ({len(res)} sensors)")
    return _save(fig, fig_dir, "14_ps2_status_counts.png")
