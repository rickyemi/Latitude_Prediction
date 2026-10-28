"""Station-month feature engineering.

The modelling unit is one (station, year, month). PS2 sensors have between 1 and 16 usable
months, so monthly rows handle partial records naturally; a station-level prediction is the
median of its monthly predictions.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import DAYLIGHT_HOURS, HOURS, MIN_OBS_PER_MONTH

# Features given to the latitude regressor. Absolute pressure (p_mean) is excluded on purpose:
# it mostly encodes elevation, and with ~40 training stations a tree model would memorise it.
MODEL_FEATURES = ["t_mean", "t_std", "t_amp", "t_phase_sin", "t_phase_cos", "td_mean", "rh_mean",
                  "rh_amp", "p_std", "p_s1", "p_s2", "rain_mean", "rain_frac", "ws_mean",
                  "gust_mean", "wind_u", "wind_v", "rad16_q90", "rad12_rel", "rad20_rel",
                  "m_sin", "m_cos"]

# Features used to recognise a sensor across years ("fingerprint"). Absolute pressure is
# included here because it is the most station-specific signal (it encodes elevation), and the
# diurnal temperature phase encodes longitude (local solar time).
FINGERPRINT_FEATURES = ["p_mean", "t_mean", "td_mean", "rh_mean", "t_amp", "ws_mean",
                        "rad12_rel", "rad20_rel", "t_phase_sin", "t_phase_cos"]


def harmonic(profile: np.ndarray, k: int) -> tuple[float, float]:
    """Amplitude and phase of the k-th daily harmonic of an evenly spaced diurnal profile.

    With 6 samples per day (every 4 h) harmonics k=1 (24 h) and k=2 (12 h, the semidiurnal
    pressure tide) are resolvable (Nyquist is k=3).
    """
    profile = np.asarray(profile, dtype=float)
    if np.isnan(profile).any():
        return np.nan, np.nan
    n = len(profile)
    ang = 2 * np.pi * k * np.arange(n) / n
    a = 2 / n * np.sum(profile * np.cos(ang))
    b = 2 / n * np.sum(profile * np.sin(ang))
    return float(np.hypot(a, b)), float(np.arctan2(b, a))


def month_summary(g: pd.DataFrame) -> dict:
    """Summarise one station-month of 4-hourly observations into a feature dict."""
    prof = {c: g.groupby("h")[c].mean().reindex(HOURS).to_numpy() for c in ["temp", "hmdy", "stp"]}
    t_amp, t_ph = harmonic(prof["temp"], 1)          # diurnal temperature cycle
    p_s1, _ = harmonic(prof["stp"], 1)               # pressure tides: S2 amplitude ~ cos^3(lat)
    p_s2, _ = harmonic(prof["stp"], 2)
    rad = (g[g.h.isin(DAYLIGHT_HOURS)].groupby("h")["gbrd"].quantile(0.9).reindex(DAYLIGHT_HOURS).to_numpy())
    r12, r16, r20 = rad
    wd = np.radians(g["wdct"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN wind months
        wind_u = np.nanmean(g["wdsp"] * np.sin(wd))
        wind_v = np.nanmean(g["wdsp"] * np.cos(wd))
    ok16 = bool(np.isfinite(r16) and r16 > 0)
    return {
        "n_obs": int(g["temp"].notna().sum()),
        "t_mean": g["temp"].mean(), "t_std": g["temp"].std(),
        "t_amp": t_amp, "t_phase_sin": np.sin(t_ph), "t_phase_cos": np.cos(t_ph),
        "td_mean": g["dewp"].mean(), "rh_mean": g["hmdy"].mean(),
        "rh_amp": harmonic(prof["hmdy"], 1)[0],
        "p_mean": g["stp"].mean(), "p_std": g["stp"].std(), "p_s1": p_s1, "p_s2": p_s2,
        "rain_mean": g["prcp"].mean(), "rain_frac": (g["prcp"] > 0).mean(),
        "ws_mean": g["wdsp"].mean(), "gust_mean": g["gust"].mean(),
        "wind_u": wind_u, "wind_v": wind_v,
        "rad16_q90": r16,
        "rad12_rel": r12 / r16 if ok16 else np.nan,   # morning vs midday: day length and longitude
        "rad20_rel": r20 / r16 if ok16 else np.nan,   # afternoon vs midday
    }


def station_month_features(df: pd.DataFrame, key: str, min_obs: int = MIN_OBS_PER_MONTH) -> pd.DataFrame:
    """Build the station-month feature table keyed by ``key`` (``station_id`` for PS1, ``file`` for PS2).

    Months with fewer than ``min_obs`` valid temperature readings are dropped.
    """
    rows = []
    for (k, y, m), g in df.groupby([key, "year", "month"]):
        if g["temp"].notna().sum() < min_obs:
            continue
        r = month_summary(g)
        r.update({key: k, "year": y, "month": m})
        rows.append(r)
    out = pd.DataFrame(rows)
    out["m_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["m_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out
