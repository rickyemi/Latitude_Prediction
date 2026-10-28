"""Physics-based latitude (and longitude) from the clear-sky radiation profile.

For each sensor, the 90th percentile of hourly radiation per (month, UTC hour) approximates
clear-sky radiation. A grid search finds the latitude, longitude and transmissivity scale k
whose theoretical clear-sky radiation (integrated over the same accumulation hour) fits best.

References
- Spencer, J. W. (1971). Fourier series representation of the position of the sun. Search 2(5).
  (solar declination and equation of time, as used by NOAA)
- Haurwitz, B. (1945). Insolation in relation to cloudiness and cloud density. J. Meteorology 2.
  (clear-sky model GHI = 1098 cos z exp(-0.057 / cos z))

Because it uses no latitude labels, the estimate is not limited to the PS1 training range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DAYLIGHT_HOURS, PHYS_K_RANGE, PHYS_RMSE_MAX

LAT_GRID = np.arange(-35, 10.01, 0.25)
LON_GRID = np.arange(-75, -34.9, 0.5)


def solar_declination_eot(doy):
    """Solar declination (rad) and equation of time (hours) for day-of-year ``doy``."""
    g = 2 * np.pi * (np.asarray(doy, dtype=float) - 1) / 365
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot_h = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                      - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)) / 60
    return dec, eot_h


def clear_sky_hourly(lat, lon, doy, hour_utc, nsub: int = 12):
    """Mean Haurwitz clear-sky GHI (W/m2) over the hour ending at ``hour_utc``. Inputs broadcast."""
    dec, eot_h = solar_declination_eot(doy)
    la = np.radians(lat)
    total = 0.0
    for s in range(nsub):
        t = np.asarray(hour_utc) - 1 + (s + 0.5) / nsub
        ha = np.radians((t + np.asarray(lon) / 15 + eot_h - 12) * 15)
        cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
        cz = np.clip(cz, 1e-3, None)
        total = total + np.where(cz > 1e-3, 1098 * cz * np.exp(-0.057 / cz), 0.0)
    return total / nsub


def radiation_profile(g: pd.DataFrame, q: float = 0.9, min_n: int = 8) -> pd.DataFrame:
    """Upper-quantile radiation per (month, daylight hour): a proxy for clear-sky radiation."""
    x = g[g.h.isin(DAYLIGHT_HOURS) & g.gbrd.notna()]
    p = x.groupby(["month", "h"])["gbrd"].agg(n="size", q=lambda s: s.quantile(q)).reset_index()
    return p[p.n >= min_n]


def invert_profile(p: pd.DataFrame) -> dict:
    """Grid-search lat/lon/k for one radiation profile (columns month, h, q)."""
    out = {"phys_lat": np.nan, "phys_lon": np.nan, "phys_k": np.nan, "phys_rmse": np.nan,
           "phys_months": int(p.month.nunique()) if len(p) else 0}
    if len(p) < 6 or p.month.nunique() < 3:
        return out
    LA, LO = np.meshgrid(LAT_GRID, LON_GRID, indexing="ij")
    doy = (p.month.to_numpy() - 1) * 30.44 + 15
    model = clear_sky_hourly(LA[..., None], LO[..., None], doy, p.h.to_numpy()) * 3.6  # W/m2 -> kJ/m2 per h
    obs = p.q.to_numpy()
    k = (model * obs).sum(-1) / np.maximum((model ** 2).sum(-1), 1e-9)          # least-squares scale
    sse = ((obs - k[..., None] * model) ** 2).mean(-1)
    i = np.unravel_index(np.argmin(sse), sse.shape)
    out.update(phys_k=float(k[i]), phys_rmse=float(np.sqrt(sse[i])))
    if PHYS_K_RANGE[0] <= k[i] <= PHYS_K_RANGE[1]:
        out.update(phys_lat=float(LA[i]), phys_lon=float(LO[i]))
    return out


def solar_invert_station(g: pd.DataFrame) -> dict:
    """Physics estimate for one station's raw observations."""
    return invert_profile(radiation_profile(g))


def physics_table(df: pd.DataFrame, key: str, rmse_max: float = PHYS_RMSE_MAX) -> pd.DataFrame:
    """Physics estimate for every station/sensor, with a quality flag ``phys_ok``."""
    rows = []
    for k, g in df.groupby(key):
        r = solar_invert_station(g)
        r[key] = k
        rows.append(r)
    t = pd.DataFrame(rows).set_index(key)
    t["phys_ok"] = t.phys_lat.notna() & (t.phys_rmse < rmse_max)
    return t


def evaluate_physics(phys: pd.DataFrame, coords: pd.DataFrame) -> dict:
    """Latitude/longitude error of the physics estimate on stations with known coordinates."""
    t = phys.join(coords[["lat", "lon"]])
    ok = t[t.phys_ok]
    e = ok.phys_lat - ok.lat
    return {"n_valid": len(ok), "n_total": len(t),
            "lat_mae": float(e.abs().mean()), "lat_p90": float(e.abs().quantile(0.9)),
            "lat_bias": float(e.mean()), "lon_mae": float((ok.phys_lon - ok.lon).abs().mean())}
