"""Loading, cleaning and station grouping."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from .config import NUMERIC, RAW_COLUMNS, STATION_CLUSTER_TOL_DEG, VALID_RANGE


def load_folder(folder: str | os.PathLike) -> pd.DataFrame:
    """Read every station CSV in ``folder`` into one long frame and clean it.

    Cleaning steps
    1. Rename the Portuguese headers to short codes (see ``config.RAW_COLUMNS``).
    2. Coerce measurements to numeric and set values outside ``config.VALID_RANGE`` to NaN.
       This removes the -9999 sentinel and placeholders such as ``tmax == 0``.
    3. Parse a UTC timestamp and add year, month and hour columns.
    """
    paths = sorted(glob.glob(os.path.join(str(folder), "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no CSV files in {folder}")
    frames = []
    for path in paths:
        d = pd.read_csv(path, header=0, names=RAW_COLUMNS, low_memory=False)
        d["file"] = os.path.basename(path)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).drop(columns="idx")
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply range checks and derive time columns (separated from I/O so it is unit-testable)."""
    df = df.copy()
    df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    for col, (lo, hi) in VALID_RANGE.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    # Exact 0.0 in the hourly max/min columns is a placeholder when it contradicts the
    # instantaneous reading (e.g. tmax = 0 while temp = 27 C).
    for col, ref in [("tmax", "temp"), ("tmin", "temp"), ("dmax", "dewp"), ("dmin", "dewp")]:
        df.loc[(df[col] == 0) & (df[ref] > 5), col] = np.nan
    df["dt"] = pd.to_datetime(df["date"].astype(str) + " " + df["hour"].astype(str))
    df["year"], df["month"], df["h"] = df.dt.dt.year, df.dt.dt.month, df.dt.dt.hour
    for col in ["lat", "lon", "elev"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(-9999, np.nan)
    return df


def assign_station_ids(df: pd.DataFrame, tol_deg: float = STATION_CLUSTER_TOL_DEG) -> pd.DataFrame:
    """Cluster PS1 files into physical stations by location (greedy, largest file first).

    Every PS1 station is split into a 2000-2018 file and a 2019-2020 file with slightly different
    coordinates, and station codes are not reliable (one site appears as both A120 and A126).
    Grouping by location keeps all files of a station in the same cross-validation fold.
    """
    meta = df.groupby("file").agg(lat=("lat", "first"), lon=("lon", "first"), n=("temp", "size"))
    centers: list[tuple[float, float]] = []
    ids: dict[str, int] = {}
    for f, r in meta.sort_values("n", ascending=False).iterrows():
        for k, (la, lo) in enumerate(centers):
            if abs(la - r.lat) < tol_deg and abs(lo - r.lon) < tol_deg:
                ids[f] = k
                break
        else:
            centers.append((r.lat, r.lon))
            ids[f] = len(centers) - 1
    out = df.copy()
    out["station_id"] = out["file"].map(ids).map(lambda k: f"STN{k:02d}")
    return out


def station_coordinates(ps1: pd.DataFrame) -> pd.DataFrame:
    """Median latitude/longitude and most recent station code per PS1 station."""
    g = ps1.sort_values("dt").groupby("station_id")
    return pd.DataFrame({"lat": g.lat.median(), "lon": g.lon.median(), "code": g.station_code.last()})


def audit_table(ps1: pd.DataFrame, ps2: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side summary of the two datasets."""
    def rng(d, c):
        return f"{d[c].min():.2f} to {d[c].max():.2f}" if d[c].notna().any() else "all -9999"
    return pd.DataFrame({
        "files": [ps1.file.nunique(), ps2.file.nunique()],
        "physical stations": [ps1.station_id.nunique() if "station_id" in ps1 else "?", "unknown"],
        "rows (4-hourly)": [len(ps1), len(ps2)],
        "date range": [f"{ps1.dt.min():%Y-%m-%d} to {ps1.dt.max():%Y-%m-%d}",
                       f"{ps2.dt.min():%Y-%m-%d} to {ps2.dt.max():%Y-%m-%d}"],
        "latitude range": [rng(ps1, "lat"), rng(ps2, "lat")],
        "longitude range": [rng(ps1, "lon"), rng(ps2, "lon")],
    }, index=["PS1", "PS2"]).T


def missingness(ps1: pd.DataFrame, ps2: pd.DataFrame) -> pd.DataFrame:
    """Percent missing per variable after cleaning."""
    return pd.DataFrame({"PS1 % missing": ps1[NUMERIC].isna().mean() * 100,
                         "PS2 % missing": ps2[NUMERIC].isna().mean() * 100}).round(1)
