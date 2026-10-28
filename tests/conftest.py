"""Synthetic fixtures so the tests run without the (confidential) challenge data."""
import numpy as np
import pandas as pd
import pytest

from latpred.physics import clear_sky_hourly


def make_station(lat: float, lon: float, start="2018-01-01", days=365, seed=0, sentinel=False) -> pd.DataFrame:
    """Raw-schema frame for one synthetic station with physically consistent radiation."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 6, freq="4h")
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    ghi = clear_sky_hourly(lat, lon, doy, hour) * 3.6 * rng.uniform(0.6, 1.0, len(idx))
    temp = 26 + 3 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 0.5, len(idx))
    d = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"), "hour": idx.strftime("%H:%M"),
        "prcp": rng.exponential(0.2, len(idx)), "stp": 1005 + rng.normal(0, 1, len(idx)),
        "smax": 1006.0, "smin": 1004.0, "gbrd": ghi, "temp": temp, "dewp": temp - 4,
        "tmax": temp + 0.5, "tmin": temp - 0.5, "dmax": temp - 3.5, "dmin": temp - 4.5,
        "hmax": 85.0, "hmin": 75.0, "hmdy": 80 + rng.normal(0, 3, len(idx)),
        "wdct": rng.uniform(0, 360, len(idx)), "gust": 3.0, "wdsp": 1.2,
        "region": "N", "state": "AM", "station": "X", "station_code": "A000",
        "lat": lat, "lon": lon, "elev": 50.0,
    })
    if sentinel:
        d.loc[d.index[:10], "temp"] = -9999
        d.loc[d.index[10:15], "tmax"] = 0
    return d


@pytest.fixture
def station_df():
    from latpred.preprocessing import clean
    return clean(make_station(-2.0, -60.0, sentinel=True)).assign(file="f1.csv")
