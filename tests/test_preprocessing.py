import numpy as np
import pandas as pd

from latpred.preprocessing import assign_station_ids, clean
from tests.conftest import make_station


def test_sentinels_and_placeholders_become_nan(station_df):
    assert station_df.temp.iloc[:10].isna().all()
    assert station_df.tmax.iloc[10:15].isna().all()
    assert (station_df.temp.dropna() > -100).all()


def test_time_columns(station_df):
    assert set(station_df.h.unique()) == {0, 4, 8, 12, 16, 20}
    assert station_df.month.between(1, 12).all()


def test_station_clustering_groups_nearby_files():
    a = clean(make_station(-2.0, -60.0)).assign(file="a.csv")
    b = clean(make_station(-2.03, -60.02, start="2019-01-01")).assign(file="b.csv")   # same site, new file
    c = clean(make_station(1.5, -50.0)).assign(file="c.csv")                         # different site
    out = assign_station_ids(pd.concat([a, b, c]))
    ids = out.groupby("file").station_id.first()
    assert ids["a.csv"] == ids["b.csv"] != ids["c.csv"]


def test_ps2_style_metadata_is_nan():
    d = make_station(-2.0, -60.0)
    d[["lat", "lon", "elev"]] = -9999
    out = clean(d)
    assert out[["lat", "lon", "elev"]].isna().all().all()
    assert np.isfinite(out.temp).all()
