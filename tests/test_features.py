import numpy as np

from latpred.features import MODEL_FEATURES, harmonic, station_month_features


def test_harmonic_recovers_amplitude():
    t = np.arange(6)
    prof = 26 + 3 * np.cos(2 * np.pi * t / 6)
    amp, phase = harmonic(prof, 1)
    assert abs(amp - 3) < 1e-9 and abs(phase) < 1e-9
    assert harmonic(prof, 2)[0] < 1e-9


def test_harmonic_nan_profile():
    assert np.isnan(harmonic([1, 2, np.nan, 4, 5, 6], 1)[0])


def test_station_month_features_shape(station_df):
    f = station_month_features(station_df.assign(station_id="S"), "station_id")
    assert len(f) == 12
    assert set(MODEL_FEATURES) <= set(f.columns)
    assert f.t_amp.between(2, 4).all()          # synthetic diurnal amplitude is 3 C


def test_sparse_months_are_dropped(station_df):
    sparse = station_df[station_df.month != 1].copy()
    sparse = sparse._append(station_df[station_df.month == 1].head(20))
    f = station_month_features(sparse.assign(station_id="S"), "station_id")
    assert 1 not in set(f.month)
