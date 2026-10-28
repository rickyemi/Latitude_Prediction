import numpy as np
import pytest

from latpred.physics import clear_sky_hourly, solar_declination_eot, solar_invert_station
from latpred.preprocessing import clean
from tests.conftest import make_station


def test_declination_extremes():
    dec_jun, _ = solar_declination_eot(172)
    dec_dec, _ = solar_declination_eot(355)
    assert np.degrees(dec_jun) == pytest.approx(23.44, abs=0.3)
    assert np.degrees(dec_dec) == pytest.approx(-23.44, abs=0.3)


def test_no_radiation_at_night():
    assert clear_sky_hourly(-2.0, -60.0, 100, 4) == pytest.approx(0, abs=1e-6)   # 04 UTC = midnight local
    assert clear_sky_hourly(-2.0, -60.0, 100, 17) > 600                        # 17 UTC = 1 pm local


@pytest.mark.parametrize("lat,lon", [(-2.0, -60.0), (-25.0, -48.0)])
def test_inversion_recovers_location(lat, lon):
    df = clean(make_station(lat, lon, seed=1))
    est = solar_invert_station(df)
    assert est["phys_lat"] == pytest.approx(lat, abs=2.5)
    assert est["phys_lon"] == pytest.approx(lon, abs=2.0)


def test_inversion_needs_radiation():
    df = clean(make_station(-2.0, -60.0))
    df["gbrd"] = np.nan
    assert np.isnan(solar_invert_station(df)["phys_lat"])
