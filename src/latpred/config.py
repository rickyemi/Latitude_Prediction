"""Central configuration: every tunable constant lives here so that experiments are reproducible."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SEED = 0

# ---------------------------------------------------------------------------------------------
# Raw data schema. Short codes follow columns_description.csv (abbreviation column).
# ---------------------------------------------------------------------------------------------
RAW_COLUMNS = ["idx", "date", "hour", "prcp", "stp", "smax", "smin", "gbrd", "temp", "dewp",
               "tmax", "tmin", "dmax", "dmin", "hmax", "hmin", "hmdy", "wdct", "gust", "wdsp",
               "region", "state", "station", "station_code", "lat", "lon", "elev"]
NUMERIC = ["prcp", "stp", "smax", "smin", "gbrd", "temp", "dewp", "tmax", "tmin", "dmax",
           "dmin", "hmax", "hmin", "hmdy", "wdct", "gust", "wdsp"]

# Plausible physical ranges. Anything outside is a sensor fault or placeholder (e.g. tmax == 0,
# dewpoint == -10, radiation above the solar constant). The pressure floor allows high stations.
VALID_RANGE = {
    "prcp": (0, 150), "stp": (700, 1100), "smax": (700, 1100), "smin": (700, 1100),
    "gbrd": (0, 5000),  # kJ/m2 per hour; solar constant is about 4900 kJ/m2/h
    "temp": (-10, 45), "dewp": (-9.9, 35), "tmax": (-10, 46), "tmin": (-10, 45),
    "dmax": (-9.9, 35), "dmin": (-9.9, 35), "hmax": (8, 100), "hmin": (8, 100),
    "hmdy": (8, 100), "wdct": (0, 360), "gust": (0, 50), "wdsp": (0, 30),
}

HOURS = [0, 4, 8, 12, 16, 20]          # observation hours (UTC)
DAYLIGHT_HOURS = [12, 16, 20]          # hours with non-zero radiation in Brazil

# ---------------------------------------------------------------------------------------------
# Thresholds (each is justified and, where possible, calibrated in the notebook)
# ---------------------------------------------------------------------------------------------
STATION_CLUSTER_TOL_DEG = 0.1          # files within 0.1 deg (~11 km) are the same physical station
MIN_OBS_PER_MONTH = 60                 # ~10 days of 4-hourly readings for a usable month
PHYS_RMSE_MAX = 250.0                  # kJ/m2; worse fits mean a faulty radiation sensor
PHYS_K_RANGE = (0.6, 1.2)              # plausible clear-sky transmissivity scale
MATCH_THRESHOLD_GRID = (0.20, 1.00, 0.01)
NOVELTY_K = 5
REFERENCE_YEARS_CALIBRATION = (2016, 2018)   # PS1 reference period when calibrating matching
QUERY_YEAR_CALIBRATION = 2019                # PS1 query period when calibrating matching
REFERENCE_YEAR_PS2 = 2017                    # PS1 data from this year on is the PS2 reference


@dataclass
class Paths:
    """Filesystem locations. Override with environment variables or CLI flags."""
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", "data/raw/DataScienceChallenge")))
    output_dir: Path = field(default_factory=lambda: Path(os.environ.get("OUTPUT_DIR", "reports")))
    processed_dir: Path = field(default_factory=lambda: Path(os.environ.get("PROCESSED_DIR", "data/processed")))

    @property
    def figures(self) -> Path:
        return self.output_dir / "figures"

    def make_dirs(self) -> None:
        for p in (self.output_dir, self.figures, self.processed_dir):
            p.mkdir(parents=True, exist_ok=True)
