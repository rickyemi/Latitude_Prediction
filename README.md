# Recovering Weather-Station Latitude from Weather Observations

*Yemi Odeyemi · October 2026*

## Problem statement
A client collects observations from weather stations across Brazil to forecast weather. For data after 2020, station latitude was corrupted to -9999. In practice every location field was lost: region, state, station name and code, longitude and elevation. Two datasets were provided:
- **PS1 (training):** 73 files from 2000–2020 with known latitude. These are 43 physical stations, all in Brazil's North region (-3.9° to 4.5°).
- **PS2 (prediction):** 35 sensors from 2020–21 with no location. Some are new and may lie entirely outside the PS1 region.

## Goal
1. Predict the latitude of every PS2 sensor.
2. Identify which PS2 sensors are not part of PS1.
3. Deliver a reproducible pipeline (Docker) that writes `prediction_results.csv`.

## Methods
**Preprocessing.**
- Converted the -9999 sentinel and physically impossible values (e.g. radiation above the solar constant) to missing.
- Removed 0 °C max/min placeholders that contradicted the instantaneous reading.
- Clustered the 73 PS1 files into 43 physical stations by location (within 0.1°). Each station is split across 2–3 files and station codes are inconsistent, so this prevents leakage during validation.
- Used the station-month as the modelling unit and dropped months with fewer than 60 valid readings. Raw values were not imputed.

**Feature engineering.** 22 features per station-month: temperature, dew point and humidity levels; daily-cycle amplitude and timing; pressure tides; rain rate and frequency; wind speed and mean wind direction; the shape of the daily sunlight curve; and season. Absolute pressure was left out of the model because it mostly reflects elevation.

**Model development.**
- **Learned model:** compared a mean baseline, Ridge, Random Forest and gradient boosting. Gradient boosting was selected. A station's latitude is the median of its monthly predictions.
- **Physics model:** inverted clear-sky solar geometry to find the latitude and longitude that best explain each sensor's sunlight readings by month and hour (Spencer 1971; Haurwitz 1945). It needs no latitude labels, so it can estimate locations outside the training range.
- **Is the sensor in PS1?** Three checks:
  - a monthly weather "fingerprint" matched one-to-one against PS1 stations;
  - a score for how unlike PS1's climate the sensor is;
  - a check that the physics latitude does not contradict the match.
- **Decision rule:** outside the region → physics estimate; matched to a PS1 station → that station's known latitude; otherwise → learned model.

**Evaluation.**
- Leave-one-station-out cross-validation: each test station is never seen in training.
- An extrapolation test: train on the southern stations, predict the northernmost.
- Physics accuracy on PS1.
- Calibration of the matching threshold on PS1 (2016–18 data vs 2019).
- A simulated end-to-end test of the decision rule.

## Results and outcome
| Evaluation | Result |
|---|---|
| Learned model on unseen stations | **MAE 1.14°** vs 1.56° for the mean baseline (**27% better**); RF 1.25°, Ridge 1.40° |
| Extrapolation test | Stations at 2.8°–4.5° predicted at -0.9° to -2.7°: the trees cannot predict outside the training range |
| Physics model on PS1 (no labels) | Latitude MAE 1.64°, longitude MAE 1.02° |
| Station re-identification | 85% correct; threshold balanced accuracy 76% |
| Simulated end-to-end | **MAE 0.31°** for continuing stations, 1.09° for new stations |

**Outcome on PS2 (35 sensors):**
- 22 are continuing PS1 stations.
- 8 are new stations inside the North region.
- **3 are new stations far outside it, at about -16°, -28° and -31°** (central-west and southern Brazil). One averages ~16 °C in July, a climate absent from PS1.
- 2 have no usable data.

**Takeaways.**
- Matched-station predictions are reliable. Out-of-region estimates are approximate (±3–5°) because PS1 has no southern stations.
- The largest next lift is adding training stations from all of Brazil (the national weather agency INMET's station list, or ERA5 reanalysis). That turns extrapolation into interpolation.
- The data pipeline should also be fixed at source, since every location field was lost, not only latitude.


```
data/
├── raw/
│   └── DataScienceChallenge/
│       ├── ps1/                    # 73 CSVs, 2000-2020, known latitude (training)
│       ├── ps2/                    # 35 CSVs, 2020-2021, latitude = -9999 (prediction)
│       └── columns_description.csv
└── processed/                      # written by the pipeline (station-month feature tables)
```

Or point the code elsewhere: `./run.sh /path/to/DataScienceChallenge`, `make run DATA_DIR=...`,
or `DATA_DIR=... python -m latpred.pipeline`.

## Schema notes
* One file per station and period; readings every 4 hours at 00, 04, 08, 12, 16, 20 UTC.
* Missing values are encoded as `-9999`. In PS2, *all* location fields (region, state, station,
  station code, latitude, longitude, height) are `-9999`, not only latitude.
* Column meanings and short codes are in `columns_description.csv`; the code maps the Portuguese
  headers to those short codes (`src/latpred/config.py`, `RAW_COLUMNS`).
