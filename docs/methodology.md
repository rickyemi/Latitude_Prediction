# Methodology

## 1. Problem framing
* **Target:** station latitude (a constant per station), not a per-reading label.
* **Unit of analysis:** station-month. A station's prediction is the median of its monthly predictions,
  which handles PS2 sensors with anything from 1 to 16 usable months.
* **Two questions:** (1) what is the latitude, (2) is the sensor one of the PS1 stations.

## 2. Preprocessing (`preprocessing.py`)
| Issue | Treatment |
|---|---|
| `-9999` sentinel | set to NaN |
| impossible values (radiation > solar constant, dew point -10, humidity < 8%) | range checks in `config.VALID_RANGE` |
| `tmax/tmin = 0` while the instantaneous reading is > 5 C | placeholder, set to NaN |
| one station split across 2-3 files, inconsistent station codes | cluster files within 0.1 deg into 43 physical stations |
| sparse months | months with < 60 valid 4-hourly readings are dropped |

## 3. Features (`features.py`)
22 station-month features: temperature, dew point and humidity levels; diurnal harmonics of
temperature and humidity (amplitude and phase); pressure tides S1/S2; rain rate and frequency;
wind speed, gust and the mean wind vector (u, v); radiation-profile shape (morning and afternoon
radiation relative to midday); month as sin/cos. Absolute pressure is excluded from the regressor
because it mostly encodes elevation (with ~40 stations a tree model would memorise it), but it is the
strongest signal for re-identifying a station.

## 4. Models (`modeling.py`)
Mean baseline, Ridge, Random Forest and HistGradientBoosting, compared by **leave-one-station-out**
cross-validation at the station level.

| Model | Station MAE (deg) | Lift vs baseline |
|---|---|---|
| Baseline (mean latitude) | 1.56 | 0% |
| Ridge | 1.40 | 10% |
| Random Forest | 1.25 | 20% |
| **Gradient Boosting (HGB)** | **1.14** | **27%** |

An extrapolation test (train at or below 2 deg, predict 2.8-4.5 deg) shows the tree model cannot
predict beyond its training range, which motivates the physics estimator.

## 5. Physics estimator (`physics.py`)
Grid search over latitude, longitude and a transmissivity scale *k* to fit the observed 90th-percentile
radiation per (month, UTC hour) with theoretical clear-sky radiation (Spencer 1971 solar position;
Haurwitz 1945 clear-sky model). On PS1: latitude MAE 1.64 deg, longitude MAE 1.02 deg, with no latitude labels.

## 6. Is the sensor in PS1? (`matching.py`)
* **Fingerprint re-identification:** per-calendar-month distance on 10 station-specific features;
  threshold calibrated on PS1 (2019 query vs 2016-2018 reference). Top-1 accuracy 85%; threshold 0.53
  accepts 78% of true matches and rejects 74% of simulated new stations. PS2 sensors are assigned
  one-to-one to PS1 stations (Hungarian algorithm).
* **Novelty:** kNN distance (k=5) of each station-month to PS1 station-months; the threshold is the
  maximum leave-one-station-out score inside PS1.
* **Consistency:** a match is rejected if a quality-checked physics latitude disagrees with it by more
  than the physics 90th-percentile error.

## 7. Decision rule (`decision.py`) and validation (`evaluation.py`)
Out of region → physics; re-identified → known latitude; otherwise → ML. Replayed on PS1 2019 data:
MAE 0.31 deg for continuing stations and 1.09 deg for simulated new stations (ML alone: 1.04 deg).

## 8. Limitations and next steps
See section 9 of the notebook. The largest expected lift comes from adding training stations outside
the North region (INMET national station list, ERA5 reanalysis).

## References and libraries
* Spencer, J. W. (1971). Fourier series representation of the position of the sun. *Search* 2(5), 172.
* Haurwitz, B. (1945). Insolation in relation to cloudiness and cloud density. *Journal of Meteorology* 2, 154-166.
* NumPy, pandas, SciPy (`linear_sum_assignment`), scikit-learn, Matplotlib, Jupyter.
