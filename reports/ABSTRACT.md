# Abstract: Recovering Weather-Station Latitude (PS1 → PS2)

**Problem.** For 2020–21 data, latitude was corrupted to -9999. In fact every location field in PS2 is corrupted (region, state, station, longitude, height), so only the weather signal remains. PS1 (2000–2020, 73 files) has known latitudes but covers only Brazil's North region (-3.9° to 4.5°), while some PS2 sensors may sit outside that region.

**Approach.** After converting sentinels and physically impossible values to missing, I clustered PS1 files by location into 43 physical stations, since each station spans several files and station codes are inconsistent. The modelling unit is a station-month: climate levels, diurnal-cycle harmonics (temperature, humidity, pressure tides), wind vector and radiation-profile shape, plus the calendar month. This handles PS2 sensors with anything from 1 to 16 usable months; a station's prediction is the median across its months. Three components:
1. **Learned regressor** (scikit-learn HistGradientBoosting), validated by leave-one-station-out CV against mean, Ridge and Random Forest baselines.
2. **Physics estimator** that inverts clear-sky solar geometry (Spencer 1971 declination/equation of time; Haurwitz 1945 clear-sky model) against each sensor's 90th-percentile radiation by month and UTC hour. It recovers latitude *and* longitude with no training range, so it can extrapolate.
3. **"In PS1?" detection**: a monthly fingerprint matched one-to-one (Hungarian assignment) to PS1 stations, with a threshold calibrated on PS1's own 2016–18 vs 2019 data; a kNN climate-novelty score; and a physics-consistency check.

**Findings.** Gradient boosting predicts unseen PS1 stations with MAE 1.14° (27% better than the 1.56° baseline). However, it cannot extrapolate: stations above the training range were pulled back inside it. The physics estimator reaches MAE 1.6° (longitude 1.0°) on PS1 without using latitude labels. Re-identification is 85% top-1 accurate. A simulated end-to-end test of the decision rule gives MAE 0.31° for continuing stations and 1.09° for new in-region stations. On PS2: **22 sensors are continuing PS1 stations, 8 are new stations within the North region, 3 are new and far outside it (≈-16°, -28°, -31°), and 2 have no usable data.**

**Limitations / next steps.** Out-of-region estimates are approximate (±3–5°) because no southern training data exist. The biggest lift would come from adding INMET national station metadata or ERA5 climatology. Other improvements: anomaly cross-correlation with known stations for exact re-identification, a turbidity-aware clear-sky model (pvlib), and conformal uncertainty intervals.

**Open-source tools:** Python, NumPy, pandas, SciPy (`linear_sum_assignment`), scikit-learn, Matplotlib, Jupyter/nbconvert, Docker.
