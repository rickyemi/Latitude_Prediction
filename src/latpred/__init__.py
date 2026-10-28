"""latpred: recover weather-station latitude from weather observations.

Modules
-------
config          paths, constants and tunable thresholds
preprocessing   loading, cleaning and PS1 station grouping
features        station-month feature engineering
physics         clear-sky solar-geometry inversion (latitude and longitude)
modeling        candidate regressors and leave-one-station-out cross-validation
matching        station re-identification and climate-novelty scoring
evaluation      metrics, extrapolation test and simulated end-to-end validation
decision        decision rule that turns the estimators into final predictions
visualization   all figures (saved as PNG)
pipeline        end-to-end command-line entry point
"""

__version__ = "1.0.0"
