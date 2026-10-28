"""Candidate regressors and leave-one-station-out (LOSO) cross-validation.

LOSO is the honest test for this problem: PS2 may contain stations the model has never seen,
and each PS1 station is split across several files, so a random split would leak.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED
from .features import MODEL_FEATURES

BEST_MODEL = "Gradient Boosting (HGB)"


def build_models(seed: int = SEED) -> dict:
    """Model zoo, from a naive baseline to gradient boosting (None = predict the training mean)."""
    return {
        "Baseline (mean latitude)": None,
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5,
                                                             n_jobs=-1, random_state=seed)),
        # HGB handles NaN natively, so sensors without e.g. a radiation sensor need no imputation.
        BEST_MODEL: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                  min_samples_leaf=30, l2_regularization=1.0, random_state=seed),
    }


def loso_predict(model, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> np.ndarray:
    """Out-of-fold month-level predictions, holding out one whole station per fold."""
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if model is None:
            pred[te] = y.iloc[tr].mean()
        else:
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            pred[te] = m.predict(X.iloc[te])
    return pred


def to_station_level(pred: np.ndarray, y: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Aggregate month-level predictions to one prediction per station (median)."""
    return (pd.DataFrame({"station_id": groups.values, "lat": y.values, "pred": pred})
            .groupby("station_id").agg(lat=("lat", "first"), pred=("pred", "median"), n_months=("pred", "size")))


def compare_models(f1: pd.DataFrame, models: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Run LOSO for every model. Returns (metrics table, {model: station-level OOF frame})."""
    models = models or build_models()
    X, y, groups = f1[MODEL_FEATURES], f1["lat"], f1["station_id"]
    rows, oof = [], {}
    for name, m in models.items():
        t0 = time.time()
        p = loso_predict(m, X, y, groups)
        s = to_station_level(p, y, groups)
        oof[name] = s
        err = s.pred - s.lat
        rows.append({"model": name, "station MAE (deg)": err.abs().mean(),
                     "station RMSE (deg)": np.sqrt((err ** 2).mean()), "station median AE (deg)": err.abs().median(),
                     "month-level MAE (deg)": np.abs(p - y).mean(), "fit time (s)": time.time() - t0})
    cv = pd.DataFrame(rows).set_index("model")
    base = cv.loc["Baseline (mean latitude)", "station MAE (deg)"]
    cv["lift vs baseline"] = 1 - cv["station MAE (deg)"] / base
    return cv, oof


def fit_final(f1: pd.DataFrame, name: str = BEST_MODEL):
    """Fit the chosen model on every PS1 station-month."""
    return clone(build_models()[name]).fit(f1[MODEL_FEATURES], f1["lat"])


def predict_stations(model, f: pd.DataFrame, key: str) -> pd.DataFrame:
    """Station-level prediction (median) and inter-quartile range across months."""
    p = f.assign(ml_pred=model.predict(f[MODEL_FEATURES]))
    return p.groupby(key).agg(ml_lat=("ml_pred", "median"),
                              ml_q25=("ml_pred", lambda v: v.quantile(0.25)),
                              ml_q75=("ml_pred", lambda v: v.quantile(0.75)),
                              n_months=("ml_pred", "size"))


def feature_importance(model, f1: pd.DataFrame, seed: int = SEED, n_repeats: int = 5) -> pd.Series:
    """Permutation importance (increase in MAE, degrees) of each feature."""
    pi = permutation_importance(model, f1[MODEL_FEATURES], f1["lat"], n_repeats=n_repeats,
                                random_state=seed, scoring="neg_mean_absolute_error")
    return pd.Series(pi.importances_mean, index=MODEL_FEATURES).sort_values()
