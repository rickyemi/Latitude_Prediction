"""Evaluation: metrics, extrapolation test and simulated end-to-end validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from .features import MODEL_FEATURES
from .matching import MatchCalibration
from .modeling import BEST_MODEL, build_models


def regression_metrics(y_true, y_pred) -> dict:
    e = np.asarray(y_pred) - np.asarray(y_true)
    return {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "median_ae": float(np.median(np.abs(e))), "bias": float(e.mean()), "n": len(e)}


def extrapolation_test(f1: pd.DataFrame, station_lat: pd.Series, cut: float = 2.0) -> pd.DataFrame:
    """Train on stations with latitude <= ``cut`` and predict the stations above it.

    Mimics PS2 sensors that lie outside the training range. Tree ensembles cannot predict
    beyond the targets they saw, so the held-out stations are pulled back inside the range.
    """
    north = station_lat[station_lat > cut].index
    tr, te = ~f1.station_id.isin(north), f1.station_id.isin(north)
    m = clone(build_models()[BEST_MODEL]).fit(f1.loc[tr, MODEL_FEATURES], f1.loc[tr, "lat"])
    out = (f1[te].assign(pred=m.predict(f1.loc[te, MODEL_FEATURES]))
           .groupby("station_id").agg(true_lat=("lat", "first"), ml_pred=("pred", "median")))
    out["max_train_lat"] = station_lat[~station_lat.index.isin(north)].max()
    return out


def simulate_decision_rule(cal: MatchCalibration, ml_oof: pd.Series, station_lat: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the decision rule on PS1 queries (each station's 2019+ data).

    Scenario "known station": its older data is in the reference pool (continuing sensor).
    Scenario "new station": its older data is removed, so any match is a false match.
    ``ml_oof`` must be leave-one-station-out predictions so the ML never saw the station.
    """
    rows = []
    for q in cal.d_true.index:
        for scen, pool in [("known station", list(cal.D.columns)),
                           ("new station", [c for c in cal.D.columns if c != q])]:
            d = cal.D.loc[q, pool].dropna()
            if len(d) and d.min() < cal.threshold:
                lat_hat, src = station_lat[d.idxmin()], "matched"
            else:
                lat_hat, src = ml_oof[q], "ml"
            rows.append({"scenario": scen, "station": q, "err": abs(lat_hat - station_lat[q]), "source": src})
    sim = pd.DataFrame(rows)
    summary = sim.groupby("scenario").agg(MAE=("err", "mean"), median_AE=("err", "median"),
                                          share_matched=("source", lambda s: (s == "matched").mean()))
    summary["ML-only MAE"] = (ml_oof.reindex(cal.d_true.index) - station_lat.reindex(cal.d_true.index)).abs().mean()
    return sim, summary
