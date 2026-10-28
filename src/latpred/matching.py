"""Is a PS2 sensor part of PS1?

1. Re-identification: compare per-calendar-month fingerprints of a PS2 sensor with each PS1
   station. The acceptance threshold is calibrated on PS1 itself (2016-2018 reference vs
   2019 query), which gives both true matches and simulated "new station" impostors.
2. Novelty: kNN distance from a sensor's station-months to all PS1 station-months, compared
   with the leave-one-station-out distribution inside PS1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import (
    MATCH_THRESHOLD_GRID,
    NOVELTY_K,
    QUERY_YEAR_CALIBRATION,
    REFERENCE_YEAR_PS2,
    REFERENCE_YEARS_CALIBRATION,
)
from .features import FINGERPRINT_FEATURES, MODEL_FEATURES


# --------------------------------------------------------------------------------------------
# Re-identification
# --------------------------------------------------------------------------------------------
def fingerprint_scale(f1: pd.DataFrame, features=FINGERPRINT_FEATURES) -> pd.Series:
    """Typical between-station spread of each fingerprint feature (mean of per-month std)."""
    return f1.groupby("month")[features].std().mean()


def fingerprint_distance(a: pd.DataFrame, b: pd.DataFrame, scale: pd.Series,
                         features=FINGERPRINT_FEATURES, min_common: int = 2) -> float:
    """Median over shared calendar months of the scaled Euclidean distance between two sensors.

    Comparing like-for-like calendar months removes the seasonal cycle from the comparison.
    """
    A = a.groupby("month")[features].mean()
    B = b.groupby("month")[features].mean()
    common = A.index.intersection(B.index)
    if len(common) < min_common:
        return np.nan
    diff = (A.loc[common] - B.loc[common]) / scale[features]
    with np.errstate(all="ignore"):
        d = np.sqrt(np.nanmean(diff.to_numpy() ** 2, axis=1))
    return float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan


def distance_matrix(queries: pd.DataFrame, qkey: str, refs: pd.DataFrame, rkey: str,
                    scale: pd.Series) -> pd.DataFrame:
    """Fingerprint distance between every query sensor (rows) and reference station (columns)."""
    qg = {k: g for k, g in queries.groupby(qkey)}
    rg = {k: g for k, g in refs.groupby(rkey)}
    return pd.DataFrame({r: {q: fingerprint_distance(qg[q], rg[r], scale) for q in sorted(qg)} for r in sorted(rg)})


@dataclass
class MatchCalibration:
    D: pd.DataFrame          # query (2019+) x reference (2016-18) distances
    d_true: pd.Series        # distance to the same station
    d_imp: pd.Series         # distance to the closest other station
    top1: pd.Series          # nearest neighbour is the same station
    threshold: float
    recall: float            # share of known stations correctly re-identified at the threshold
    reject: float            # share of simulated new stations correctly rejected
    balanced_accuracy: float
    grid: np.ndarray
    ba_curve: np.ndarray


def calibrate_matching(f1: pd.DataFrame, scale: pd.Series) -> MatchCalibration:
    """Choose the distance threshold that maximises balanced accuracy on PS1."""
    y0, y1 = REFERENCE_YEARS_CALIBRATION
    ref = f1[(f1.year >= y0) & (f1.year <= y1)]
    qry = f1[f1.year >= QUERY_YEAR_CALIBRATION]
    qs = sorted(set(ref.station_id) & set(qry.station_id))
    D = distance_matrix(qry[qry.station_id.isin(qs)], "station_id", ref, "station_id", scale)
    d_true = pd.Series({q: D.loc[q, q] for q in qs})
    d_imp = pd.Series({q: D.loc[q].drop(q).min() for q in qs})
    top1 = pd.Series({q: D.loc[q].idxmin() == q for q in qs})
    grid = np.arange(*MATCH_THRESHOLD_GRID)
    ba = np.array([((top1 & (d_true < t)).mean() + (d_imp >= t).mean()) / 2 for t in grid])
    th = float(grid[int(np.argmax(ba))])
    return MatchCalibration(D, d_true, d_imp, top1, th, float((top1 & (d_true < th)).mean()),
                            float((d_imp >= th).mean()), float(ba.max()), grid, ba)


def match_sensors(f1: pd.DataFrame, f2: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """One-to-one assignment (Hungarian algorithm) of PS2 sensors to PS1 stations.

    Each PS1 station can explain at most one PS2 sensor. The reference is each PS1 station's
    most recent data (from ``REFERENCE_YEAR_PS2`` on), closest in time to PS2.
    """
    ref = f1[f1.year >= REFERENCE_YEAR_PS2]
    M = distance_matrix(f2, "file", ref, "station_id", scale)
    cost = M.fillna(99).to_numpy()
    ri, ci = linear_sum_assignment(cost)
    out = pd.DataFrame({"match_station": M.columns[ci], "match_dist": cost[ri, ci]}, index=M.index[ri])
    out.loc[out.match_dist >= 99, ["match_station", "match_dist"]] = [None, np.nan]
    out.index.name = "file"
    return out


# --------------------------------------------------------------------------------------------
# Novelty
# --------------------------------------------------------------------------------------------
def knn_novelty(query: np.ndarray, reference: np.ndarray, k: int = NOVELTY_K) -> np.ndarray:
    """Mean distance from each query row to its k nearest reference rows."""
    d = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(-1))
    return np.sort(d, axis=1)[:, :k].mean(1)


class NoveltyScorer:
    """kNN climate novelty against PS1, calibrated leave-one-station-out."""

    season_weight = 2.0   # up-weight month sin/cos so months are compared like-with-like

    def __init__(self, features=MODEL_FEATURES):
        self.features = list(features)

    def _z(self, f: pd.DataFrame) -> np.ndarray:
        z = self.scaler.transform(self.imputer.transform(f[self.features]))
        z[:, [self.features.index("m_sin"), self.features.index("m_cos")]] *= self.season_weight
        return z

    def fit(self, f1: pd.DataFrame) -> NoveltyScorer:
        self.imputer = SimpleImputer(strategy="median").fit(f1[self.features])
        self.scaler = StandardScaler().fit(self.imputer.transform(f1[self.features]))
        self.Z1 = self._z(f1)
        null = {}
        for s in f1.station_id.unique():
            msk = (f1.station_id == s).to_numpy()
            null[s] = float(np.median(knn_novelty(self.Z1[msk], self.Z1[~msk])))
        self.null_ = pd.Series(null)
        self.threshold_ = float(self.null_.max())
        return self

    def score(self, f: pd.DataFrame, key: str) -> pd.Series:
        s = pd.Series(knn_novelty(self._z(f), self.Z1), index=f.index)
        return s.groupby(f[key]).median()
