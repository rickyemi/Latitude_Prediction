import numpy as np
import pandas as pd

from latpred.features import FINGERPRINT_FEATURES
from latpred.matching import fingerprint_distance, knn_novelty


def _fp(offset=0.0, months=range(1, 13)):
    rows = [{"month": m, **{c: 1.0 + offset + 0.1 * m for c in FINGERPRINT_FEATURES}} for m in months]
    return pd.DataFrame(rows)


def test_identical_fingerprints_distance_zero():
    scale = pd.Series(1.0, index=FINGERPRINT_FEATURES)
    assert fingerprint_distance(_fp(), _fp(), scale) == 0


def test_distance_grows_with_offset():
    scale = pd.Series(1.0, index=FINGERPRINT_FEATURES)
    assert fingerprint_distance(_fp(), _fp(0.5), scale) < fingerprint_distance(_fp(), _fp(2.0), scale)


def test_needs_common_months():
    scale = pd.Series(1.0, index=FINGERPRINT_FEATURES)
    assert np.isnan(fingerprint_distance(_fp(months=[1]), _fp(months=[6, 7]), scale))


def test_knn_novelty():
    ref = np.zeros((10, 3))
    assert knn_novelty(np.zeros((1, 3)), ref, k=3)[0] == 0
    assert knn_novelty(np.full((1, 3), 2.0), ref, k=3)[0] > 3
