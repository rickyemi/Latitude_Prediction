"""Decision rule: combine the ML, physics and matching evidence into a final prediction.

1. Out of region: novelty above every PS1 station, or a quality-checked physics latitude more
   than the physics 90th-percentile error outside the PS1 band  ->  physics latitude.
2. Known PS1 station: fingerprint distance below the calibrated threshold, not out of region,
   and not contradicted by a quality-checked physics latitude  ->  matched station's latitude.
3. New station inside the region  ->  ML latitude.
4. No usable data  ->  PS1 mean latitude, flagged ``insufficient_data``.
"""
from __future__ import annotations

import pandas as pd

OUTPUT_COLUMNS = ["file_name", "sensor_id", "predicted_latitude", "status", "in_ps1", "method", "confidence",
                  "ml_lat", "ml_q25", "ml_q75", "phys_lat", "phys_longitude_estimate", "nearest_ps1_station_code",
                  "nearest_ps1_latitude", "nearest_ps1_distance", "novelty", "n_months"]


def _decide(r, match_th: float, fallback_lat: float):
    if r.n_months == 0 and not r.phys_ok:
        return fallback_lat, "insufficient_data", "ps1_mean", "low"
    if r.out_of_region:
        if r.phys_ok:
            return r.phys_lat, "new_out_of_region", "physics", "medium"
        return r.ml_lat, "new_out_of_region", "ml (no valid physics fit; likely biased north)", "low"
    if r.matched:
        return r.match_lat, "known_ps1_station", "reidentified", "high" if r.match_dist < 0.75 * match_th else "medium"
    if r.match_dist < match_th and r.match_conflict:
        return r.ml_lat, "new_in_region", "ml (match rejected: physics disagrees)", "low"
    if pd.isna(r.ml_lat):
        return r.phys_lat, "new_in_region", "physics", "low"
    return r.ml_lat, "new_in_region", "ml", "medium" if r.n_months >= 6 else "low"


def apply_decision_rule(res: pd.DataFrame, *, coords: pd.DataFrame, match_th: float, novelty_th: float,
                        phys_p90: float) -> pd.DataFrame:
    """Add status/prediction columns to the per-sensor evidence table ``res`` (indexed by file)."""
    res = res.copy()
    res["n_months"] = res["n_months"].fillna(0).astype(int)
    res["phys_ok"] = res["phys_ok"].fillna(False).astype(bool)
    res["match_lat"] = res.match_station.map(coords.lat)
    res["match_code"] = res.match_station.map(coords.code)
    lat_lo, lat_hi = coords.lat.min(), coords.lat.max()
    phys_outside = res.phys_ok & ((res.phys_lat < lat_lo - phys_p90) | (res.phys_lat > lat_hi + phys_p90))
    res["out_of_region"] = (phys_outside | (res.novelty > novelty_th)).astype(bool)
    res["match_conflict"] = res.phys_ok & ((res.phys_lat - res.match_lat).abs() > phys_p90)
    res["matched"] = (res.match_dist < match_th) & ~res.out_of_region & ~res.match_conflict
    dec = res.apply(lambda r: pd.Series(_decide(r, match_th, coords.lat.mean()),
                                        index=["predicted_latitude", "status", "method", "confidence"]), axis=1)
    res = res.join(dec)
    res["in_ps1"] = res.status.eq("known_ps1_station")
    return res


def to_output_table(res: pd.DataFrame) -> pd.DataFrame:
    """Tidy, documented output table (one row per PS2 sensor)."""
    out = res.reset_index().rename(columns={"file": "file_name", "phys_lon": "phys_longitude_estimate",
                                            "match_code": "nearest_ps1_station_code",
                                            "match_lat": "nearest_ps1_latitude", "match_dist": "nearest_ps1_distance"})
    out["sensor_id"] = out.file_name.str.extract(r"_(\d+)_test")[0]
    out = out[OUTPUT_COLUMNS]
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(3)
    return out.sort_values("predicted_latitude").reset_index(drop=True)
