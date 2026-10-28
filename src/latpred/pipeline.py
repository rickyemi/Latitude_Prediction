"""End-to-end pipeline.

    python -m latpred.pipeline --data-dir data/raw/DataScienceChallenge --output-dir reports

Steps: preprocess -> features -> model comparison (LOSO) -> extrapolation test -> physics ->
matching calibration -> novelty -> simulated validation -> final fit -> PS2 predictions.
Writes prediction_results.csv, metrics.json, processed feature tables and all figures.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from . import decision, evaluation, features, matching, modeling, physics, preprocessing
from . import visualization as viz
from .config import Paths

log = logging.getLogger("latpred")


def run(paths: Paths | None = None, make_figures: bool = True) -> dict:
    """Run every step and return a dict of intermediate results (used by the notebook)."""
    paths = paths or Paths()
    paths.make_dirs()
    figs = paths.figures if make_figures else None
    t0 = time.time()
    R: dict = {"paths": paths}

    log.info("1/9 preprocessing")
    ps1 = preprocessing.assign_station_ids(preprocessing.load_folder(Path(paths.data_dir) / "ps1"))
    ps2 = preprocessing.load_folder(Path(paths.data_dir) / "ps2")
    coords = preprocessing.station_coordinates(ps1)
    R.update(ps1=ps1, ps2=ps2, coords=coords, audit=preprocessing.audit_table(ps1, ps2),
             missing=preprocessing.missingness(ps1, ps2))

    log.info("2/9 feature engineering")
    f1 = features.station_month_features(ps1, "station_id")
    f1["lat"] = f1.station_id.map(coords.lat)
    f2 = features.station_month_features(ps2, "file")
    f1.to_csv(paths.processed_dir / "ps1_station_month_features.csv", index=False)
    f2.to_csv(paths.processed_dir / "ps2_station_month_features.csv", index=False)
    R.update(f1=f1, f2=f2)

    log.info("3/9 model comparison (leave-one-station-out)")
    cv, oof = modeling.compare_models(f1)
    final_model = modeling.fit_final(f1)
    importance = modeling.feature_importance(final_model, f1)
    R.update(cv=cv, oof=oof, model=final_model, importance=importance)

    log.info("4/9 extrapolation test")
    R["extrapolation"] = evaluation.extrapolation_test(f1, coords.lat)

    log.info("5/9 physics inversion")
    phys1 = physics.physics_table(ps1, "station_id")
    phys2 = physics.physics_table(ps2, "file")
    phys_eval = physics.evaluate_physics(phys1, coords)
    R.update(phys1=phys1, phys2=phys2, phys_eval=phys_eval)

    log.info("6/9 matching calibration")
    scale = matching.fingerprint_scale(f1)
    cal = matching.calibrate_matching(f1, scale)
    R.update(fp_scale=scale, calibration=cal)

    log.info("7/9 novelty")
    nov = matching.NoveltyScorer().fit(f1)
    R.update(novelty=nov, ps2_novelty=nov.score(f2, "file"))

    log.info("8/9 simulated end-to-end validation")
    sim, sim_summary = evaluation.simulate_decision_rule(cal, oof[modeling.BEST_MODEL].pred, coords.lat)
    R.update(simulation=sim, simulation_summary=sim_summary)

    log.info("9/9 PS2 predictions")
    res = pd.DataFrame(index=pd.Index(sorted(ps2.file.unique()), name="file"))
    res = (res.join(modeling.predict_stations(final_model, f2, "file"))
              .join(phys2[["phys_lat", "phys_lon", "phys_k", "phys_rmse", "phys_ok"]])
              .join(matching.match_sensors(f1, f2, scale)))
    res["novelty"] = R["ps2_novelty"]
    res = decision.apply_decision_rule(res, coords=coords, match_th=cal.threshold,
                                       novelty_th=nov.threshold_, phys_p90=phys_eval["lat_p90"])
    out = decision.to_output_table(res)
    out.to_csv(paths.output_dir / "prediction_results.csv", index=False)
    R.update(results=res, predictions=out)

    metrics = {
        "cv_station_level": cv.round(4).reset_index().to_dict(orient="records"),
        "physics": phys_eval,
        "matching": {"threshold": cal.threshold, "top1_accuracy": float(cal.top1.mean()),
                     "recall_known": cal.recall, "reject_new": cal.reject, "balanced_accuracy": cal.balanced_accuracy},
        "novelty_threshold": nov.threshold_,
        "simulation": sim_summary.round(4).reset_index().to_dict(orient="records"),
        "ps2_status_counts": out.status.value_counts().to_dict(),
        "runtime_s": round(time.time() - t0, 1),
    }
    (paths.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    R["metrics"] = metrics

    if make_figures:
        log.info("figures -> %s", figs)
        make_all_figures(R, figs)
    log.info("done in %.0f s", time.time() - t0)
    return R


def make_all_figures(R: dict, fig_dir) -> None:
    """Render every PNG from the results dict."""
    viz.plot_missingness(R["missing"], fig_dir)
    viz.plot_ps2_coverage(R["ps2"], fig_dir)
    viz.plot_seasonal_cycles(R["f1"], R["f2"], R["coords"].lat, fig_dir)
    viz.plot_feature_vs_latitude(R["f1"], R["coords"].lat, fig_dir=fig_dir)
    viz.plot_model_comparison(R["cv"], fig_dir)
    viz.plot_cv_predictions(R["oof"][modeling.BEST_MODEL], modeling.BEST_MODEL, fig_dir)
    viz.plot_feature_importance(R["importance"], fig_dir=fig_dir)
    viz.plot_extrapolation(R["extrapolation"], fig_dir)
    viz.plot_physics_validation(R["phys1"], R["coords"], fig_dir)
    viz.plot_matching_calibration(R["calibration"], fig_dir)
    viz.plot_novelty(R["novelty"].null_, R["ps2_novelty"], R["novelty"].threshold_, fig_dir)
    viz.plot_simulation(R["simulation_summary"], fig_dir)
    viz.plot_ps2_map(R["results"], R["coords"], fig_dir)
    viz.plot_ps2_status(R["results"], fig_dir)
    import matplotlib.pyplot as plt
    plt.close("all")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Predict weather-station latitude (PS1 -> PS2).")
    ap.add_argument("--data-dir", default=None, help="folder containing ps1/ and ps2/")
    ap.add_argument("--output-dir", default=None, help="where predictions, metrics and figures go")
    ap.add_argument("--processed-dir", default=None, help="where processed feature tables go")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = Paths()
    if a.data_dir:
        p.data_dir = Path(a.data_dir)
    if a.output_dir:
        p.output_dir = Path(a.output_dir)
    if a.processed_dir:
        p.processed_dir = Path(a.processed_dir)
    R = run(p, make_figures=not a.no_figures)
    print(R["predictions"].status.value_counts().to_string())
    print(f"predictions -> {p.output_dir / 'prediction_results.csv'}")


if __name__ == "__main__":
    main()
