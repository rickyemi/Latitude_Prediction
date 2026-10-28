#!/usr/bin/env bash
# Container entry point.
#   MODE=notebook (default): execute the notebook (writes predictions + figures) and export HTML
#   MODE=pipeline          : run the headless pipeline only (faster, no notebook)
set -euo pipefail
MODE="${MODE:-notebook}"
NB=01_latitude_prediction

if [[ "${MODE}" == "pipeline" ]]; then
  python -m latpred.pipeline --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}" --processed-dir "${PROCESSED_DIR}"
else
  cd /app/notebooks
  jupyter nbconvert --to notebook --execute "${NB}.ipynb" --ExecutePreprocessor.timeout=3600 \
      --output-dir "${OUTPUT_DIR}" --output "${NB}.ipynb"
  jupyter nbconvert --to html "${OUTPUT_DIR}/${NB}.ipynb" --output-dir "${OUTPUT_DIR}"
  rm -f "${OUTPUT_DIR}/${NB}.ipynb"   # keep only the HTML report; the source notebook lives in notebooks/
fi

test -f "${OUTPUT_DIR}/prediction_results.csv"
echo "[container] done: ${OUTPUT_DIR}/prediction_results.csv"
