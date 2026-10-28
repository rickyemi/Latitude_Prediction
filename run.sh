#!/usr/bin/env bash
# Build the Docker image and run the pipeline in a container.
#
# Usage:
#   ./run.sh [DATA_DIR] [MODE]
#     DATA_DIR  folder containing ps1/ and ps2/   (default: ./data/raw/DataScienceChallenge)
#     MODE      notebook (default) | pipeline
#
# Outputs (in ./reports): prediction_results.csv, metrics.json, figures/*.png,
#                         01_latitude_prediction.html (notebook mode)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "${1:-${ROOT}/data/raw/DataScienceChallenge}" && pwd)"
MODE="${2:-notebook}"
IMAGE="latpred:latest"

for sub in ps1 ps2; do
  [[ -d "${DATA_DIR}/${sub}" ]] || { echo "ERROR: ${DATA_DIR}/${sub} not found (see data/README.md)" >&2; exit 1; }
done
mkdir -p "${ROOT}/reports" "${ROOT}/data/processed"

echo "==> (a) building image ${IMAGE}"
docker build -t "${IMAGE}" "${ROOT}"

echo "==> (b) running container (mode: ${MODE})"
docker run --rm \
  -e MODE="${MODE}" \
  -v "${DATA_DIR}:/app/data/raw/DataScienceChallenge:ro" \
  -v "${ROOT}/reports:/app/reports" \
  -v "${ROOT}/data/processed:/app/data/processed" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  "${IMAGE}"

echo "==> done: ${ROOT}/reports/prediction_results.csv"
