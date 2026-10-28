# Reproducible environment for the latitude-prediction pipeline.
# The image holds code and libraries only; data is mounted read-only at run time (see run.sh).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    REPO_ROOT=/app \
    DATA_DIR=/app/data/raw/DataScienceChallenge \
    OUTPUT_DIR=/app/reports \
    PROCESSED_DIR=/app/data/processed

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m ipykernel install --name python3 --display-name "Python 3"

COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .

COPY notebooks/ ./notebooks/
COPY docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
