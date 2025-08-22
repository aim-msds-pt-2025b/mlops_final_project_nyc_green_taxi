FROM python:3.11-slim
WORKDIR /app

# Create airflow user with same UID as airflow image (50000)
RUN groupadd -r airflow --gid=50000 && \
    useradd -r -g airflow --uid=50000 --home-dir=/home/airflow --shell=/bin/bash airflow && \
    mkdir -p /home/airflow && \
    chown -R airflow:airflow /home/airflow

RUN pip install --no-cache-dir mlflow==2.14.1 psycopg2-binary==2.9.9

# Create and chown the artifacts directory
RUN mkdir -p /mlflow_artifacts && \
    chown -R airflow:airflow /mlflow_artifacts

ENV BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow
ENV ARTIFACT_ROOT=/mlflow_artifacts

USER airflow
EXPOSE 5000
# hadolint ignore=DL3025
CMD mlflow server --backend-store-uri "$BACKEND_STORE_URI" --default-artifact-root "$ARTIFACT_ROOT" --host 0.0.0.0 --port 5000
