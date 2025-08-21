# MLOps Final Project — NYC Green Taxi (100% Compliance)

This repository implements an **end‑to‑end ML system** that satisfies the rubric requirements:
- **Airflow** orchestration with 3 DAGs (training, drift, deployment)
- **Programmatic data ingestion** (NYC TLC Green Taxi Jan 2024 parquet)
- **Feature engineering** pipeline with type handling & range checks
- **Experiment tracking** with **MLflow** (+ Model Registry) and **SHAP** plots
- **Drift detection** using **Evidently AI** (HTML reports logged to MLflow)
- **FastAPI** model service with `/predict`, `/model`, `/reload`
- **Containerization** with separate Dockerfiles and **docker‑compose**
- **Pre‑commit + Ruff** for style and quality
- **GitHub Actions** CI to run tests & hooks
- **Docs**: architecture, data dictionary, drift plan, dataset notes
- **Jupyter notebook** demo

## Quickstart (Local Python)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Pull data
python -m src.data.get_data

# Transform
python -m src.features.transform

# Train & validate (logs to MLflow)
export MLFLOW_TRACKING_URI="file:./mlruns"   # Windows PowerShell: $env:MLFLOW_TRACKING_URI="file:./mlruns"
python -m src.models.train
python -m src.models.validate

# Generate drift (simulates & logs to MLflow)
python -m src.data.simulate_drift
python -m src.monitoring.generate_drift
```

## Services (Docker Compose)

```bash
docker compose up --build
```
Services:
- **Airflow UI**: http://localhost:8080 (admin/admin for first login if initialized)
- **MLflow UI**: http://localhost:5000
- **FastAPI**: http://localhost:8000 (Swagger at /docs)

The **FastAPI** app loads the **champion** model from the MLflow Model Registry (stage **Production**).
If none exists, it falls back to local artifacts.

## Endpoints

- `POST /predict` — returns `{"prediction": value}`
- `GET /model` — hyperparams, top feature names, input schema
- `POST /reload` — reload champion from MLflow Registry

## Make targets

```bash
make data
make transform
make train
make validate
make drift
make api
make airflow-init   # one-time init of Airflow DB and user
```

See **docs/** for architecture, data dictionary, and drift plan. The notebook `notebooks/demo.ipynb` demonstrates prediction and accessing MLflow artifacts and drift reports.
