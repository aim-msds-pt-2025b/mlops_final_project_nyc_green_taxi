# MLOps Final Project — NYC Green Taxi Trip Duration Prediction

This repository implements an **end‑to‑end ML system** for predicting NYC Green Taxi trip durations with comprehensive MLOps practices including experiment tracking, drift detection, model serving, and containerized deployment.

## 🎯 Project Overview

**Objective**: Predict taxi trip duration using NYC TLC Green Taxi data with a production-ready MLOps pipeline.

**Key Features**:
- **RandomForest Regressor** with feature engineering pipeline
- **MLflow** experiment tracking and model registry
- **SHAP** model interpretability (Windows compatible)
- **Evidently AI** drift detection and monitoring
- **FastAPI** model serving with validation
- **Docker Compose** orchestration
- **Comprehensive testing** with pytest

---

## 📘 Detailed Implementation & Developer Notes (Consolidated)

The system is designed to satisfy an end‑to‑end MLOps rubric emphasizing **reproducibility, automation, observability, and deployment readiness**. Below is a concise breakdown of implementation components merged from the former `README_dev.md`.

### 1. ML Problem & Dataset
* **Task**: Supervised regression — predict trip duration (minutes).
* **Dataset**: NYC TLC Green Taxi (Jan 2024) ingested programmatically.
* **Preprocessing**: Compute duration from timestamps; filter 1–120 minutes; engineer `hour`, `day_of_week`; retain categorical location & payment fields.
* **Docs**: See `docs/data_dictionary.md`, `docs/dataset.md`.

### 2. Architecture
Documented in `docs/architecture.md` + `docs/architecture.png`.
* **Airflow**: Orchestrates training, deployment, drift workflows.
* **MLflow**: Tracking + Model Registry (backed by Postgres / artifacts volume).
* **FastAPI**: Real‑time inference + metadata endpoints.
* **Evidently**: Drift & target drift reporting.
* **Docker Compose**: Multi‑service runtime (MLflow, Postgres, Airflow, API).

### 3. Airflow DAGs
| DAG | Purpose | Key Tasks |
|-----|---------|-----------|
| `training_dag` | Retrain & log model | ingest → transform → train → validate → log |
| `drift_dag` | Periodic drift monitoring | simulate / fetch → generate Evidently report → log |
| `deployment_dag` | Promote best model | evaluate → register → promote → reload API |

### 4. Experiment Tracking (MLflow)
* Logs: parameters, metrics (MAE, MSE, R²), artifacts (model, preprocessor, SHAP plot, drift reports, metrics JSON).
* Registry: Production model promotion automated via deployment DAG or manual script `src/deployment/promote.py`.

### 5. Model Serving (FastAPI)
Endpoints:
* `POST /predict` — prediction with Pydantic validation.
* `GET /model` — metadata: params, metrics, schema, important features.
* `POST /reload` — reload Production model from registry.
* `GET /health` — service readiness.
* `GET /docs` — Swagger UI.

### 6. Containerization
* Distinct Dockerfiles under `docker/` for MLflow, FastAPI, Airflow.
* `docker-compose.yml` wires volumes (artifacts, db data) & networks.
* Exposed ports: Airflow 8080, MLflow 5000, FastAPI 8000, Postgres 5432.

### 7. Drift Detection (Evidently)
* Drift simulation: `src/data/simulate_drift.py`.
* Report generation: `src/monitoring/generate_drift.py` (HTML & logged to MLflow).
* Strategy & thresholds: `docs/drift_plan.md`.

### 8. Testing & CI/CD
* **Pytest**: Data shape & feature tests (`tests/`).
* **Pre‑commit**: Ruff lint/format, whitespace, secret scan.
* **CI (GitHub Actions)**: Install → pre‑commit → smoke pipeline → tests.

### 9. Demo Notebook
`notebooks/demo.ipynb` provides an interactive walkthrough: predictions, model metadata, experiment browsing, drift simulation, pipeline overview, validation checks.

### 10. Documentation & Tooling
* `Makefile` shortcuts: `data`, `transform`, `train`, `validate`, `drift`, `api`.
* Config centralization: `config.yaml` & `src/config.py`.
* Reproducible dependency sets via `pyproject.toml` + lock file.

> Former developer companion file `README_dev.md` has been merged here for a single authoritative source.

---

## 📊 Model Performance

- **MAE (validation)**: 3.38 minutes
- **R² (validation)**: 0.65
- **Model**: RandomForest with 150 estimators
- **Features**: trip_distance, passenger_count, location IDs, hour, day_of_week, payment_type

## 🚀 Quickstart

### Clone and Setup:

```bash
git clone <this_repo>
cd mlops_final_project_nyc_green_taxi
uv sync  # or pip install -r requirements.txt
```

### Containerized Deployment using Airflow for Orchestration

In order to access logs from the Airflow WebUI, set an secret key in the `.env` file like so:
```bash
echo AIRFLOW_SECRET_KEY=my-secret >> .env
```
On Linux, you should also set the UID with
```bash
echo "AIRFLOW_UID=$(id -u)" >> .env
```
Then build and launch all services with:
```bash
docker-compose up --build
```

**Services Available**:
- **Airflow UI**: http://localhost:8080 (orchestration, DAG execution)
- **MLflow UI**: http://localhost:5000 (experiment tracking, model registry)
- **FastAPI API**: http://localhost:8000 (model serving with docs at /docs)
- **PostgreSQL**: localhost:5432 (persistent storage)

Once the services have all been created (verify this with `docker compose ps`), trigger the deployment with
```bash
docker compose exec airflow-scheduler airflow dags trigger deployment_dag
```
You should then be able to verify that the model serving API is healthy (at `http://localhost:8000/health`) and query it with
```bash
python -m src.serve.sample_request
```


### Local Development

1. **Start MLflow Server**:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
export MLFLOW_TRACKING_URI=http://localhost:5000
```

2. **Run Complete Pipeline**:
```bash
python -m src.models.train            # Train model + log to MLflow
python -m src.monitoring.generate_drift  # Generate drift report
python -m src.deployment.promote     # Promote model to "Production"
```

3. **Start API Server**:
```bash
python -m src.serve.app              # Start FastAPI server (port 8000)
python -m src.serve.sample_request   # Test with sample requests
```

## 📈 Pipeline Results

### Model Training Success
```
✅ Data loaded: 57,457 samples
✅ Features engineered: trip_distance, passenger_count, pickup/dropoff locations
✅ Model trained: RandomForest (150 estimators, 42 random_state)
✅ Validation metrics: MAE=3.38min, MSE=38.34, R²=0.65
✅ SHAP interpretability: Feature importance visualizations generated
✅ MLflow logged: Experiment "green_taxi_duration", model promoted to Production
```

### Drift Detection Verified
```
✅ Reference dataset: Jan 2024 (original training data)
✅ Current dataset: Simulated with moderate drift
✅ Evidently report: HTML dashboard with drift analysis
✅ Drift status: Target drift detected (expected for demo)
```

### API Testing Validated
```
✅ Health endpoint: Service responsive
✅ Model prediction: Realistic outputs validated
   - 5.2 miles → 23.3 minutes (highway speed)
   - 1.2 miles → 5.9 minutes (city traffic)
   - 10.5 miles → 36.2 minutes (long trip)
✅ Input validation: Proper error handling for invalid inputs
✅ All 8 tests passing: Comprehensive test suite verified
```

## 🔧 MLflow UI

Visit **http://localhost:5000** to explore:
- **Experiments**: Training runs with metrics, parameters, and artifacts
- **Models Registry**: Production model versions with signatures
- **Artifacts**: SHAP plots, model files, and drift reports

## 🚀 API Endpoints

The FastAPI server (http://localhost:8000) provides:

- **POST /predict** — Make predictions with taxi trip data
- **GET /model** — View model metadata, hyperparameters, and feature schema
- **POST /reload** — Reload the champion model from MLflow Registry
- **GET /health** — Service health check
- **GET /docs** — Interactive API documentation (Swagger UI)

## 🧪 Testing

Run the comprehensive test suite:
```bash
pytest -v
```

**Test Coverage**:
- **Data processing**: Validation of feature engineering pipeline
- **Transform functions**: Unit tests for data transformations
- **API functionality**: End-to-end model prediction validation
- **Input validation**: Error handling for malformed requests

## 🐳 Airflow Integration

The project includes Airflow DAGs for automated workflows:

```bash
make airflow-init   # One-time initialization
docker-compose up   # Includes Airflow services
```

**DAGs Available**:
- **Training DAG**: Automated model retraining pipeline
- **Drift DAG**: Scheduled drift detection and reporting
- **Deployment DAG**: Model promotion and deployment automation

**Airflow UI**: http://localhost:8080 (admin/admin)

## 📁 Project Structure

```
├── config.yaml              # Configuration settings
├── docker-compose.yml       # Multi-service orchestration
├── requirements.txt         # Python dependencies
├── src/
│   ├── data/                # Data ingestion and simulation
│   ├── features/            # Feature engineering pipeline
│   ├── models/              # Model training and validation
│   ├── deployment/          # Model promotion utilities
│   ├── monitoring/          # Drift detection and reporting
│   └── serve/               # FastAPI model serving
├── tests/                   # Comprehensive test suite
├── docs/                    # Architecture and documentation
├── dags/                    # Airflow workflow definitions
└── notebooks/               # Jupyter demo and exploration
```

## 📚 Documentation

Explore detailed documentation in the `docs/` folder:
- **Architecture**: System design and component interactions
- **Data Dictionary**: Feature definitions and data schema
- **Dataset Notes**: NYC TLC Green Taxi data insights
- **Drift Plan**: Monitoring strategy and thresholds

## 🛠️ Make Targets

Convenient commands for common tasks:
```bash
make data        # Download and prepare dataset
make transform   # Run feature engineering
make train       # Train model with MLflow logging
make validate    # Validate model performance
make drift       # Generate drift detection report
make api         # Start FastAPI development server
```

## 🎯 Next Steps

1. **Scale Data**: Process larger datasets with distributed computing
2. **Advanced Models**: Experiment with XGBoost, neural networks
3. **Real-time Serving**: Deploy to cloud with autoscaling
4. **Monitoring**: Set up alerts for model performance degradation
5. **A/B Testing**: Implement champion/challenger model comparison

---

**Built with**: Python, MLflow, FastAPI, Docker, Airflow, Evidently AI, SHAP
