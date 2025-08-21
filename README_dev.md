# Dev Notes
Final Project Write-Up — End-to-End MLOps System
Overview
This project delivers a fully compliant, end-to-end machine learning system built with modern MLOps practices. The focus is not on model complexity but on pipeline orchestration, reproducibility, deployment, and monitoring. The system uses the NYC TLC Green Taxi Trip Records (Jan 2024) dataset to predict trip duration (minutes) as a supervised regression problem.
The implementation achieves 100% alignment with the rubric requirements, covering data ingestion, preprocessing, training, validation, deployment, monitoring, CI/CD, containerization, documentation, and testing.
Accomplishments
1. Machine Learning Problem & Dataset
Problem: Regression — predict trip duration in minutes.
Dataset: NYC Green Taxi Trip Records (Jan 2024), programmatically ingested from the TLC CloudFront endpoint.
Preprocessing: Duration computed from pickup/dropoff timestamps; filtered to 1–120 minutes. Engineered temporal features (hour, day_of_week) plus categorical location/payment features.
Documentation: Dataset notes, data dictionary, and drift plan are included in docs/.
2. System Architecture
Documented in docs/architecture.md with a supporting diagram (docs/architecture.png).
Components:
Airflow orchestrates three DAGs.
MLflow manages experiment tracking, artifacts, and model registry.
FastAPI serves predictions and model metadata.
Evidently AI monitors drift and logs reports.
Docker Compose orchestrates the entire system with separate services for MLflow, Airflow, FastAPI, and Postgres.
3. Data Pipeline (Airflow)
Training DAG (training_dag.py):

Ingests → Transforms → Trains → Validates model.
Drift DAG (drift_dag.py):

Simulates new data → Generates Evidently reports (DataDrift + TargetDrift).
Deployment DAG (deployment_dag.py):

Evaluates metrics → Registers model to MLflow Registry → Promotes to Production → Reloads FastAPI.
4. Experiment Tracking (MLflow)
Experiment setup with MLflow using Docker + Postgres backend.
Parameters and metrics logged (MAE, R², hyperparameters).
Artifacts logged:
Trained model
Preprocessor
SHAP summary plot for explainability
Drift reports (HTML)
Metrics JSON
Model Registry: Trained models are registered and promoted to Production if they meet defined thresholds.
5. Model Serving (FastAPI)
Endpoints:
/predict: serve predictions with Pydantic input validation.
/model: return hyperparameters, metrics, input schema, and important features.
/reload: reload champion model from MLflow Registry.
/health: simple readiness check.
Swagger UI automatically available at /docs.
Error handling: invalid inputs or missing models return clear error responses.
6. Containerization (Docker + Compose)
Separate Dockerfiles for Airflow, MLflow, and FastAPI services.
docker-compose.yml orchestrates all components, networks, and volumes (MLflow artifacts, Postgres DB).
Ports exposed and documented:
Airflow: :8080
MLflow: :5000
FastAPI: :8000
7. Drift Detection (Evidently AI)
Simulation: src/data/simulate_drift.py perturbs features and introduces drift.
Drift reports: src/monitoring/generate_drift.py produces Evidently HTML reports for data and target drift.
Logging: Drift reports are logged as MLflow artifacts and linked to experiment runs.
Documentation: Drift plan is written in docs/drift_plan.md.
8. Testing & CI/CD
Unit tests (pytest): Validate preprocessing ranges, config existence, data ingestion.
Pre-commit hooks: Ruff linting/formatting, trailing whitespace fixes, key detection.
CI/CD (GitHub Actions):
Installs dependencies
Runs pre-commit hooks
Executes smoke pipeline (data → train → drift)
Runs unit tests
9. Jupyter Notebook Demo
notebooks/demo.ipynb demonstrates:
Sending requests to /predict endpoint
Fetching latest run ID from MLflow
Explaining workflow with markdown and comments
Provides an interactive, reproducible walkthrough of the deployed system.
10. Documentation & Deliverables
Docs: docs/architecture.md, docs/data_dictionary.md, docs/drift_plan.md, docs/dataset.md.
Pinned dependencies in requirements.txt and pyproject.toml.
Makefile with shortcuts (make data, make train, make drift, make api).
README.md with complete setup, usage, and service instructions.
Repository structure matches rubric expectations (src/, dags/, tests/, docs/, docker/, notebooks/).
 
 