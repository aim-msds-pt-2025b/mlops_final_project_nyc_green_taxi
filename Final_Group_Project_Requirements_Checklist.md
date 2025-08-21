# Final Group Project Requirements (with Checklists)

This project assesses your understanding of MLOps principles by requiring you to build a functional machine
learning system. You will develop a system encompassing data ingestion, preprocessing, model training,
validation, deployment, and monitoring. The system will be containerized using Docker, orchestrated by
Airflow, tracked using MLflow, and served via a FastAPI application.

---

## 1. Project Scope and Data
- [ ] Choose a supervised ML problem (classification or regression)
- [ ] Ensure problem is manageable (not overly complex)
- [ ] Select or create dataset (≥ 500 rows, ideally ≥ 1000 rows)
- [ ] Implement `src/data/get_data.py` to fetch or generate data
- [ ] Save dataset to `data/raw/`
- [ ] Provide documentation (`docs/dataset.md` or README)
- [ ] Provide data dictionary in `docs/data_dictionary.md`
- [ ] Plan for drift simulation in `docs/drift_plan.md`

## 2. System Architecture and Components
- [ ] Create `docs/architecture.png` (diagram of system)
- [ ] Create `docs/architecture.md` (detailed explanation)

## 3. Data Pipeline (Airflow)
### Training DAG (`dags/training_dag.py`)
- [ ] Implement Data Ingestion (`src/data/ingest.py`)
- [ ] Implement Data Transformation (`src/features/transform.py`)
- [ ] Implement Model Training (`src/models/train.py`)
- [ ] Implement Model Validation (`src/models/validate.py`)

### Drift Detection DAG (`dags/drift_dag.py`)
- [ ] Implement Drift Detection with Evidently (`src/monitoring/generate_drift.py`)
- [ ] Log reports to MLflow

### Deployment DAG (`dags/deployment_dag.py`)
- [ ] Implement Model Promotion (`src/deployment/promote.py`)
- [ ] Automate redeployment of FastAPI

### Pipeline Standards
- [ ] Ensure reproducibility (fixed seeds, `config.yaml`)
- [ ] Implement error handling with logging
- [ ] Use environment variables/config for parameters

## 4. Experiment Tracking (MLflow)
- [ ] Deploy MLflow server (Docker, `http://mlflow:5000`)
- [ ] Log parameters, metrics, and artifacts
- [ ] Log SHAP plots and Evidently drift reports
- [ ] Ensure MLflow UI is accessible

## 5. Model Serving (FastAPI)
- [ ] Implement `/predict` endpoint (`src/serve/app.py`)
- [ ] Implement `/model` endpoint (hyperparams, features, schema)
- [ ] Load champion model from MLflow
- [ ] Validate input schema with Pydantic
- [ ] Handle errors with proper HTTP codes
- [ ] Ensure API docs available at `/docs`

## 6. Containerization (Docker)
- [ ] Create `docker/airflow.Dockerfile`
- [ ] Create `docker/mlflow.Dockerfile`
- [ ] Create `docker/fastapi.Dockerfile`
- [ ] Write `docker-compose.yml` to orchestrate all services
- [ ] Document exposed ports in README

## 7. Drift Detection (Evidently AI)
- [ ] Save reference dataset (`data/reference.parquet`)
- [ ] Simulate drifted dataset (`src/data/simulate_drift.py`)
- [ ] Generate Data Drift and Target Drift reports
- [ ] Log drift reports as MLflow artifacts
- [ ] Demonstrate drift detection in notebook

## 8. Testing (Pytest + GitHub Actions)
- [ ] Implement unit tests (`tests/` directory)
- [ ] Write tests for missing values, type conversions, edge cases
- [ ] Add GitHub Actions workflow (`.github/workflows/ci.yml`)
- [ ] Run pytest on pushes and PRs
- [ ] Configure pre-commit hooks with Ruff

## 9. Jupyter Notebook (`demo.ipynb`)
- [ ] Provide overview of system and setup instructions
- [ ] Show happy path (prediction + model info)
- [ ] Demonstrate drift detection reports from MLflow
- [ ] Add explanations and comments throughout
- [ ] Ensure reproducibility (restart + run all)

## 10. Submission
- [ ] Organize repo structure (`src/`, `data/`, `dags/`, `tests/`, `docs/`, `docker/`, `.github/`, `notebooks/`)
- [ ] Write `README.md` with setup instructions and dataset description
- [ ] Ensure code quality (PEP8, docstrings, meaningful names)
- [ ] Ensure completeness (end-to-end run works)

---

✅ Use this checklist to track progress and ensure full compliance with the project requirements.
