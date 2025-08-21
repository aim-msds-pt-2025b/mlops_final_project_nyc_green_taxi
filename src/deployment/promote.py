import os

import mlflow
import requests
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.config import get_tracking_uri, load_config


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg.mlflow["experiment"])
    runs = client.search_runs(
        exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1
    )
    if not runs:
        raise RuntimeError("No training runs")
    run = runs[0]

    mae = run.data.metrics.get("mae_val", 999)
    r2 = run.data.metrics.get("r2_val", -999)
    if (
        mae <= cfg.validation_thresholds["mae_max"]
        and r2 >= cfg.validation_thresholds["r2_min"]
    ):
        print(f"[promote] thresholds passed (mae={mae:.3f}, r2={r2:.3f})")
    else:
        raise RuntimeError(f"[promote] thresholds failed (mae={mae:.3f}, r2={r2:.3f})")

    # Register model
    model_uri = f"runs:/{run.info.run_id}/model"
    model_name = cfg.mlflow["model_name"]
    try:
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    except RestException:
        # model may already exist; register new version
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[promote] Promoted {model_name} v{mv.version} to Production")

    # Reload FastAPI
    api = os.environ.get("API_URL", "http://fastapi:8000")
    try:
        r = requests.post(f"{api}/reload", timeout=10)
        print("[promote] Reload response:", r.status_code, r.text[:200])
    except Exception as e:
        print("[promote] Reload failed:", e)


if __name__ == "__main__":
    main()
