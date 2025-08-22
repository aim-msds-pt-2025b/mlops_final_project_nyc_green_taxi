import logging
import os
import time

import mlflow
import requests
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from src.config import get_tracking_uri, load_config
from src.logging_utils import setup_logging

log = logging.getLogger(__name__)


def try_post(url: str, path="/reload", retries=3, delay=1.0, timeout=5.0):
    last_err = None
    for i in range(retries):
        try:
            r = requests.post(f"{url.rstrip('/')}{path}", timeout=timeout)
            log.info(
                "api reload", extra={"url": url, "status": r.status_code, "body": r.text[:200]}
            )
            return True
        except Exception as e:
            last_err = e
            time.sleep(delay)
    log.warning("api reload failed", extra={"url": url, "error": str(last_err)})
    return False


def reload_fastapi():
    # Respect explicit env first
    candidates = []
    env_url = os.environ.get("API_URL")
    if env_url:
        candidates.append(env_url)

    # Detect if likely inside Docker (heuristic)
    inside_docker = os.path.exists("/.dockerenv") or os.environ.get("RUNNING_IN_DOCKER") == "1"

    # Add context-aware fallbacks
    if inside_docker:
        # Service name on the docker network
        candidates += ["http://fastapi:8000", "http://web:8000"]
        # On Linux containers, host gateway is often available:
        candidates += ["http://host.docker.internal:8000", "http://172.17.0.1:8000"]
    else:
        # Running on host
        candidates += ["http://localhost:8000", "http://127.0.0.1:8000", "http://fastapi:8000"]

    # De-duplicate while preserving order
    seen = set()
    candidates = [u for u in candidates if not (u in seen or seen.add(u))]

    for url in candidates:
        if try_post(url):
            return True
    return False


def main():
    cfg = load_config()
    setup_logging(cfg)
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    client = MlflowClient()
    exp = client.get_experiment_by_name(cfg.mlflow["experiment"])
    runs = client.search_runs(
        exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=5
    )
    if not runs:
        raise RuntimeError("No training runs")

    # Find the latest run with metrics
    run = None
    for r in runs:
        if "mae_val" in r.data.metrics and "r2_val" in r.data.metrics:
            run = r
            break

    if not run:
        raise RuntimeError("No training runs with metrics found")

    mae = run.data.metrics.get("mae_val", 999)
    r2 = run.data.metrics.get("r2_val", -999)
    if mae <= cfg.validation_thresholds["mae_max"] and r2 >= cfg.validation_thresholds["r2_min"]:
        logging.getLogger(__name__).info(
            "thresholds passed", extra={"mae": round(mae, 3), "r2": round(r2, 3)}
        )
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
    logging.getLogger(__name__).info(
        "promoted to production", extra={"model": model_name, "version": mv.version}
    )

    # Reload FastAPI with improved retry logic
    if not reload_fastapi():
        log.warning("api reload ultimately failed; API may still be serving previous model")


if __name__ == "__main__":
    main()
