import mlflow
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
        raise RuntimeError("No runs found. Train first.")
    run = runs[0]
    data = {
        "run_id": run.info.run_id,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }
    print("[validate] latest run:", data)


if __name__ == "__main__":
    main()
