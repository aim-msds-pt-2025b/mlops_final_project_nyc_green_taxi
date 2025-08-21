from pathlib import Path

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config import get_tracking_uri, load_config


def main():
    cfg = load_config()
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    mlflow.set_experiment(cfg.mlflow["experiment"])

    ref = Path(cfg.paths["reference_path"])
    cur_dir = Path(cfg.paths["current_dir"])
    cur = cur_dir / "current.parquet"
    if not ref.exists() or not cur.exists():
        error_msg = (
            "Missing reference or current dataset. Run transform and simulate_drift."
        )
        raise FileNotFoundError(error_msg)

    ref_df = pd.read_parquet(ref)
    cur_df = pd.read_parquet(cur)

    with mlflow.start_run(run_name="drift-report"):
        data_report = Report(metrics=[DataDriftPreset()])
        data_report.run(reference_data=ref_df, current_data=cur_df)
        target_report = Report(metrics=[TargetDriftPreset()])
        target_report.run(reference_data=ref_df, current_data=cur_df)

        out_dir = Path("reports")
        out_dir.mkdir(exist_ok=True)
        data_html = out_dir / "data_drift.html"
        target_html = out_dir / "target_drift.html"
        data_report.save_html(str(data_html))
        target_report.save_html(str(target_html))

        mlflow.log_artifact(str(data_html))
        mlflow.log_artifact(str(target_html))
        print("[drift] saved & logged HTML reports.")


if __name__ == "__main__":
    main()
