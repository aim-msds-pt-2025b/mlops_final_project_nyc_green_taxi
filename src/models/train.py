import json
import logging
import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
from mlflow import sklearn as mlflow_sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import get_tracking_uri, load_config
from src.logging_utils import setup_logging

TARGET = "duration_min"
log = logging.getLogger(__name__)


def load_features(cfg):
    p = Path(cfg.paths["features_out"])
    if not p.exists():
        error_msg = "Processed features not found. Run: python -m src.features.transform"
        raise FileNotFoundError(error_msg)
    log.info("loading features", extra={"path": str(p)})
    return pd.read_parquet(p)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if X[c].dtype != "object" and c != TARGET]
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in ["PULocationID", "DOLocationID", "payment_type"]:
        if c in num_cols:
            num_cols.remove(c)
        if c not in cat_cols and c in X.columns:
            cat_cols.append(c)
    pre = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    return pre


def main():
    cfg = load_config()
    setup_logging(cfg)
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    mlflow.set_experiment(cfg.mlflow["experiment"])

    df = load_features(cfg)
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])
    log.info("split dataset", extra={"rows": len(df), "target": TARGET})
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=cfg.random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=cfg.random_state
    )

    pre = build_preprocessor(X_train)
    hp = cfg.model["hyperparams"]
    reg = RandomForestRegressor(random_state=cfg.random_state, n_jobs=cfg.n_jobs, **hp)
    pipe = Pipeline([("prep", pre), ("model", reg)])

    with mlflow.start_run(run_name="train-rf") as run:
        # Train
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_val)
        pred_test = pipe.predict(X_test)

        metrics = {
            "mae_val": float(mean_absolute_error(y_val, pred_val)),
            "mae_test": float(mean_absolute_error(y_test, pred_test)),
            "r2_val": float(r2_score(y_val, pred_val)),
            "r2_test": float(r2_score(y_test, pred_test)),
        }

        # Log params & metrics
        for k, v in hp.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("mae_val", metrics["mae_val"])
        mlflow.log_metric("mae_test", metrics["mae_test"])
        mlflow.log_metric("r2_val", metrics["r2_val"])
        mlflow.log_metric("r2_test", metrics["r2_test"])

        # Save reports/metrics.json
        os.makedirs("reports", exist_ok=True)
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("reports/metrics.json")

        # SHAP summary (optional; disabled on Windows due to instability)
        enable_shap = os.environ.get("ENABLE_SHAP", "1") == "1"
        if enable_shap and platform.system() != "Windows":
            try:
                background = X_train.sample(n=min(500, len(X_train)), random_state=cfg.random_state)
                # Build a fitted pipeline transformer for SHAP input
                transformed = pipe.named_steps["prep"].fit_transform(background)
                # Access fitted RF
                rf = pipe.named_steps["model"]
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(transformed)
                plt.figure(figsize=(8, 4))
                shap.summary_plot(shap_values, transformed, show=False)
                shap_path = "reports/shap_summary.png"
                plt.tight_layout()
                plt.savefig(shap_path, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(shap_path)
            except Exception as e:
                with open("reports/shap_error.txt", "w") as f:
                    f.write(str(e))
                mlflow.log_artifact("reports/shap_error.txt")
        else:
            log.info(
                "SHAP disabled",
                extra={
                    "platform": platform.system(),
                    "ENABLE_SHAP": enable_shap,
                },
            )

        # Log full pipeline as MLflow model (Registry-ready)
        mlflow_sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=None,  # registration handled by deployment stage
        )

    log.info("training metrics", extra=metrics)
    log.info("mlflow run", extra={"run_id": run.info.run_id})


if __name__ == "__main__":
    main()
