import json
import logging
import os
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

        # SHAP summary (enabled for all platforms)
        enable_shap = os.environ.get("ENABLE_SHAP", "1") == "1"
        if enable_shap:
            try:
                log.info("Generating SHAP summary plot...")

                # Use a smaller sample for SHAP calculation
                sample_size = min(100, len(X_train))
                background = X_train.sample(n=sample_size, random_state=cfg.random_state)
                log.info(f"Using {sample_size} samples for SHAP calculation")

                # Transform the sample through the preprocessing pipeline
                transformed = pipe.named_steps["prep"].transform(background)
                log.info(f"Transformed sample shape: {transformed.shape}")

                # Access fitted RF model
                rf = pipe.named_steps["model"]
                log.info(f"Model type: {type(rf)}")

                # Create explainer and compute SHAP values
                explainer = shap.TreeExplainer(rf)
                log.info("TreeExplainer created successfully")

                shap_values = explainer.shap_values(transformed)
                log.info(f"SHAP values computed, shape: {shap_values.shape}")

                # Use matplotlib backend that works on Windows
                import matplotlib

                matplotlib.use("Agg")
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, transformed, show=False, plot_type="bar")
                shap_path = "reports/shap_summary.png"
                os.makedirs("reports", exist_ok=True)
                plt.tight_layout()
                plt.savefig(shap_path, bbox_inches="tight", dpi=100)
                plt.close()
                mlflow.log_artifact(shap_path)
                log.info("SHAP summary plot generated and logged")
            except Exception as e:
                log.warning(f"SHAP generation failed: {str(e)}")
                log.warning(f"Error type: {type(e).__name__}")
                import traceback

                log.warning(f"Full traceback: {traceback.format_exc()}")
                os.makedirs("reports", exist_ok=True)
                with open("reports/shap_error.txt", "w") as f:
                    f.write(f"SHAP error: {str(e)}\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    f.write(f"Traceback:\n{traceback.format_exc()}")
                mlflow.log_artifact("reports/shap_error.txt")
        else:
            log.info("SHAP disabled via ENABLE_SHAP=0")

        # Prepare input example and signature for MLflow model logging
        input_example = X_train.head(3)  # Use first 3 rows as example

        # Log full pipeline as MLflow model (Registry-ready)
        mlflow_sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=None,  # registration handled by deployment stage
        )

    log.info("training metrics", extra=metrics)
    log.info("mlflow run", extra={"run_id": run.info.run_id})


if __name__ == "__main__":
    main()
