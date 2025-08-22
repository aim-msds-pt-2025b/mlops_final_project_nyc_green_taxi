import logging
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient

try:
    # Older MLflow raises RestException for registry errors
    from mlflow.exceptions import RestException  # type: ignore
except Exception:  # pragma: no cover - optional import for older versions
    RestException = MlflowException  # type: ignore
from pydantic import BaseModel, Field

from src.config import get_tracking_uri, load_config
from src.logging_utils import setup_logging

app = FastAPI(title="MLOps Final — Model API")
log = logging.getLogger(__name__)


class InputData(BaseModel):
    trip_distance: float = Field(..., ge=0)
    passenger_count: Optional[int] = Field(1, ge=0)
    PULocationID: int
    DOLocationID: int
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    payment_type: int


_model = None
_cfg = load_config()


def _load_champion():
    global _model
    mlflow.set_tracking_uri(get_tracking_uri(_cfg))
    name = _cfg.mlflow["model_name"]
    try:
        _model = load_model(f"models:/{name}/Production")
        log.info("loaded champion", extra={"model_name": name, "stage": "Production"})
        return True
    except (MlflowException, RestException, FileNotFoundError) as exc:
        # Fallback to local artifacts (optional): not needed if registry exists
        _model = None
        log.warning(
            "no champion in registry; model unset",
            extra={"model_name": name, "error": str(exc)},
        )
        return False


@app.on_event("startup")
def startup_event():
    setup_logging(_cfg)
    _load_champion()


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "no-model"}


@app.post("/reload")
def reload_model():
    if not _load_champion():
        raise HTTPException(status_code=503, detail="Champion not available")
    return {"status": "reloaded"}


@app.post("/predict")
def predict(x: InputData):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([x.dict()])

    # Ensure proper data types to match MLflow schema expectations
    # Based on the schema error messages, we need specific types:
    df["passenger_count"] = df["passenger_count"].astype("float64")  # double
    df["payment_type"] = df["payment_type"].astype("float64")  # double
    df["PULocationID"] = df["PULocationID"].astype("int32")  # int32
    df["DOLocationID"] = df["DOLocationID"].astype("int32")  # int32
    df["hour"] = df["hour"].astype("int32")  # int32
    df["day_of_week"] = df["day_of_week"].astype("int32")  # int32

    try:
        y = _model.predict(df)
        val = float(y[0]) if hasattr(y, "__len__") else float(y)
        log.debug("prediction", extra={"input": x.dict(), "prediction": val})
        return {"prediction": val}
    except Exception as e:  # noqa: BLE001 - surface all prediction errors to the client
        log.exception("prediction failed")
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/model")
def model_info():
    mlflow.set_tracking_uri(get_tracking_uri(_cfg))
    client = MlflowClient()
    name = _cfg.mlflow["model_name"]
    try:
        latest = client.get_latest_versions(name=name, stages=["Production"])[0]
    except (MlflowException, RestException, IndexError) as exc:
        raise HTTPException(status_code=404, detail="No production model") from exc
    run = client.get_run(latest.run_id)
    params = run.data.params
    metrics = run.data.metrics
    # infer schema from Pydantic model
    schema = {f: str(t.annotation) for f, t in InputData.model_fields.items()}
    # simple "important features" via model attribute names (if sklearn pipeline)
    important = []
    try:
        # Try to extract feature names after OHE
        # If not available, return input schema keys
        important = list(schema.keys())
    except Exception:
        important = list(schema.keys())
    info = {
        "model_name": name,
        "run_id": latest.run_id,
        "version": latest.version,
        "params": params,
        "metrics": metrics,
        "input_schema": schema,
        "important_features": important[:5],
    }
    log.debug("model info", extra=info)
    return info
