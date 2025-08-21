from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import mlflow
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import load_config, get_tracking_uri

app = FastAPI(title="MLOps Final — Model API")

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
        return True
    except Exception:
        # Fallback to local artifacts (optional): not needed if registry exists
        _model = None
        return False

@app.on_event("startup")
def startup_event():
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
    try:
        y = _model.predict(df)
        val = float(y[0]) if hasattr(y, "__len__") else float(y)
        return {"prediction": val}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model")
def model_info():
    mlflow.set_tracking_uri(get_tracking_uri(_cfg))
    client = MlflowClient()
    name = _cfg.mlflow["model_name"]
    try:
        latest = client.get_latest_versions(name=name, stages=["Production"])[0]
    except Exception:
        raise HTTPException(status_code=404, detail="No production model")
    run = client.get_run(latest.run_id)
    params = run.data.params
    metrics = run.data.metrics
    # infer schema from Pydantic model
    schema = {f: str(t.annotation) for f, t in InputData.model_fields.items()}
    # simple "important features" via model attribute names (if sklearn pipeline)
    important = []
    try:
        import sklearn
        # Try to extract feature names after OHE
        # If not available, return input schema keys
        important = list(schema.keys())
    except Exception:
        important = list(schema.keys())
    return {
        "model_name": name,
        "run_id": latest.run_id,
        "version": latest.version,
        "params": params,
        "metrics": metrics,
        "input_schema": schema,
        "important_features": important[:5],
    }
