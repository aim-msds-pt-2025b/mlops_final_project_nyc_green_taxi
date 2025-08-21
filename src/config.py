import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    random_state: int
    n_jobs: int
    paths: Dict[str, Any]
    data: Dict[str, Any]
    features: Dict[str, Any]
    model: Dict[str, Any]
    validation_thresholds: Dict[str, Any]
    mlflow: Dict[str, Any]


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def get_tracking_uri(cfg: Config) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"file:{cfg.paths['mlruns_dir']}")
