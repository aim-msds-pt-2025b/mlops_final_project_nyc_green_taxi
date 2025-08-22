import logging
import os
from dataclasses import dataclass, field
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
    logging: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str | None = None) -> Config:
    cfg_path = path or os.environ.get("CONFIG_PATH", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def get_tracking_uri(cfg: Config) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", f"file:{cfg.paths['mlruns_dir']}")


def get_log_level(cfg: Config | None = None) -> int:
    """Resolve log level from env LOG_LEVEL, fallback to cfg.logging.level, else DEBUG."""
    env_level = os.environ.get("LOG_LEVEL")
    level_str = None
    if env_level:
        level_str = env_level
    elif cfg and isinstance(cfg.logging, dict):
        level_str = cfg.logging.get("level")
    if not level_str:
        return logging.DEBUG
    try:
        return getattr(logging, level_str.upper())
    except AttributeError:
        return logging.DEBUG
