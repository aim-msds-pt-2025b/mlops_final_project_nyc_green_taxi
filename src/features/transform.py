import logging
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.logging_utils import setup_logging

TARGET = "duration_min"
log = logging.getLogger(__name__)


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pick = pd.to_datetime(df["lpep_pickup_datetime"])
    drop = pd.to_datetime(df["lpep_dropoff_datetime"])
    df[TARGET] = (drop - pick).dt.total_seconds() / 60.0
    return df


def engineer(df: pd.DataFrame, cfg) -> pd.DataFrame:
    log.debug("engineering start", extra={"rows": len(df), "cols": list(df.columns)})
    df = compute_duration(df)
    min_dur = cfg.features["min_duration_min"]
    max_dur = cfg.features["max_duration_min"]
    df = df[(df[TARGET] >= min_dur) & (df[TARGET] <= max_dur)]
    if "trip_distance" in df.columns:
        mask = df["trip_distance"] >= 0
        df = df[mask]
    dt = pd.to_datetime(df["lpep_pickup_datetime"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    cols = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "hour",
        "day_of_week",
        TARGET,
    ]
    return df[[c for c in cols if c in df.columns]]


def main():
    cfg = load_config()
    setup_logging(cfg)
    raw_dir = Path(cfg.paths["raw_dir"])
    files = list(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No raw parquet found. Run: python -m src.data.get_data")
    log.info("reading raw parquet", extra={"path": str(files[0])})
    df = pd.read_parquet(files[0])
    # optional downsample
    frac = float(cfg.data.get("sample_fraction", 1.0))
    if 0 < frac < 1.0:
        log.info("downsampling", extra={"frac": frac})
        df = df.sample(frac=frac, random_state=cfg.random_state)
    out_path = Path(cfg.paths["features_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = engineer(df, cfg)
    features.to_parquet(out_path, index=False)
    # Save reference for drift
    Path(cfg.paths["reference_path"]).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cfg.paths["reference_path"], index=False)
    ref_path = cfg.paths["reference_path"]
    log.info(
        "wrote features and reference",
        extra={
            "features_path": str(out_path),
            "reference_path": str(ref_path),
            "rows": len(features),
        },
    )
    return str(out_path)


if __name__ == "__main__":
    main()
