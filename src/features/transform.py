import pandas as pd
from pathlib import Path
from src.config import load_config

TARGET = "duration_min"

def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pick = pd.to_datetime(df["lpep_pickup_datetime"])
    drop = pd.to_datetime(df["lpep_dropoff_datetime"])
    df[TARGET] = (drop - pick).dt.total_seconds() / 60.0
    return df

def engineer(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = compute_duration(df)
    df = df[(df[TARGET] >= cfg.features["min_duration_min"]) & (df[TARGET] <= cfg.features["max_duration_min"])]
    df = df[df.get("trip_distance", 0) >= 0]
    dt = pd.to_datetime(df["lpep_pickup_datetime"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    cols = ["trip_distance","passenger_count","PULocationID","DOLocationID","payment_type","hour","day_of_week",TARGET]
    return df[[c for c in cols if c in df.columns]]

def main():
    cfg = load_config()
    raw_dir = Path(cfg.paths["raw_dir"])
    files = list(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No raw parquet found. Run: python -m src.data.get_data")
    df = pd.read_parquet(files[0])
    # optional downsample
    frac = float(cfg.data.get("sample_fraction", 1.0))
    if 0 < frac < 1.0:
        df = df.sample(frac=frac, random_state=cfg.random_state)
    out_path = Path(cfg.paths["features_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = engineer(df, cfg)
    features.to_parquet(out_path, index=False)
    # Save reference for drift
    Path(cfg.paths["reference_path"]).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(cfg.paths["reference_path"], index=False)
    print(f"[transform] Wrote features to {out_path} and reference to {cfg.paths['reference_path']}")
    return str(out_path)

if __name__ == "__main__":
    main()
