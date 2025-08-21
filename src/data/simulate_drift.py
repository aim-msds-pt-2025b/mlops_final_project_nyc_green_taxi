import pandas as pd, numpy as np
from pathlib import Path
from src.config import load_config

def main():
    cfg = load_config()
    ref = Path(cfg.paths["reference_path"])
    if not ref.exists():
        raise FileNotFoundError("Reference not found. Run: python -m src.features.transform")
    df = pd.read_parquet(ref)

    rng = np.random.default_rng(cfg.random_state + 7)
    drift = df.copy()
    if "trip_distance" in drift:
        drift["trip_distance"] = (drift["trip_distance"] * rng.normal(1.2, 0.1, len(drift))).clip(lower=0)
    if "hour" in drift:
        # bias to late-night hours
        drift["hour"] = rng.choice(np.concatenate([np.arange(20,24), np.arange(0,6)]), size=len(drift))
    if "passenger_count" in drift:
        drift["passenger_count"] = (drift["passenger_count"].fillna(1) + rng.integers(-1, 2, len(drift))).clip(0, 6)
    if "duration_min" in drift:
        drift["duration_min"] = (drift["duration_min"] * rng.normal(1.1, 0.15, len(drift))).clip(1, 180)

    current_dir = Path(cfg.paths["current_dir"])
    current_dir.mkdir(parents=True, exist_ok=True)
    out = current_dir / "current.parquet"
    drift.to_parquet(out, index=False)
    print(f"[drift] Wrote simulated current batch: {out}")
    return str(out)

if __name__ == "__main__":
    main()
