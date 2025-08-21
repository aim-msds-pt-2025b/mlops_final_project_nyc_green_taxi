import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.logging_utils import setup_logging


def main():
    cfg = load_config()
    setup_logging(cfg)
    ref = Path(cfg.paths["reference_path"])
    if not ref.exists():
        raise FileNotFoundError("Reference not found. Run: python -m src.features.transform")
    df = pd.read_parquet(ref)

    rng = np.random.default_rng(cfg.random_state + 7)
    drift = df.copy()
    if "trip_distance" in drift:
        drift["trip_distance"] = (drift["trip_distance"] * rng.normal(1.2, 0.1, len(drift))).clip(
            lower=0
        )
    if "hour" in drift:
        # bias to late-night hours
        late_hours = np.concatenate([np.arange(20, 24), np.arange(0, 6)])
        drift["hour"] = rng.choice(late_hours, size=len(drift))
    if "passenger_count" in drift:
        drift["passenger_count"] = (
            drift["passenger_count"].fillna(1) + rng.integers(-1, 2, len(drift))
        ).clip(0, 6)
    if "duration_min" in drift:
        drift["duration_min"] = (drift["duration_min"] * rng.normal(1.1, 0.15, len(drift))).clip(
            1, 180
        )

    current_dir = Path(cfg.paths["current_dir"])
    current_dir.mkdir(parents=True, exist_ok=True)
    out = current_dir / "current.parquet"
    drift.to_parquet(out, index=False)
    logging.getLogger(__name__).info(
        "wrote simulated current batch", extra={"path": str(out), "rows": len(drift)}
    )
    return str(out)


if __name__ == "__main__":
    main()
