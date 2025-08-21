# Drift Plan

We detect:
- **Data drift**: feature distribution shifts (distance, hour, passenger_count, locations).
- **Target/concept drift**: relationship between features and duration.

**Simulation** (`src/data/simulate_drift.py`):
- Add Gaussian noise to `trip_distance` and `duration_min`.
- Skew `hour` to nighttime.
- Randomly remap a fraction of `PULocationID`/`DOLocationID`.

**Reporting** (`src/monitoring/generate_drift.py`):
- Use Evidently's `DataDriftPreset` and `TargetDriftPreset`.
- Save HTML reports; log to MLflow as artifacts.
