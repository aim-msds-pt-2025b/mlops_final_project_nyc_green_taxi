import pandas as pd
from src.features.transform import compute_duration, engineer
from src.config import load_config

def test_engineer_ranges():
    cfg = load_config()
    df = pd.DataFrame({
        'lpep_pickup_datetime': ['2024-01-01 12:00:00','2024-01-01 13:00:00'],
        'lpep_dropoff_datetime': ['2024-01-01 12:30:00','2024-01-01 14:00:00'],
        'trip_distance':[1.2, 3.4],
        'passenger_count':[1,2],
        'PULocationID':[1,2],
        'DOLocationID':[3,4],
        'payment_type':[1,1],
    })
    out = engineer(df, cfg)
    assert out['duration_min'].between(cfg.features['min_duration_min'], cfg.features['max_duration_min']).all()
    assert (out['trip_distance'] >= 0).all()
    assert out['hour'].between(0,23).all()
    assert out['day_of_week'].between(0,6).all()
