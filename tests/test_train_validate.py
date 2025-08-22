import pandas as pd

from src.config import load_config
from src.features.transform import engineer


def test_feature_engineering():
    """Test feature engineering function with minimal data."""
    cfg = load_config()
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2020-01-01 00:00:00", "2020-01-01 01:00:00"],
            "lpep_dropoff_datetime": ["2020-01-01 00:10:00", "2020-01-01 01:07:00"],
            "trip_distance": [1.2, 3.4],
            "passenger_count": [1, 2],
            "PULocationID": [10, 20],
            "DOLocationID": [30, 40],
            "payment_type": [1, 2],
        }
    )

    features = engineer(df, cfg)

    # Check expected columns exist
    expected_cols = [
        "trip_distance",
        "passenger_count",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "hour",
        "day_of_week",
        "duration_min",
    ]
    for col in expected_cols:
        assert col in features.columns

    # Check duration is reasonable
    assert features["duration_min"].min() >= 1
    assert features["duration_min"].max() <= 120
