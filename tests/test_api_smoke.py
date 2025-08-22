import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serve.app import app


def test_api_predict_smoke():
    """Test API endpoints without full training pipeline."""
    client = TestClient(app)

    # Test health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200

    # Test predict endpoint (will fail gracefully if no model)
    payload = {
        "trip_distance": 2.5,
        "passenger_count": 1,
        "PULocationID": 10,
        "DOLocationID": 30,
        "hour": 0,
        "day_of_week": 2,
        "payment_type": 1,
    }
    resp = client.post("/predict", json=payload)
    # Should return 503 (no model) or 200 (if model exists)
    assert resp.status_code in [200, 503]


def test_model_prediction_end_to_end():
    """Test model prediction using the latest trained model directly from MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.config import get_tracking_uri, load_config

    cfg = load_config()
    mlflow.set_tracking_uri(get_tracking_uri(cfg))
    client = MlflowClient()

    # Get the latest run with a model
    exp = client.get_experiment_by_name(cfg.mlflow["experiment"])
    runs = client.search_runs(
        exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=5
    )

    # Find a run with a model artifact
    model_run = None
    for run in runs:
        try:
            artifacts = client.list_artifacts(run.info.run_id)
            if any(art.path == "model" for art in artifacts):
                model_run = run
                break
        except Exception:
            continue

    if not model_run:
        pytest.skip("No trained model found in MLflow runs")

    # Load model directly from the run
    model_uri = f"runs:/{model_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    # Test various scenarios
    test_scenarios = [
        {
            "trip_distance": 5.2,
            "passenger_count": 2,
            "PULocationID": 161,
            "DOLocationID": 237,
            "hour": 14,
            "day_of_week": 2,
            "payment_type": 1,
            "expected_range": (15, 35),  # Expected duration range in minutes
        },
        {
            "trip_distance": 1.2,
            "passenger_count": 1,
            "PULocationID": 50,
            "DOLocationID": 75,
            "hour": 8,
            "day_of_week": 1,
            "payment_type": 1,
            "expected_range": (3, 10),
        },
        {
            "trip_distance": 10.5,
            "passenger_count": 4,
            "PULocationID": 161,
            "DOLocationID": 237,
            "hour": 18,
            "day_of_week": 5,
            "payment_type": 2,
            "expected_range": (25, 50),
        },
    ]

    for i, scenario in enumerate(test_scenarios):
        # Extract expected range and test data
        expected_range = scenario.pop("expected_range")
        test_data = pd.DataFrame([scenario])

        # Convert data types to match model schema expectations
        test_data = test_data.astype(
            {
                "trip_distance": "float64",
                "passenger_count": "float64",
                "PULocationID": "int32",
                "DOLocationID": "int32",
                "hour": "int32",
                "day_of_week": "int32",
                "payment_type": "float64",
            }
        )

        # Make prediction
        prediction = model.predict(test_data)
        pred_value = float(prediction[0])

        # Validate prediction is reasonable
        assert expected_range[0] <= pred_value <= expected_range[1], (
            f"Scenario {i+1}: Prediction {pred_value:.2f} outside expected range "
            f"{expected_range} for trip distance {scenario['trip_distance']}mi"
        )

        # Validate prediction is positive
        assert pred_value > 0, f"Scenario {i+1}: Prediction should be positive"

        print(f"✓ Scenario {i+1}: {scenario['trip_distance']}mi → {pred_value:.2f} min")


def test_api_input_validation():
    """Test API input validation with invalid data."""
    client = TestClient(app)

    # Test missing required fields
    invalid_payloads = [
        {},  # Empty payload
        {"trip_distance": -1},  # Negative distance
        {"trip_distance": 5, "hour": 25},  # Invalid hour
        {"trip_distance": 5, "day_of_week": 7},  # Invalid day
        {"trip_distance": 5, "passenger_count": -1},  # Negative passengers
    ]

    for payload in invalid_payloads:
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422, f"Should reject invalid payload: {payload}"


def test_api_endpoints_exist():
    """Test that all expected API endpoints exist."""
    client = TestClient(app)

    # Test health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200

    # Test model info endpoint (may return 404 if no model)
    resp = client.get("/model")
    assert resp.status_code in [200, 404]

    # Test reload endpoint (may return 503 if no model)
    resp = client.post("/reload")
    assert resp.status_code in [200, 503]
