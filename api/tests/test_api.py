import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

VALID_CAR = {
    "mileage": 140000,
    "engine_power": 130,
    "model_key": "Citroën",
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": False,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": False,
    "winter_tires": True,
}


@pytest.fixture(scope="module")
def client():
    mock = MagicMock()
    mock.predict.side_effect = lambda df: np.full(len(df), 120.0)
    with patch("joblib.load", return_value=mock):
        if "main" in sys.modules:
            del sys.modules["main"]
        from main import app
        yield TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_valid(client):
    response = client.post("/predict", json={"input": [VALID_CAR]})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert len(data["prediction"]) == 1
    assert isinstance(data["prediction"][0], float)


def test_predict_batch(client):
    response = client.post("/predict", json={"input": [VALID_CAR, VALID_CAR]})
    assert response.status_code == 200
    assert len(response.json()["prediction"]) == 2


def test_predict_negative_mileage(client):
    response = client.post("/predict", json={"input": [{**VALID_CAR, "mileage": -1}]})
    assert response.status_code == 422


def test_predict_zero_engine_power(client):
    response = client.post("/predict", json={"input": [{**VALID_CAR, "engine_power": 0}]})
    assert response.status_code == 422


def test_predict_invalid_fuel(client):
    response = client.post("/predict", json={"input": [{**VALID_CAR, "fuel": "nuclear"}]})
    assert response.status_code == 422


def test_predict_invalid_car_type(client):
    response = client.post("/predict", json={"input": [{**VALID_CAR, "car_type": "spaceship"}]})
    assert response.status_code == 422


def test_predict_missing_field(client):
    car = {k: v for k, v in VALID_CAR.items() if k != "fuel"}
    response = client.post("/predict", json={"input": [car]})
    assert response.status_code == 422


def test_predict_wrong_type(client):
    response = client.post("/predict", json={"input": [{**VALID_CAR, "mileage": "not_a_number"}]})
    assert response.status_code == 422


def test_predict_empty_input(client):
    response = client.post("/predict", json={"input": []})
    assert response.status_code == 200
    assert response.json()["prediction"] == []
