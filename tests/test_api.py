"""FastAPI prediction API: routes and input validation (model mocked), plus the saved model."""

import sys
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

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


def test_model_path_default_is_repo_model(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    from main import REPO_ROOT, model_path

    assert model_path() == REPO_ROOT / "ml" / "model" / "model.joblib"


def test_model_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "other.joblib"))
    from main import model_path

    assert model_path() == tmp_path / "other.joblib"


def test_saved_model_predicts_plausible_price():
    """The committed pipeline loads and prices a typical car within the dataset range."""
    from main import REPO_ROOT

    pipeline = joblib.load(REPO_ROOT / "ml" / "model" / "model.joblib")
    price = pipeline.predict(pd.DataFrame([VALID_CAR]))[0]
    assert 10 <= price <= 422
