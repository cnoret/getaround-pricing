"""The Streamlit dashboard runs without exception for every filter and in the prediction form."""

from unittest.mock import MagicMock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT, TIMEOUT, no_exception

APP = str(ROOT / "dashboard" / "app.py")


def run_app():
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def metric(at, label):
    return next(m for m in at.metric if m.label == label)


def test_app_starts():
    at = run_app()
    no_exception(at)
    assert not at.error
    assert len(at.tabs) == 3


def test_headline_metrics():
    at = run_app()
    assert metric(at, "Conflicts avoided").value == "70%"
    assert metric(at, "Rentals blocked (revenue at risk)").value == "2.0%"
    assert metric(at, "Cancellations avoided").value == "19 of 37"


@pytest.mark.parametrize("scope", ["All", "Connect only"])
@pytest.mark.parametrize("threshold", [0, 60, 180])
def test_delay_filters(scope, threshold):
    at = run_app()
    at.sidebar.slider[0].set_value(threshold)
    at.sidebar.radio[0].set_value(scope)
    at.run()
    no_exception(at)
    assert not at.error


def test_prediction_with_local_model(monkeypatch):
    """Without API_URL (Streamlit Community Cloud), the dashboard predicts with the saved model."""
    monkeypatch.delenv("API_URL", raising=False)
    with patch("requests.post") as post:
        at = run_app()
        at.button[0].click().run()
    no_exception(at)
    post.assert_not_called()
    assert any("/ day" in m.body for m in at.markdown)


def test_prediction_through_api(monkeypatch):
    monkeypatch.setenv("API_URL", "http://api:8001/predict")
    response = MagicMock(status_code=200)
    response.json.return_value = {"prediction": [135.4]}
    with patch("requests.post", return_value=response) as post:
        at = run_app()
        at.button[0].click().run()
    no_exception(at)
    post.assert_called_once()
    assert post.call_args.args[0] == "http://api:8001/predict"
    payload = post.call_args.kwargs["json"]["input"][0]
    assert payload["mileage"] == 140_000.0
    assert any("135 €" in m.body for m in at.markdown)


def test_prediction_falls_back_when_api_down(monkeypatch):
    """A paused or unreachable API does not break the form: the saved model answers instead."""
    monkeypatch.setenv("API_URL", "http://api:8001/predict")
    with patch("requests.post", side_effect=requests.ConnectionError("refused")) as post:
        at = run_app()
        at.button[0].click().run()
    no_exception(at)
    post.assert_called_once()
    assert not at.error
    assert any("did not answer" in c.value for c in at.caption)
    assert any("/ day" in m.body for m in at.markdown)
