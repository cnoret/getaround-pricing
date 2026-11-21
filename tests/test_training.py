"""Training pipeline: data cleaning, a quick fit on a sample, and the committed model."""

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.model_training import MODEL_PATH, TARGET, build_pipeline, load_data, train


def test_load_data_drops_impossible_listings():
    df = load_data()
    assert len(df) == 4841
    assert (df["mileage"] >= 0).all()
    assert (df["engine_power"] > 0).all()
    assert "Unnamed: 0" not in df.columns


def test_pipeline_handles_unseen_brand():
    df = load_data().sample(600, random_state=0)
    pipeline = build_pipeline().fit(df.drop(columns=[TARGET]), df[TARGET])
    car = df.drop(columns=[TARGET]).iloc[[0]].assign(model_key="Unknown brand")
    assert np.isfinite(pipeline.predict(car)).all()


def test_train_beats_median_baseline():
    df = load_data().sample(1500, random_state=0)
    _, metrics, importance, _ = train(df)
    assert metrics["mae"] < metrics["baseline_mae"]
    assert abs(importance.sum() - 1) < 1e-9


def test_committed_model_is_gradient_boosting():
    pipeline = joblib.load(MODEL_PATH)
    assert isinstance(pipeline.named_steps["model"], HistGradientBoostingRegressor)
