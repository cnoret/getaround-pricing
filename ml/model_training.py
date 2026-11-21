"""
Train the Getaround daily price model.

Loads the car listings, drops impossible rows, cross-validates a gradient
boosting pipeline on 80% of the data, scores it once on the 20% holdout against
a median-price baseline, logs everything to MLflow and saves the pipeline to
ml/model/model.joblib (the file the API and the dashboard load).

Usage, from anywhere: python ml/model_training.py
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "get_around_pricing_project.csv"
MODEL_PATH = REPO_ROOT / "ml" / "model" / "model.joblib"

TARGET = "rental_price_per_day"
NUMERIC_FEATURES = ["mileage", "engine_power"]
CATEGORICAL_FEATURES = ["model_key", "fuel", "paint_color", "car_type"]
SEED = 42


def load_data(path=DATA_PATH):
    """Car listings without the index column and without impossible values."""
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
    # One listing has a negative mileage and one an engine power of 0
    df = df[(df["mileage"] >= 0) & (df["engine_power"] > 0)]
    # Floats, so the MLflow signature accepts decimal inputs
    return df.astype({"mileage": float, "engine_power": float})


def build_pipeline():
    """One-hot encoding (brands seen fewer than 5 times are grouped) + gradient boosting."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=5,
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        # Numeric and boolean features go to the trees as they are
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, random_state=SEED
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def train(df):
    """Cross-validate on the training split, then fit and score on the holdout."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    pipeline = build_pipeline()
    cv = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="neg_mean_absolute_error",
    )
    pipeline.fit(X_train, y_train)

    holdout = evaluate(y_test, pipeline.predict(X_test))
    metrics = {
        "cv_mae": -cv["test_score"].mean(),
        "cv_mae_std": cv["test_score"].std(),
        **holdout,
        "baseline_mae": mean_absolute_error(
            y_test, np.full(len(y_test), y_train.median())
        ),
    }

    # Share of the holdout error increase when each feature is shuffled
    importance = permutation_importance(
        pipeline, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=10, random_state=SEED,
    ).importances_mean
    importance = pd.Series(importance.clip(min=0), index=X.columns)
    importance = (importance / importance.sum()).sort_values(ascending=False)

    return pipeline, metrics, importance, X_test


def main():
    import mlflow
    from mlflow.models.signature import infer_signature

    df = load_data()
    pipeline, metrics, importance, X_test = train(df)

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", (REPO_ROOT / "mlruns").as_uri())
    )
    mlflow.set_experiment("getaround_pricing")
    with mlflow.start_run():
        mlflow.log_params(pipeline.named_steps["model"].get_params())
        mlflow.log_param("rows", len(df))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="getaround_model",
            input_example=X_test.iloc[:2],
            signature=infer_signature(X_test, pipeline.predict(X_test)),
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Trained on {len(df):,} listings, saved to {MODEL_PATH.relative_to(REPO_ROOT)}")
    print(f"5-fold CV MAE: {metrics['cv_mae']:.2f} ± {metrics['cv_mae_std']:.2f}")
    print(
        f"Holdout MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | "
        f"R²: {metrics['r2']:.3f} | median baseline MAE: {metrics['baseline_mae']:.2f}"
    )
    print("Permutation importance:")
    print(importance.map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
