"""
Model training script for Getaround pricing prediction.
This script loads the dataset, preprocesses the data, trains a Random Forest model,
evaluates it, and logs the results to MLflow.
"""

import os
import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/get_around_pricing_project.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

    # Define target and features
    TARGET = "rental_price_per_day"
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Convert mileage and engine_power to float for mlflow compatibility
    X["mileage"] = X["mileage"].astype(float)
    X["engine_power"] = X["engine_power"].astype(float)

    # Define feature groups
    numeric_features = ["mileage", "engine_power"]
    categorical_features = ["model_key", "fuel", "paint_color", "car_type"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    # Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow experiment
    mlflow.set_tracking_uri("file:///app/mlruns")
    mlflow.set_experiment("getaround_pricing")

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log model with input example and signature
        signature = infer_signature(X_test, y_pred)
        input_example = X_test.iloc[:2].to_dict(orient="records")

        # Log model to MLflow tracking server
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="getaround_model",
            input_example=input_example,
            signature=signature,
        )

        # Save model locally
        os.makedirs("model", exist_ok=True)
        joblib.dump(pipeline, "model/model.joblib")

        print("Model trained and saved successfully")
        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")
