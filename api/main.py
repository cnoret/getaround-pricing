"""
Getaround Price Prediction API
This API provides a model for predicting optimal rental prices based on car features.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd


class CarFeatures(BaseModel):
    """Model representing the features of a car for price prediction."""

    mileage: float
    engine_power: float
    model_key: str
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool


class InputData(BaseModel):
    """Model for the input data to the prediction endpoint."""

    input: list[CarFeatures]


app = FastAPI(
    title="Getaround Price Prediction API",
    description="API to predict rental prices for cars based on features",
    version="1.0",
)

# Load the trained model once
model = joblib.load("model/model.joblib")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint that returns a stylish HTML welcome page."""
    html_content = """
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Getaround Price Prediction API</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: 'Segoe UI', Arial, sans-serif;
                    background: linear-gradient(135deg, #e8f0fe 0%, #f7fafd 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }
                .container {
                    background: white;
                    padding: 36px 44px;
                    border-radius: 18px;
                    box-shadow: 0 12px 36px rgba(44, 62, 80, 0.10);
                    text-align: center;
                    animation: fadein 1.2s;
                }
                @keyframes fadein {
                    from { opacity: 0; transform: translateY(30px);}
                    to { opacity: 1; transform: translateY(0);}
                }
                .logo {
                    width: 68px;
                    height: 68px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #2980b9 60%, #1abc9c 100%);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin: 0 auto 22px auto;
                    box-shadow: 0 4px 16px rgba(41, 128, 185, 0.13);
                }
                .logo-text {
                    color: #fff;
                    font-size: 2.2rem;
                    font-weight: bold;
                    letter-spacing: 2px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }
                h1 {
                    color: #2c3e50;
                    font-size: 2.2rem;
                    margin-bottom: 16px;
                }
                p {
                    color: #555;
                    font-size: 1.13rem;
                    margin-bottom: 16px;
                }
                .docs-link {
                    display: inline-block;
                    margin-top: 14px;
                    padding: 12px 26px;
                    background: linear-gradient(135deg, #2980b9 40%, #1abc9c 100%);
                    color: #fff;
                    border-radius: 8px;
                    font-weight: 600;
                    text-decoration: none;
                    transition: background 0.3s, transform 0.2s;
                    box-shadow: 0 2px 8px rgba(41, 128, 185, 0.12);
                }
                .docs-link:hover {
                    background: linear-gradient(135deg, #1abc9c 60%, #2980b9 100%);
                    transform: translateY(-2px) scale(1.04);
                }
                @media (max-width: 600px) {
                    .container { padding: 16px 4vw; }
                    h1 { font-size: 1.4rem; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="logo">
                    <span class="logo-text">GA</span>
                </div>
                <h1>Welcome to Getaround Price Prediction API</h1>
                <p>
                    🚗 Predict optimal car rental prices with our fast, secure, and scalable API.<br>
                    Just send your car’s features and get your price instantly!
                </p>
                <a class="docs-link" href="/docs">API Documentation (Swagger UI)</a>
            </div>
        </body>
    </html>
    """
    return html_content


@app.post("/predict")
def predict(data: InputData):
    """Endpoint to predict rental prices based on car features."""
    # Convert input list of CarFeatures to a pandas DataFrame
    input_dicts = [item.model_dump() for item in data.input]
    df = pd.DataFrame(input_dicts)
    # Predict using the loaded model pipeline
    predictions = model.predict(df)
    # Return predictions as a list
    return {"prediction": predictions.tolist()}


