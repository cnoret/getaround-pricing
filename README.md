# Getaround — Price Prediction Dashboard & API

[![Live Dashboard](https://img.shields.io/badge/Live%20Dashboard-Open-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://cnoret-getaround-dashboard.hf.space/)
[![Live API](https://img.shields.io/badge/Live%20API-Open-009688?style=flat&logo=fastapi&logoColor=white)](https://cnoret-getaround-API.hf.space/)
[![CI](https://github.com/cnoret/getaround-pricing/actions/workflows/ci.yml/badge.svg)](https://github.com/cnoret/getaround-pricing/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end ML project on Getaround car rental data - a price prediction API, an interactive analytics dashboard, and a full training pipeline with MLflow experiment tracking.

**Datasets:** pricing (~4,800 car listings · 13 features) · delay analysis (rental records with checkout times and consecutive booking gaps)

![Getaround Dashboard](dashboard_preview.png)

---

## Features

### Delay Analysis

- Interactive histogram of checkout delays with sentinel-value filtering (126 min encoding for >2h delays)
- Adjustable buffer threshold to simulate minimum time between consecutive rentals
- Business impact metrics: revenue at risk, blocked rentals, conflict resolution rate
- Delay breakdown by check-in type (Connect vs. mobile) with box plots

### Pricing Analysis

- Dataset KPIs: car count, average price, average mileage, average engine power
- Distribution charts for mileage and daily rental price
- Scatter plot: rental price vs. mileage colored by fuel type
- Average price by car type and by fuel type

### Price Prediction

- Form wired live to the FastAPI `/predict` endpoint
- Instant price estimate with delta vs. platform average
- Model metrics (MAE, RMSE, R²) in an expandable section

### Prediction API

- `POST /predict`: accepts one or multiple cars, returns predicted daily prices in euros
- `GET /health`: health check endpoint
- Pydantic v2 validation: field constraints + `Literal` types for all categoricals
- Interactive Swagger UI at `/docs`

---

## Tech Stack

| Layer              | Libraries                                                                 |
|--------------------|---------------------------------------------------------------------------|
| Data               | Pandas                                                                    |
| ML                 | Scikit-Learn (RandomForest, Pipeline, ColumnTransformer) · Joblib         |
| Experiment tracking| MLflow                                                                    |
| API                | FastAPI · Pydantic v2 · Uvicorn                                           |
| Dashboard          | Streamlit · Plotly                                                        |
| Infra              | Docker · Docker Compose                                                   |
| CI                 | GitHub Actions                                                            |
| Deployment         | Hugging Face Spaces                                                       |

---

## Quick Start

```bash
docker compose up --build
```

| Service          | URL                                               |
|------------------|---------------------------------------------------|
| Dashboard        | [localhost:8501](http://localhost:8501)           |
| API + Swagger UI | [localhost:8001/docs](http://localhost:8001/docs) |
| MLflow UI        | [localhost:5001](http://localhost:5001)           |

Docker Compose trains the model first (`condition: service_completed_successfully`), then starts the API and dashboard. Inter-service communication uses the `API_URL` environment variable.

### Run locally without Docker

```bash
git clone https://github.com/cnoret/getaround-pricing.git
cd getaround-pricing
pip install -r api/requirements.txt -r dashboard/requirements.txt

# Terminal 1 — API
uvicorn api.main:app --reload --port 8001

# Terminal 2 — Dashboard
streamlit run dashboard/app.py --server.port=8501
```

---

## Project Structure

```text
.
├── api/                   # FastAPI prediction API
│   ├── main.py
│   ├── requirements.txt
│   └── tests/
│       └── test_api.py
├── dashboard/             # Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
├── data/                  # Raw datasets
│   ├── get_around_delay_analysis.csv
│   └── get_around_pricing_project.csv
├── ml/                    # Model training
│   ├── model/
│   │   └── model.joblib
│   ├── model_training.py
│   └── requirements.txt
├── notebooks/             # Exploratory Data Analysis
│   ├── eda_delay.ipynb
│   └── eda_pricing.ipynb
├── .github/workflows/     # CI
│   └── ci.yml
├── Dockerfile.fastapi
├── Dockerfile.dashboard
├── Dockerfile.training
└── docker-compose.yml
```

---

## Notebooks

| Notebook                                         | Description                                                      |
|--------------------------------------------------|------------------------------------------------------------------|
| [eda_delay.ipynb](notebooks/eda_delay.ipynb)     | Delay distribution, sentinel value analysis, conflict simulation |
| [eda_pricing.ipynb](notebooks/eda_pricing.ipynb) | Pricing EDA, feature correlations, model selection rationale     |

---

## Results

| Metric | Value   |
|--------|---------|
| MAE    | 10.68 € |
| RMSE   | 16.73 € |
| R²     | 0.734   |

Random Forest with StandardScaler on numeric features (mileage, engine power) and OneHotEncoder on categoricals (brand, fuel, color, car type), evaluated on a 20% holdout set (random_state=42). Key predictors: engine power, car type, and model brand.

---

## Tests

```bash
pip install -r api/requirements.txt pytest httpx
pytest api/tests/ -v
```

11 tests covering `/health`, `/`, and `/predict`, including Pydantic validation errors (negative mileage, invalid fuel, missing fields). Model is mocked, no `.joblib` required. Also runs on every push via GitHub Actions.

---

## Retrain

```bash
pip install -r ml/requirements.txt
python ml/model_training.py
```

Metrics and artifacts are tracked in MLflow (`./mlruns`).

---

## License

MIT — [LICENSE](LICENSE)
