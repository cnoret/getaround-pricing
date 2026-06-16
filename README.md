# Getaround Price Prediction with Dashboard & API
![Getaround dashboard screenshot](https://placehold.co/900x180?text=Getaround+Dashboard+Demo)

---

## 🚀 Deployed Project

- **Streamlit Dashboard:** [Hugging Face Space – Dashboard](https://cnoret-getaround-dashboard.hf.space/)
- **Prediction API (FastAPI):** [Hugging Face Space – API](https://cnoret-getaround-API.hf.space/)


---

## 💡 Project Overview

This repository provides:
- An **interactive dashboard** for delay & pricing analytics (**Streamlit**)
- A **machine learning API** for real-time car price prediction (**FastAPI**)
- **All-in-one deployment** (locally, with Docker, or on Hugging Face Spaces)

---

## 🛠️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/cnoret/getaround-ml-dashboard-api.git
cd getaround-ml-dashboard-api
```

### 2. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r api/requirements.txt
pip install -r dashboard/requirements.txt
```

### 3. Start the FastAPI prediction API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
```
Visit: [http://localhost:8001/docs](http://localhost:8001/docs)

### 4. Start the Streamlit dashboard
```bash
streamlit run dashboard/app.py --server.port=8501
```
Visit: [http://localhost:8501](http://localhost:8501)

> The dashboard reads the API URL from the `API_URL` environment variable.
>
> - **Docker**: set automatically to `http://api:8001/predict` via `docker-compose.yml`.
> - **Local (no Docker)**: defaults to `http://localhost:8001/predict` — no change needed.

---

## 🐳 How to Run with Docker

### 1. Build and start all services
```bash
docker compose up --build
```
- **Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Prediction API:** [http://localhost:8001](http://localhost:8001)
- **MLflow UI:** [http://localhost:5001](http://localhost:5001)

### 2. API internal communication
Within Docker, the dashboard communicates with the API at:
```
http://api:8001/predict
```
*Do not use localhost for inter-container communication!*

---

## 📂 Project Structure

```text
.
├── api/                   # FastAPI ML API
│   ├── main.py
│   └── requirements.txt
├── dashboard/             # Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
├── data/                  # Datasets
│   ├── get_around_delay_analysis.csv
│   └── get_around_pricing_project.csv
├── ml/                    # Model training scripts
│   ├── model_training.py
│   └── requirements.txt
├── Dockerfile.fastapi
├── Dockerfile.dashboard
├── Dockerfile.training
├── docker-compose.yml
└── README.md
```

---

## 📎 Useful Links

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

*Feel free to contact me for any questions or contributions!*
