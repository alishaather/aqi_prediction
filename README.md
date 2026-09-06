# Karachi AQI Predictor

An end-to-end Air Quality Index (AQI) forecasting system for Karachi, predicting AQI 24, 48, and 72 hours in advance using automated data pipelines,Gradient Boosting models and a live Streamlit dashboard.

---

## Overview

- Fetches historical and live weather + pollutant data from the Open-Meteo API
- Engineers time-based, lag, rolling, and rate-of-change features
- Stores features in a Hopsworks Feature Store (separate groups for training and live inference)
- Trains and compares Linear Regression, Ridge, Random Forest, and Gradient Boosting using time-series cross-validation
- Tunes and registers the best model (Gradient Boosting) per forecast horizon in the Hopsworks Model Registry
- Serves forecasts through a Streamlit dashboard with SHAP explainability and hazard alerts
- Fully automated via GitHub Actions — hourly feature pipeline, daily training pipeline

---


## Project Documentation

**Full Project Report:** [Karachi_AQI_Report.pdf](report/Karachi_AQI_Report.pdf)

**Live Demo:** [https://aqiprediction-karachi.streamlit.app/](https://aqiprediction-karachi.streamlit.app/)

**GitHub Repository:** [https://github.com/alishaather/aqi_prediction](https://github.com/alishaather/aqi_prediction)

---

## Project Objectives

- Ingest raw weather and pollutant data from the Open-Meteo API and engineer relevant features and target variables
- Persist engineered features in a centralized Feature Store
- Backfill sufficient historical feature-target pairs to support robust model training
- Train and evaluate multiple models (Linear Regression, Ridge, Random Forest, Gradient Boosting) using RMSE, MAE, and R²
- Register the winning model in a Model Registry
- Automate the feature pipeline hourly and the training pipeline daily using CI/CD
- Serve predictions through a Streamlit dashboard
- Implement SHAP for model interpretability and feature importance explanation
- Integrate hazard alerts for dangerous forecasted AQI levels

---

## System Architecture

```text
Open-Meteo API
      │
      ▼
Feature Generation ──(hourly)──> GitHub Actions
      │
      ▼
Feature Store (Hopsworks)
      │
      ├──(training features)──> Model Training ──(daily)──> GitHub Actions
      │                              │
      │                              ▼
      │                        Model Registry (Hopsworks)
      │                              │
      └──(live features)─────────────┼──> Inference Service 
                                     │
                                     ▼
                          Streamlit Dashboard
                    (Forecast, SHAP, Hazard Alerts)
```

---

## Machine Learning Models

Five models were compared per forecast horizon using time-series cross-validation: a persistence baseline, Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting.

| Horizon | Best Model | RMSE | MAE | R² |
|---|---|---|---|---|
| 24h | Gradient Boosting (tuned) | 15.46 | 11.81 | 0.24 |
| 48h | Gradient Boosting (tuned) | 18.77 | 14.58 | -0.18 |
| 72h | Gradient Boosting (tuned) | 20.27 | 16.20 | -0.38 |

Gradient Boosting outperformed all other models and the naive persistence baseline at every horizon after hyperparameter tuning via `RandomizedSearchCV` with `TimeSeriesSplit`. See the full report for the R² discussion and methodology.

---

## Feature Engineering

**Time Features:** hour (cyclical sin/cos), day, month, day of week, weekend indicator

**Lag Features:** `us_aqi_lag_1h`, `us_aqi_lag_2h`, `us_aqi_lag_24h`, `us_aqi_lag_48h`

**Rolling Features:** `us_aqi_roll_mean_6h`, `us_aqi_roll_std_6h`, `us_aqi_roll_mean_24h`, `us_aqi_roll_std_24h`

**Rate-of-Change Features:** `us_aqi_change_rate_1h`, `us_aqi_change_rate_24h`

**Target:** `target_aqi = us_aqi.shift(-horizon)`, where horizon is 24, 48, or 72 hours, trained as three separate direct-forecast models rather than one recursive model.

---

## AQI Alert System

| AQI Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301+ | Hazardous |

The dashboard flags any forecast horizon where predicted AQI exceeds 150.

---

## MLOps Components

**Feature Store (Hopsworks)** — two feature groups: one for historical training data (composite primary key of `time` + `horizon`), one for live inference data.

**Model Registry (Hopsworks)** — Gradient Boosting models registered per horizon, with versioning.

---

## CI/CD Automation

Both pipelines run through GitHub Actions:

- **Feature Pipeline** (`feature_pipeline.yml`) — hourly. Fetches live data, engineers features, pushes to `aqi_live_features`.
- **Training Pipeline** (`training_pipeline.yml`) — daily at midnight PKT. Retrains and registers Gradient Boosting models for all horizons, commits updated model files back to the repo.

---


## Project Structure

```text
aqi_prediction/
├── .github/workflows/
│   ├── feature_pipeline.yml
│   └── training_pipeline.yml
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   └── streamlit_app.py
│
├── model_registry/
│   ├── gb_day1.pkl
│   ├── scaler_day1.pkl
│   ├── gb_day2.pkl
│   ├── scaler_day2.pkl
│   ├── gb_day3.pkl
│   └── scaler_day3.pkl
│
├── notebooks/
│   └── khi_eda.ipynb
│
├── src/
│   ├── feature_pipeline/
│   │   ├── __init__.py
│   │   ├── fetch_data.py
│   │   ├── build_features.py
│   │   └── run.py
│   │
│   ├── training_pipeline/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── load_features.py
│   │   ├── train.py
│   │   └── run.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── explain.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── feature_store.py
│   │   └── model_registry.py
│   │
│   └── __init__.py
│
├── report/
│   └── Karachi_AQI_Report.pdf
│
├── .gitignore
├── requirements.txt
├── runtime.txt
├── LICENSE
└── README.md
```
---

## Installation

Clone the repository:
```bash
git clone https://github.com/alishaather/aqi_prediction.git  
cd aqi_prediction
```

Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file with your Hopsworks API key:
HOPSWORKS_API_KEY=your_key_here

---

## Usage

Run the feature pipeline:
```bash
python -m src.feature_pipeline.run
```

Run the training pipeline:
```bash
python -m src.training_pipeline.run
```

Launch the dashboard:
```bash
streamlit run app/streamlit_app.py
```

---

## Technologies Used

Python, Pandas, NumPy, Scikit-Learn, SHAP, Streamlit, Hopsworks, GitHub Actions, Open-Meteo API

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

Alisha Ather 