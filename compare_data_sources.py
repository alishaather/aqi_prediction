"""
One-off comparison script: does Open-Meteo's archive API vs forecast API
weather data produce meaningfully different model performance, over the
SAME 90-day window and SAME horizon (72h)? This does not modify any
existing pipeline files — it's a standalone diagnostic.

Run with: python compare_data_sources.py
"""

import pandas as pd
import requests
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.feature_pipeline.fetch_data import fetch_aqi_data, KARACHI_LAT, KARACHI_LON, TIMEZONE
from src.feature_pipeline.build_features import create_feature_pipeline
from src.training_pipeline.evaluate import evaluate_with_cv, evaluate_persistence_baseline
from src.training_pipeline.train import HORIZON_CONFIG

PAST_DAYS = 90
HORIZON = 72  # compare at the hardest horizon, day3


def fetch_weather_archive(past_days=90):
    end_date = (pd.Timestamp.now() - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=past_days)).strftime('%Y-%m-%d')
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": TIMEZONE
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_weather_forecast(past_days=90):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "past_days": past_days,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "timezone": TIMEZONE
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def build_dataset(weather_df, aqi_df):
    combined = pd.merge(aqi_df, weather_df, on="time", how="inner")
    df = create_feature_pipeline(combined, horizon=HORIZON)
    drop_cols = ["time", "us_aqi", "target_aqi"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target_aqi"]
    return X, y


def run_all_models(label, X, y):
    print(f"\n========== {label} ==========")
    print(f"Dataset shape: {X.shape}")

    print("\n--- BASELINE: Naive Persistence ---")
    evaluate_persistence_baseline(X, y)

    print("\n--- MODEL 1: Linear Regression ---")
    evaluate_with_cv(LinearRegression(), X, y)

    print("\n--- MODEL 2: Ridge Regression ---")
    evaluate_with_cv(Ridge(alpha=10.0), X, y)

    print("\n--- MODEL 3: Random Forest ---")
    evaluate_with_cv(RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=3, random_state=42), X, y)

    print("\n--- MODEL 4: Gradient Boosting - tuned ---")
    params = HORIZON_CONFIG["day3"]["params"]
    evaluate_with_cv(GradientBoostingRegressor(**params, random_state=42), X, y)


if __name__ == "__main__":
    print("Fetching AQI data (shared between both comparisons)...")
    aqi_df = fetch_aqi_data(past_days=PAST_DAYS)

    print("\nFetching ARCHIVE weather data...")
    archive_weather = fetch_weather_archive(past_days=PAST_DAYS)
    X_archive, y_archive = build_dataset(archive_weather, aqi_df)
    run_all_models("ARCHIVE API (90 days)", X_archive, y_archive)

    print("\nFetching FORECAST API weather data...")
    forecast_weather = fetch_weather_forecast(past_days=PAST_DAYS)
    X_forecast, y_forecast = build_dataset(forecast_weather, aqi_df)
    run_all_models("FORECAST API (90 days)", X_forecast, y_forecast)