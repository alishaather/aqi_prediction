from src.utils.feature_store import load_live_features, load_recent_live_features
from src.utils.model_registry import load_model_from_registry

HORIZONS = {"day1": ("24h", 24), "day2": ("48h", 48), "day3": ("72h", 72)}


def get_latest_forecast(latest_row=None):
    if latest_row is None:
        df = load_live_features()
        if df is None or df.empty:
            return None, None, None

        df = df.sort_values("time").reset_index(drop=True)
        latest_row = df.iloc[[-1]]
    current_aqi = latest_row["us_aqi"].values[0]
    current_time = latest_row["time"].values[0]

    conditions ={
        "temperature": latest_row["temperature_2m"].values[0],
        "humidity": latest_row["relative_humidity_2m"].values[0],
        "wind_speed": latest_row["wind_speed_10m"].values[0],
        "wind_direction": latest_row["wind_direction_10m"].values[0],
        "precipitation": latest_row["precipitation"].values[0],
        
    }
    feature_cols = [c for c in latest_row.columns if c not in ["time", "us_aqi"]]

    forecasts = {}
    for label, (display, horizon) in HORIZONS.items():
        model, scaler = load_model_from_registry(label)
        X_scaled = scaler.transform(latest_row[feature_cols])
        pred = model.predict(X_scaled)[0]
        forecasts[label] = (display, pred)

    return current_aqi, current_time, forecasts, conditions

def get_recent_trend(days=7):
    df = load_recent_live_features(days=days)
    return df[["time", "us_aqi"]]