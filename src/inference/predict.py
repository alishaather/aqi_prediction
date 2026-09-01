from src.utils.feature_store import load_live_features, load_recent_live_features
from src.utils.model_registry import load_model_from_registry

HORIZONS = {"day1": ("24h", 24), "day2": ("48h", 48), "day3": ("72h", 72)}


def get_latest_forecast():
    df = load_live_features()
    if df is None or df.empty:
        return None, None, None

    df = df.sort_values("time").reset_index(drop=True)
    latest = df.iloc[[-1]]
    current_aqi = latest["us_aqi"].values[0]
    current_time = latest["time"].values[0]

    feature_cols = [c for c in latest.columns if c not in ["time", "us_aqi"]]

    forecasts = {}
    for label, (display, horizon) in HORIZONS.items():
        model, scaler = load_model_from_registry(label)
        X_scaled = scaler.transform(latest[feature_cols])
        pred = model.predict(X_scaled)[0]
        forecasts[label] = (display, pred)

    return current_aqi, current_time, forecasts

def get_recent_trend(days=7):
    df = load_recent_live_features(days=days)
    return df[["time", "us_aqi"]]