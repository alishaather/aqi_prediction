from src.utils.feature_store import load_live_features
from src.utils.model_registry import load_model_from_registry

HORIZONS = {"day1": ("24h", 24), "day2": ("48h", 48), "day3": ("72h", 72)}


def get_latest_forecast():
    """Loads the latest live features and produces a 3-day forecast
    using models pulled from the Hopsworks Model Registry."""
    feat_df = load_live_features()
    if feat_df.empty:
        return None

    latest_row = feat_df.iloc[[-1]]
    drop_cols = ["time", "us_aqi"]
    feature_cols = [c for c in latest_row.columns if c not in drop_cols]

    forecasts = {}
    for label, (display, horizon) in HORIZONS.items():
        model, scaler = load_model_from_registry(label)
        X = latest_row[feature_cols]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        forecasts[label] = (display, pred)

    return forecasts