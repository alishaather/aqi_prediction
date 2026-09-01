"""
Live SHAP explainability for the current forecast, using the deployed
model from the Hopsworks Model Registry and the latest live features.
"""
from src.utils.model_registry import load_model_from_registry
from src.utils.feature_store import load_live_features

def explain_forecast(label="day3"):
    import shap

    model, scaler = load_model_from_registry(label)
    df = load_live_features()
    df = df.sort_values("time").reset_index(drop=True)
    latest = df.iloc[[-1]]
    feature_cols = [c for c in latest.columns if c not in ["time", "us_aqi"]]

    X_scaled = scaler.transform(latest[feature_cols])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)
    shap_values.feature_names = feature_cols

    return shap_values, feature_cols

def summarize_top_drivers(shap_values, feature_cols, top_n=3):
    values = shap_values.values[0]
    pairs = sorted(zip(feature_cols, values), key=lambda p: abs(p[1]), reverse=True)
    top = pairs[:top_n]

    parts = []
    for name, val in top:
        direction = "raising" if val > 0 else "lowering"
        parts.append(f"{name} ({direction} the forecast)")

    return "The forecast was most influenced by " + ", ".join(parts) + "."