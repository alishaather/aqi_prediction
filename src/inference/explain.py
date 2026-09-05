"""
Live SHAP explainability for the current forecast, using the deployed
model from the Hopsworks Model Registry and the latest live features.
"""
from src.utils.model_registry import load_model_from_registry
from src.utils.feature_store import load_live_features

def explain_forecast(latest_row, label="day3"):
    import shap

    model, scaler = load_model_from_registry(label)
    feature_cols = [c for c in latest_row.columns if c not in ["time" ,"us_aqi"]]
    X_scaled = scaler.transform(latest_row[feature_cols])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_scaled)
    shap_values.feature_names = feature_cols

    return shap_values, feature_cols, scaler

def get_top_feature_breakdown(shap_values, scaler, feature_cols, top_n=8):
    values = shap_values.values[0]
    scaled_data = shap_values.data[0].reshape(1, -1)
    real_data = scaler.inverse_transform(scaled_data)[0]

    pairs = sorted(zip(feature_cols, real_data, values), key=lambda p: abs(p[2]), reverse=True)

    rows = []
    for rank, (name, feat_val, impact) in enumerate(pairs[:top_n], start=1):
        direction = "Raises AQI" if impact > 0 else "Lowers AQI"
        rows.append({
            "rank": rank,
            "feature": name,
            "value": round(float(feat_val), 3),
            "direction": direction,
            "impact": round(float(impact), 3),
        })
    return rows

def summarize_top_drivers(shap_values, feature_cols, top_n=3):
    values = shap_values.values[0]
    pairs = sorted(zip(feature_cols, values), key=lambda p: abs(p[1]), reverse=True)
    top = pairs[:top_n]

    parts = []
    for name, val in top:
        direction = "raising" if val > 0 else "lowering"
        parts.append(f"{name} ({direction} the forecast)")

    return "The forecast was most influenced by " + ", ".join(parts) + "."