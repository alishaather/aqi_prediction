"""
AQI Forecast Dashboard - Karachi

Run with:
    streamlit run app/streamlit_app.py

Assumes it's run from the project root (so `src.*` imports resolve), and
that model_registry/ already contains gb_day1.pkl, gb_day2.pkl, gb_day3.pkl
and their matching scaler_*.pkl files.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import streamlit as st
from src.feature_pipeline.fetch_data import fetch_combined_data
from src.feature_pipeline.build_features import create_inference_features
from src.utils.model_registry import load_model_from_registry

HORIZONS = {"day1": ("24h", 24), "day2": ("48h", 48), "day3": ("72h", 72)}
REGISTRY_DIR = "model_registry"


def aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "#00e400"
    elif aqi <= 100:
        return "Moderate", "#ffff00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97"
    else:
        return "Hazardous", "#7e0023"


@st.cache_resource
def load_models():
    models = {}
    for label in HORIZONS:
        model, scaler = load_model_from_registry(label)
        models[label] = (model, scaler)
    return models


@st.cache_data(ttl=1800)
def load_latest_data():
    raw_df = fetch_combined_data(past_days=14)
    return raw_df


def get_feature_row(raw_df):
    feat_df = create_inference_features(raw_df)
    if feat_df.empty:
        return None, None
    latest_row = feat_df.iloc[[-1]]
    drop_cols = ["time", "us_aqi"]
    feature_cols = [c for c in latest_row.columns if c not in drop_cols]
    return latest_row, feature_cols


def main():
    st.set_page_config(page_title="Karachi AQI Forecast", layout="centered")
    st.title("Karachi AQI Forecast")
    st.caption("3-day Air Quality Index forecast, powered by Open-Meteo data and a Gradient Boosting model.")

    with st.spinner("Fetching latest air quality data..."):
        raw_df = load_latest_data()

    if raw_df is None or raw_df.empty:
        st.error("Couldn't fetch live data right now. Please try again shortly.")
        return

    current_aqi = raw_df["us_aqi"].iloc[-1]
    current_time = raw_df["time"].iloc[-1]
    cat_name, cat_color = aqi_category(current_aqi)

    st.subheader("Current AQI")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("US AQI", f"{current_aqi:.0f}")
    with col2:
        st.markdown(
            f"<div style='background-color:{cat_color};padding:10px;border-radius:8px;"
            f"text-align:center;font-weight:bold;'>{cat_name}</div>",
            unsafe_allow_html=True,
        )
    st.caption(f"As of {current_time}")

    latest_row, feature_cols = get_feature_row(raw_df)
    if latest_row is None:
        st.warning(
            "Not enough recent history to build all features yet "
            "(need at least ~48 hours of continuous data). Try again later."
        )
        return

    models = load_models()

    st.subheader("3-Day Forecast")
    forecast_cols = st.columns(3)
    hazard_alerts = []

    for i, (label, (display, horizon)) in enumerate(HORIZONS.items()):
        model, scaler = models[label]
        try:
            X = latest_row[feature_cols]
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
        except ValueError as e:
            st.error(
                f"Model for {display} expects a different set of features than what's "
                f"available. Retrain with `python -m src.training_pipeline.run` "
                f"and refresh this page.\n\nDetails: {e}"
            )
            return

        cat_name, cat_color = aqi_category(pred)
        with forecast_cols[i]:
            st.markdown(f"**{display} ahead**")
            st.markdown(
                f"<div style='background-color:{cat_color};padding:14px;border-radius:8px;"
                f"text-align:center;'><span style='font-size:28px;font-weight:bold;'>{pred:.0f}</span>"
                f"<br>{cat_name}</div>",
                unsafe_allow_html=True,
            )

        if pred > 150:
            hazard_alerts.append((display, pred, cat_name))

    if hazard_alerts:
        st.subheader("Hazard Alerts")
        for display, pred, cat_name in hazard_alerts:
            st.error(f"{display} forecast: {pred:.0f} ({cat_name}) - sensitive groups should limit outdoor exposure.")
    else:
        st.success("No hazardous AQI levels expected in the next 3 days.")

    st.subheader("Recent AQI Trend")
    trend_df = raw_df[["time", "us_aqi"]].set_index("time").tail(24 * 7)
    st.line_chart(trend_df)

    with st.expander("Model explainability (SHAP)"):
        st.write(
            "SHAP summary plot for the 3-day model - shows which features push the "
            "forecast up or down, generated during training."
        )
        try:
            st.image("shap_summary_day3.png")
        except Exception:
            st.info("Run explain_model() in run.py first to generate shap_summary_day3.png.")


if __name__ == "__main__":
    main()