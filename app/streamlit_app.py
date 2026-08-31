"""
AQI Forecast Dashboard - Karachi

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.inference.predict import get_latest_forecast
from src.inference.explain import explain_forecast

HORIZONS_ORDER = ["day1", "day2", "day3"]


def aqi_category(aqi):
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


@st.cache_data(ttl=1800, show_spinner="Loading latest air quality forecast...")
def load_forecast_data():
    return get_latest_forecast()


def main():
    st.set_page_config(page_title="Karachi AQI Forecast", layout="centered")
    st.title("Karachi AQI Forecast")
    st.caption("3-day Air Quality Index forecast, powered by Open-Meteo data and a Gradient Boosting model.")

    with st.spinner("Loading latest forecast..."):
        current_aqi, current_time, forecasts = load_forecast_data()

    if current_aqi is None:
        st.error("Couldn't load live features right now. Please try again shortly.")
        return

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

    st.subheader("3-Day Forecast")
    forecast_cols = st.columns(3)
    hazard_alerts = []

    for i, label in enumerate(HORIZONS_ORDER):
        display, pred = forecasts[label]
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

    with st.expander("Model explainability (SHAP)"):
        st.write("Feature contributions behind the 72h forecast.")
        try:
            import matplotlib.pyplot as plt
            import shap
            shap_values, feature_cols = explain_forecast(label="day3")
            fig = plt.figure()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(f"Explainability temporarily unavailable: {e}")
            # import traceback
            # st.text(traceback.format_exc())


if __name__ == "__main__":
    main()





