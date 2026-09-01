# """
# AQI Forecast Dashboard - Karachi

# Run with:
#     streamlit run app/streamlit_app.py
# """
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import streamlit as st
# from src.inference.predict import get_latest_forecast
# from src.inference.explain import explain_forecast

# HORIZONS_ORDER = ["day1", "day2", "day3"]


# def aqi_category(aqi):
#     if aqi <= 50:
#         return "Good", "#00e400"
#     elif aqi <= 100:
#         return "Moderate", "#ffff00"
#     elif aqi <= 150:
#         return "Unhealthy for Sensitive Groups", "#ff7e00"
#     elif aqi <= 200:
#         return "Unhealthy", "#ff0000"
#     elif aqi <= 300:
#         return "Very Unhealthy", "#8f3f97"
#     else:
#         return "Hazardous", "#7e0023"


# @st.cache_data(ttl=1800, show_spinner="Loading latest air quality forecast...")
# def load_forecast_data():
#     return get_latest_forecast()


# def main():
#     st.set_page_config(page_title="Karachi AQI Forecast", layout="centered")
#     st.title("Karachi AQI Forecast")
#     st.caption("3-day Air Quality Index forecast, powered by Open-Meteo data and a Gradient Boosting model.")

#     with st.spinner("Loading latest forecast..."):
#         current_aqi, current_time, forecasts = load_forecast_data()

#     if current_aqi is None:
#         st.error("Couldn't load live features right now. Please try again shortly.")
#         return

#     cat_name, cat_color = aqi_category(current_aqi)

#     st.subheader("Current AQI")
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.metric("US AQI", f"{current_aqi:.0f}")
#     with col2:
#         st.markdown(
#             f"<div style='background-color:{cat_color};padding:10px;border-radius:8px;"
#             f"text-align:center;font-weight:bold;'>{cat_name}</div>",
#             unsafe_allow_html=True,
#         )
#     st.caption(f"As of {current_time}")

#     st.subheader("3-Day Forecast")
#     forecast_cols = st.columns(3)
#     hazard_alerts = []

#     for i, label in enumerate(HORIZONS_ORDER):
#         display, pred = forecasts[label]
#         cat_name, cat_color = aqi_category(pred)
#         with forecast_cols[i]:
#             st.markdown(f"**{display} ahead**")
#             st.markdown(
#                 f"<div style='background-color:{cat_color};padding:14px;border-radius:8px;"
#                 f"text-align:center;'><span style='font-size:28px;font-weight:bold;'>{pred:.0f}</span>"
#                 f"<br>{cat_name}</div>",
#                 unsafe_allow_html=True,
#             )
#         if pred > 150:
#             hazard_alerts.append((display, pred, cat_name))

#     if hazard_alerts:
#         st.subheader("Hazard Alerts")
#         for display, pred, cat_name in hazard_alerts:
#             st.error(f"{display} forecast: {pred:.0f} ({cat_name}) - sensitive groups should limit outdoor exposure.")
#     else:
#         st.success("No hazardous AQI levels expected in the next 3 days.")

#     with st.expander("Model explainability (SHAP)"):
#         st.write("Feature contributions behind the 72h forecast.")
#         try:
#             import matplotlib.pyplot as plt
#             import shap
#             shap_values, feature_cols = explain_forecast(label="day3")
#             fig = plt.figure()
#             shap.plots.waterfall(shap_values[0], show=False)
#             st.pyplot(fig)
#         except Exception as e:
#             st.info(f"Explainability temporarily unavailable: {e}")
#             # import traceback
#             # st.text(traceback.format_exc())


# if __name__ == "__main__":
#     main()


"""
AQI Forecast Dashboard - Karachi

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.inference.predict import get_latest_forecast, get_recent_trend
from src.inference.explain import explain_forecast, summarize_top_drivers


def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#A8E6A3"
    elif aqi <= 100:
        return "Moderate", "#F5E27A"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#F5B87A"
    elif aqi <= 200:
        return "Unhealthy", "#E88686"
    elif aqi <= 300:
        return "Very Unhealthy", "#C99AD1"
    else:
        return "Hazardous", "#B37A8C"


def inject_css():
    st.markdown("""
        <style>
        .metric-card {
            background-color: #12507A;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .metric-card .value {
            font-size: 32px;
            font-weight: 700;
            color: #FFFFFF;
        }
        .metric-card .label {
            font-size: 13px;
            color: #B8D4E3;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .category-badge {
            border-radius: 10px;
            padding: 14px;
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            color: #0B3B5C;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner="Loading latest air quality forecast...")
def load_forecast_data():
    return get_latest_forecast()


@st.cache_data(ttl=1800, show_spinner="Loading recent trend...")
def load_trend_data():
    return get_recent_trend(days=7)


def render_overview(current_aqi, cat_name, cat_color):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='metric-card'><div class='value'>{current_aqi:.0f}</div>"
            f"<div class='label'>Current AQI</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='category-badge' style='background-color:{cat_color};'>{cat_name}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("7-Day AQI Trend")
    trend_df = load_trend_data()
    if trend_df is not None and not trend_df.empty:
        trend_df = trend_df.set_index("time")
        st.line_chart(trend_df)
    else:
        st.info("Not enough trend data yet - check back once the pipeline has run for a while.", icon=":material/info:")


def render_forecast(forecasts):
    st.subheader("3-Day Forecast")
    cols = st.columns(3)
    hazard_alerts = []
    preds = []
    horizons_order = ["day1", "day2", "day3"]

    for i, label in enumerate(horizons_order):
        display, pred = forecasts[label]
        preds.append((display, pred))
        cat_name, cat_color = aqi_category(pred)
        with cols[i]:
            st.markdown(f"**{display} ahead**")
            st.markdown(
                f"<div class='category-badge' style='background-color:{cat_color};padding:20px;'>"
                f"<span style='font-size:30px;font-weight:700;'>{pred:.0f}</span>"
                f"<br><span style='font-size:14px;'>{cat_name}</span></div>",
                unsafe_allow_html=True,
            )
        if pred > 150:
            hazard_alerts.append((display, pred, cat_name))

    peak = max(preds, key=lambda p: p[1])
    best = min(preds, key=lambda p: p[1])
    c1, c2 = st.columns(2)
    c1.metric("Peak Predicted AQI", f"{peak[1]:.0f}", help=f"at {peak[0]}")
    c2.metric("Best Predicted Period", f"{best[1]:.0f}", help=f"at {best[0]}")

    st.divider()
    if hazard_alerts:
        for display, pred, cat_name in hazard_alerts:
            st.error(
                f"{display} forecast: {pred:.0f} ({cat_name}) - sensitive groups should limit outdoor exposure.",
                icon=":material/warning:",
            )
    else:
        st.success("No hazardous AQI levels expected in the next 3 days.", icon=":material/check_circle:")


def render_explainability():
    st.write("Feature contributions behind the 72h forecast.")
    try:
        import matplotlib.pyplot as plt
        import shap
        shap_values, feature_cols = explain_forecast(label="day3")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        st.write(summarize_top_drivers(shap_values, feature_cols))
    except Exception:
        st.info("Explainability temporarily unavailable.", icon=":material/info:")


def main():
    st.set_page_config(page_title="Karachi AQI Forecast", layout="centered")
    inject_css()
    st.title("Karachi AQI Forecast")
    st.caption("Air quality forecasting for Karachi, powered by Open-Meteo data and a tuned Gradient Boosting model.")

    with st.spinner("Loading latest forecast..."):
        current_aqi, current_time, forecasts = load_forecast_data()

    if current_aqi is None:
        st.error("Couldn't load live features right now. Please try again shortly.")
        return

    cat_name, cat_color = aqi_category(current_aqi)

    page = st.sidebar.radio("Navigate", ["Overview", "Forecast", "Explainability"])

    if page == "Overview":
        render_overview(current_aqi, cat_name, cat_color)
    elif page == "Forecast":
        render_forecast(forecasts)
    elif page == "Explainability":
        render_explainability()


if __name__ == "__main__":
    main()

