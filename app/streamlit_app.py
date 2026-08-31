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
#         st.write("SHAP summary plot for the 3-day model - shows which features push the forecast up or down.")
#         try:
#             st.image("shap_summary_day3.png")
#         except Exception:
#             st.info("Run explain_model() first to generate shap_summary_day3.png.")


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
from src.inference.predict import get_latest_forecast
from src.inference.explain import explain_forecast

HORIZONS_ORDER = ["day1", "day2", "day3"]


def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00e400", "😊"
    elif aqi <= 100:
        return "Moderate", "#ffff00", "😐"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "😷"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000", "⚠️"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97", "🚨"
    else:
        return "Hazardous", "#7e0023", "💀"


@st.cache_data(ttl=1800, show_spinner="Loading latest air quality forecast...")
def load_forecast_data():
    return get_latest_forecast()


def main():
    st.set_page_config(
        page_title="Karachi AQI Forecast",
        page_icon="🌍",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e0e0e0;
        }
        .aqi-card {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .aqi-value {
            font-size: 48px;
            font-weight: 700;
            line-height: 1.2;
        }
        .aqi-label {
            font-size: 18px;
            font-weight: 500;
            margin-top: 4px;
        }
        .forecast-card {
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            height: 100%;
        }
        .forecast-day {
            font-size: 14px;
            font-weight: 600;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .forecast-value {
            font-size: 32px;
            font-weight: 700;
            margin: 0.5rem 0;
        }
        .forecast-category {
            font-size: 14px;
            font-weight: 500;
        }
        .hazard-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            margin: 0.5rem 0;
        }
        .hazard-box-error {
            padding: 1rem;
            border-radius: 8px;
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            margin: 0.5rem 0;
        }
        .success-box {
            padding: 1rem;
            border-radius: 8px;
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            margin: 0.5rem 0;
        }
        .timestamp {
            text-align: center;
            color: #6c757d;
            font-size: 14px;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }
        .divider {
            margin: 2rem 0;
            border-top: 2px solid #e9ecef;
        }
        .feature-importance {
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🌍 Karachi AQI Forecast</h1>
            <p style="color: #6c757d; font-size: 16px; margin-top: -0.5rem;">
                3-day Air Quality Index forecast powered by Open-Meteo data & Gradient Boosting
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading latest forecast..."):
        current_aqi, current_time, forecasts = load_forecast_data()

    if current_aqi is None:
        st.error("❌ Couldn't load live features right now. Please try again shortly.")
        return

    cat_name, cat_color, cat_emoji = aqi_category(current_aqi)

    # Current AQI Card
    st.markdown("### 📊 Current Air Quality")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div class="aqi-card" style="background-color: {cat_color}20; border: 2px solid {cat_color};">
                <div style="font-size: 52px; margin-bottom: 0.25rem;">{cat_emoji}</div>
                <div class="aqi-value" style="color: {cat_color};">{current_aqi:.0f}</div>
                <div class="aqi-label" style="color: {cat_color};">{cat_name}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="timestamp">📅 As of {current_time}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 3-Day Forecast
    st.markdown("### 📈 3-Day Forecast")

    forecast_cols = st.columns(3)
    hazard_alerts = []

    day_labels = {
        "day1": "Tomorrow",
        "day2": "Day 2",
        "day3": "Day 3"
    }

    for i, label in enumerate(HORIZONS_ORDER):
        display, pred = forecasts[label]
        cat_name_f, cat_color_f, cat_emoji_f = aqi_category(pred)
        
        with forecast_cols[i]:
            st.markdown(f"""
                <div class="forecast-card">
                    <div class="forecast-day">{day_labels.get(display, display)}</div>
                    <div style="font-size: 28px; margin: 0.25rem 0;">{cat_emoji_f}</div>
                    <div class="forecast-value" style="color: {cat_color_f};">{pred:.0f}</div>
                    <div class="forecast-category" style="color: {cat_color_f};">{cat_name_f}</div>
                </div>
            """, unsafe_allow_html=True)

        if pred > 150:
            hazard_alerts.append((display, pred, cat_name_f))

    # Hazard Alerts
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🚨 Alerts")

    if hazard_alerts:
        for display, pred, cat_name_f in hazard_alerts:
            st.markdown(f"""
                <div class="hazard-box-error">
                    <strong>⚠️ {display} forecast:</strong> {pred:.0f} AQI ({cat_name_f})<br>
                    <span style="font-size: 14px;">Sensitive groups should limit outdoor exposure.</span>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="success-box">
                ✅ No hazardous AQI levels expected in the next 3 days.
            </div>
        """, unsafe_allow_html=True)

    # Model Explainability (SHAP)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    with st.expander("🔍 Model Explainability (SHAP)", expanded=False):
        st.markdown("""
            <div class="feature-importance">
                <p style="margin-bottom: 0.75rem;">
                    <strong>How the model makes predictions:</strong> 
                    The SHAP summary plot below shows which features push the forecast up or down.
                </p>
                <p style="font-size: 14px; color: #6c757d;">
                    🔴 Red = Higher feature value &nbsp;|&nbsp; 🔵 Blue = Lower feature value<br>
                    → Features on the right push AQI higher, left pushes it lower.
                </p>
            </div>
        """, unsafe_allow_html=True)

        try:
            st.image("shap_summary_day3.png", use_column_width=True)
        except Exception:
            st.info("💡 Run `explain_model()` first to generate `shap_summary_day3.png`.")

    # Footer
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0 0.5rem 0; color: #adb5bd; font-size: 12px; border-top: 1px solid #e9ecef; margin-top: 2rem;">
            Data: Open-Meteo API | Model: Gradient Boosting | Karachi, Pakistan
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()