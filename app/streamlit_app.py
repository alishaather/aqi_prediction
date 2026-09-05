"""
AQI Forecast Dashboard - Karachi

Run with:
    streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.inference.predict import get_latest_forecast, get_recent_trend
from src.inference.explain import explain_forecast, get_top_feature_breakdown
from src.utils.feature_store import load_live_features


def aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#1BB011"
    elif aqi <= 100:
        return "Moderate", "#D3B50B"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#EA5B13EC"
    elif aqi <= 200:
        return "Unhealthy", "#F30619"
    elif aqi <= 300:
        return "Very Unhealthy", "#82009ABD"
    else:
        return "Hazardous", "#680000"


def inject_css():
    st.markdown("""
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .navbar a {
            color: #FFFFFF;
            text-decoration: none;
            font-weight: 600;
            font-size: 15px;
        }
        .navbar a:hover {
            color: #7DD3E0;
        }

        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Inter', sans-serif !important;
        }

        .stApp h1 {
            text-align: center !important;
        }
        
        .metric-card {
            background-color: #12507A;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            border: 1px solid rgba(255,255,255,0.08);
            height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-card .value {
            font-size: 32px;
            font-weight: 700;
            color: #FFFFFF;
        }
        .metric-card .label {
            font-size: 11px;
            color: #B8D4E3;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
        }
        .category-badge {
            border-radius: 10px;
            padding: 14px;
            text-align: center;
            font-weight: 600;
            font-size: 16px;
            color: #0B3B5C;
        }

        .metric-card-elevated {
            background-color: #12507A;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 16px 32px rgba(0,0,0,0.4), 0 4px 8px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.15);
            height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;

        }

        div[data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .section {
            padding-top: 10px;
            margin-bottom: 40px;
        }
        [data-testid="stVegaLiteChart"] text {
            fill: #786B23 !important;
        }

        </style>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast_data():
    latest_row = load_latest_row()
    if latest_row is None:
        return None, None, None, None
    return get_latest_forecast(latest_row=latest_row)


@st.cache_data(ttl=1800, show_spinner=False)
def load_latest_row():
    df = load_live_features()
    if df is None or df.empty:
        return None
    df = df.sort_values("time").reset_index(drop=True)
    return df.iloc[[-1]]


@st.cache_data(ttl=1800, show_spinner=False)
def load_trend_data():
    return get_recent_trend(days=7)


def render_overview(current_aqi, current_time, cat_name, cat_color, conditions):
    st.header("Current Conditions", anchor=False)
    st.write("")

    col1, col2, col3, col4, col5 = st.columns([1,1,1.15,1,1])

    with col1:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Temperature</div>"
            f"<div class='value' style='font-size:20px;margin-top:6px;'>{conditions['temperature']:.1f}°C</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Humidity</div>"
            f"<div class='value' style='font-size:20px;margin-top:6px;'>{conditions['humidity']:.0f}%</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-card-elevated' style='background-color:{cat_color};'>"
            f"<div class='label' style='color:#0B3B5C;'>AQI</div>"
            f"<div class='value' style='color:#0B3B5C;margin-top:4px;font-size:30px;font-weight:700;'>{current_aqi:.0f}</div>"
            f"<div class='label' style='color:#0B3B5C;margin-top:4px;'>{cat_name}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Wind Speed</div>"
            f"<div class='value' style='font-size:20px;margin-top:6px;'>{conditions['wind_speed']:.1f} km/h</div></div>",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Wind Direction</div>"
            f"<div class='value' style='font-size:20px;margin-top:6px;'>{conditions['wind_direction']:.0f}°</div></div>",
            unsafe_allow_html=True,
        )

def render_forecast(forecasts):
    st.markdown("<div id='forecast' class='section'></div>", unsafe_allow_html=True)
    st.header("Forecast", anchor=False)

    cols = st.columns(3)
    hazard_alerts = []
    horizons_order = ["day1", "day2", "day3"]

    for i, label in enumerate(horizons_order):
        display, pred = forecasts[label]
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

    
    if hazard_alerts:
        for display, pred, cat_name in hazard_alerts:
            st.error(
                f"{display} forecast: {pred:.0f} ({cat_name}) - sensitive groups should limit outdoor exposure.",
                icon=":material/warning:",
            )
    else:
        st.markdown(
            "<div style='background-color:#2D7D46;color:white;padding:14px;"
            "border-radius:8px;font-weight:600; margin-top:24px;'>"
            "No hazardous AQI levels expected in the next 3 days.</div>",
            unsafe_allow_html=True,
        )


def render_trend():
    st.subheader("AQI Trend in Past Days", anchor=False)
    st.caption("Historical AQI levels over the past 7 days")

    with st.spinner("Loading trend data..."):
        trend_df = load_trend_data()

    if trend_df is None or trend_df.empty:
        st.info("Not enough trend data yet - check back once the pipeline has run for a while.", icon=":material/info:")
        return

    trend_df["time"] = pd.to_datetime(trend_df["time"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df["time"],
        y=trend_df["us_aqi"],
        mode="lines",
        line=dict(color="#4FA8E0", width=3, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(79,168,224,0.15)",
        name="AQI",
    ))

    thresholds = [
        (50, "Good/Moderate", "#A8E6A3"),
        (100, "Moderate", "#F5E27A"),
        (150, "Unhealthy (Sensitive)", "#F5B87A"),
        (200, "Unhealthy", "#E88686"),
    ]
    for level, label, color in thresholds:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color=color,
        )

    fig.update_layout(
        plot_bgcolor="#0B3B5C",
        paper_bgcolor="#0B3B5C",
        font_color="white",
        xaxis=dict(gridcolor="#12507A", title="Time"),
        yaxis=dict(gridcolor="#12507A", title="AQI"),
        margin=dict(t=30, b=30),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)



def render_explainability():
    st.markdown("<div id='explainability' class='section'></div>", unsafe_allow_html=True)
    st.header("Explainability", anchor=False)
    st.write("Feature contributions behind the 72h forecast.")
    try:
        import matplotlib.pyplot as plt
        import shap
        plt.style.use("dark_background")
        latest_row = load_latest_row()
        shap_values, feature_cols, scaler = explain_forecast(latest_row, label="day3")
        fig = plt.figure(facecolor="#0B3B5C")
        shap.plots.waterfall(shap_values[0], show=False)
        fig.patch.set_facecolor("#0B3B5C")
        st.pyplot(fig)
        st.subheader("Top Feature Breakdown", anchor=False)
        breakdown = get_top_feature_breakdown(shap_values,scaler,feature_cols)

        for row in breakdown:
            color = "#CE4949" if row["direction"] == "Raises AQI" else "#368730"
            arrow = "↑" if row["direction"] == "Raises AQI" else "↓"
            impact_sign = "+" if row["impact"] > 0 else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:10px 0;"
                f"border-bottom:1px solid #12507A;'>"
                f"<span style='color:#OB3B5C;'>#{row['rank']}</span>"
                f"<span style='font-weight:600;flex:1;padding-left:16px;'>{row['feature']}</span>"
                f"<span style='color:#OB3B5C;'>{row['value']}</span>"
                f"<span style='color:{color};padding-left:16px;'>{arrow} {row['direction']}</span>"
                f"<span style='color:{color};padding-left:16px;font-weight:600;'>{impact_sign}{row['impact']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        # st.info("Explainability temporarily unavailable.", icon=":material/info:")
        import traceback
        st.text(traceback.format_exc())


def main():
    st.set_page_config(page_title="Karachi AQI Forecast", layout="centered")
    inject_css()
    st.title("Karachi AQI Forecast🌍",anchor=False)
    st.markdown(
        "<p style='text-align:center;color:#0B3B5C;'>Data-driven air quality forecasts for Karachi, powered by Open-Meteo.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Fetching current air quality data..."):
        current_aqi, current_time, forecasts, conditions = load_forecast_data()

    if current_aqi is None:
        st.error("Couldn't load live features right now. Please try again shortly.")
        return

    cat_name, cat_color = aqi_category(current_aqi)

    render_overview(current_aqi,current_time, cat_name, cat_color,conditions,)
    render_forecast(forecasts)
    render_trend()
    render_explainability()


if __name__ == "__main__":
    main()

