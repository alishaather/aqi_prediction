import os
import hopsworks
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_api_key():
    if "HOPSWORKS_API_KEY" in os.environ:
        return os.environ["HOPSWORKS_API_KEY"]
    return st.secrets["HOPSWORKS_API_KEY"]


def get_feature_store():
    project = hopsworks.login(api_key_value=get_api_key())
    return project.get_feature_store()

def save_training_features(df):
    """
    Saves fully engineered training data (includes target_aqi) to a
    dedicated feature group, separate from live inference features.
    """
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_training_features",
        version=1,
        description="Engineered AQI + weather features WITH target_aqi, for model training",
        primary_key=["time","horizon"],
        event_time="time",
        time_travel_format="HUDI"
    )
    fg.insert(df)
    print("Training features saved to Hopsworks (aqi_training_features)")


def save_live_features(df):
    """
    Saves live inference features (no target_aqi, since the future value
    doesn't exist yet) to a separate feature group.
    """
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_live_features",
        version=1,
        description="Engineered AQI + weather features WITHOUT target_aqi, for live inference",
        primary_key=["time"],
        event_time="time",
        time_travel_format="HUDI"
    )
    fg.insert(df)
    print("Live features saved to Hopsworks (aqi_live_features)")


def load_training_features():
    fs = get_feature_store()
    fg = fs.get_feature_group(name="aqi_training_features", version=1)
    return fg.read()


def load_live_features():
    fs = get_feature_store()
    fg = fs.get_feature_group(name="aqi_live_features", version=1)
    return fg.read()

def load_recent_live_features(days=7):
    fs = get_feature_store()
    fg = fs.get_feature_group(name="aqi_live_features", version=1)
    df = fg.read()
    df = df.sort_values("time").reset_index(drop=True)
    return df.tail(days * 24)

