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
    project = hopsworks.login(api_key_value=get_api_key)
    return project.get_feature_store()


def save_features(df):
    fs = get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="karachi_aqi_features",
        version=1,
        description="Engineered AQI + weather features for Karachi",
        primary_key=["time"],
        event_time="time"
    )
    fg.insert(df)
    print("Features saved to Hopsworks feature store")


def load_features():
    fs = get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_features", version=1)
    return fg.read()