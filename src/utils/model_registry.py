import os
import joblib
import hopsworks
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    if "HOPSWORKS_API_KEY" in os.environ:
        return os.environ["HOPSWORKS_API_KEY"]
    return st.secrets["HOPSWORKS_API_KEY"]


def get_model_registry():
    project = hopsworks.login(api_key_value=get_api_key)
    return project.get_model_registry()


def save_model_to_registry(model, scaler, label, metrics):
    mr = get_model_registry()

    local_dir = f"model_dir_{label}"
    os.makedirs(local_dir, exist_ok=True)
    joblib.dump(model, f"{local_dir}/model.pkl")
    joblib.dump(scaler, f"{local_dir}/scaler.pkl")

    hw_model = mr.python.create_model(
        name=f"aqi_gb_{label}",
        metrics=metrics,
        description=f"Gradient Boosting AQI forecaster - {label}"
    )
    hw_model.save(local_dir)
    print(f"Model {label} registered in Hopsworks Model Registry")


def load_model_from_registry(label):
    mr = get_model_registry()
    model_obj = mr.get_model(name=f"aqi_gb_{label}", version=1)
    download_dir = model_obj.download()
    
    model = joblib.load(f"{download_dir}/model.pkl")
    scaler = joblib.load(f"{download_dir}/scaler.pkl")
    return model, scaler