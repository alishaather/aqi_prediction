"""
Live SHAP explainability for the current forecast, using the deployed
model from the Hopsworks Model Registry and the latest live features.

Older version (training-time SHAP on a sample of training data, saved as
a static PNG) is commented out below in case this doesn't work in the
deployed environment.
"""


from src.utils.model_registry import load_model_from_registry
from src.utils.feature_store import load_live_features


# import joblib
# import matplotlib.pyplot as plt
# from src.training_pipeline.load_features import load_training_data
# from src.training_pipeline.train import HORIZON_CONFIG
#
# def explain_model(label="day3", registry_dir="model_registry"):
#     import shap
#     model = joblib.load(f"{registry_dir}/gb_{label}.pkl")
#     scaler = joblib.load(f"{registry_dir}/scaler_{label}.pkl")
#     horizon = HORIZON_CONFIG[label]["horizon"]
#     X, y, feature_cols = load_training_data(horizon)
#     X_scaled = scaler.transform(X)
#     explainer = shap.TreeExplainer(model)
#     sample = X_scaled[:200]
#     shap_values = explainer(sample)
#     shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
#     plt.tight_layout()
#     plt.savefig(f"shap_summary_{label}.png")
#     plt.close()
#     print(f"Saved SHAP summary to shap_summary_{label}.png")
#     return explainer, shap_values
#
# if __name__ == "__main__":
#     explain_model(label="day3")


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

    return shap_values, feature_cols