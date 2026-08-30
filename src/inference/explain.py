import joblib
import matplotlib.pyplot as plt

from src.training_pipeline.load_features import load_training_data
from src.training_pipeline.train import HORIZON_CONFIG


def explain_model(label="day3", registry_dir="model_registry"):
    import shap

    model = joblib.load(f"{registry_dir}/gb_{label}.pkl")
    scaler = joblib.load(f"{registry_dir}/scaler_{label}.pkl")

    horizon = HORIZON_CONFIG[label]["horizon"]
    X, y, feature_cols = load_training_data(horizon)
    X_scaled = scaler.transform(X)

    explainer = shap.TreeExplainer(model)
    sample = X_scaled[:200]
    shap_values = explainer(sample)

    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{label}.png")
    plt.close()
    print(f"Saved SHAP summary to shap_summary_{label}.png")

    return explainer, shap_values


if __name__ == "__main__":
    explain_model(label="day3")