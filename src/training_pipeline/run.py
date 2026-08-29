import joblib
import matplotlib.pyplot as plt
from src.feature_pipeline.fetch_data import fetch_combined_data
from src.feature_pipeline.build_features import create_feature_pipeline
from src.training_pipeline.train import HORIZON_CONFIG, compare_models_for_horizon, train_final_models


def fetch_raw_data():
    print("STEP 1: Ingesting Raw Data")
    raw_df = fetch_combined_data(past_days=365)
    print(f"\nRaw data range: {raw_df['time'].min()} to {raw_df['time'].max()}")
    print(f"Missing values per column:\n{raw_df.isnull().sum()}")
    return raw_df


def prepare_data(raw_df, horizon=72):
    df = create_feature_pipeline(raw_df, horizon=horizon)
    drop_cols = ["time", "us_aqi", "target_aqi"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target_aqi"]
    return X, y, feature_cols


def explain_model(raw_df, label="day3", registry_dir="model_registry"):
    import shap  # lazy import — avoids blocking the rest of the pipeline if shap/numba fails to load

    model = joblib.load(f"{registry_dir}/gb_{label}.pkl")
    scaler = joblib.load(f"{registry_dir}/scaler_{label}.pkl")

    horizon = HORIZON_CONFIG[label]["horizon"]
    X, y, feature_cols = prepare_data(raw_df, horizon=horizon)
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
    raw_df = fetch_raw_data()

    # Stage 1: compare all models at all 3 horizons (uses tuned GB params)
    for label, config in HORIZON_CONFIG.items():
        print(f"\n############ Comparing models at {label} ({config['horizon']}h) ############")
        X, y, feature_cols = prepare_data(raw_df, horizon=config["horizon"])
        compare_models_for_horizon(label, X, y)

    # Stage 2: train and save final tuned models
    train_final_models(raw_df, prepare_data)

    # Stage 3: explain the 3-day model (uncomment once shap/numba issue is resolved)
    # explain_model(raw_df, label="day3")