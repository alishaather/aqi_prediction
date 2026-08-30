from src.feature_pipeline.fetch_data import fetch_combined_data
from src.feature_pipeline.build_features import create_feature_pipeline, create_inference_features
from src.utils.feature_store import save_training_features, save_live_features


def run_training_features(past_days=365):
    """Fetches historical data and pushes engineered training features
    (with target_aqi, tagged by horizon) to the Hopsworks feature store."""
    print("Fetching historical data for training features...")
    raw_df = fetch_combined_data(past_days=past_days)

    horizons = {"day1": 24, "day2": 48, "day3": 72}
    for label, horizon in horizons.items():
        print(f"Building training features for {label} ({horizon}h)...")
        df = create_feature_pipeline(raw_df, horizon=horizon)
        df = df.copy()
        df["horizon"] = horizon
        save_training_features(df)

    return raw_df


def run_live_features(past_days=14):
    """Fetches recent data and pushes live inference features
    (no target_aqi) to the Hopsworks feature store."""
    print("Fetching recent data for live features...")
    raw_df = fetch_combined_data(past_days=past_days)
    features_df = create_inference_features(raw_df)
    save_live_features(features_df)
    return features_df


if __name__ == "__main__":
    run_training_features()
    run_live_features()