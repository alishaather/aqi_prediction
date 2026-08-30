from src.utils.feature_store import load_training_features


def load_training_data(horizon):
    """Reads training features back from Hopsworks and filters to one horizon."""
    all_features = load_training_features()
    df = all_features[all_features["horizon"] == horizon].copy()
    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop(columns=["horizon"])

    drop_cols = ["time", "us_aqi", "target_aqi"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df["target_aqi"]
    return X, y, feature_cols