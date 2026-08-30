from src.training_pipeline.load_features import load_training_data
from src.training_pipeline.train import HORIZON_CONFIG, compare_models_for_horizon, train_final_models


if __name__ == "__main__":
    print("STEP 1: Comparing models per horizon (loading from feature store)")
    for label, config in HORIZON_CONFIG.items():
        print(f"\n############ Comparing models at {label} ({config['horizon']}h) ############")
        X, y, feature_cols = load_training_data(config["horizon"])
        compare_models_for_horizon(label, X, y)

    print("\nSTEP 2: Training and registering final tuned models")
    train_final_models(load_training_data)