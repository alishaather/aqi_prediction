from src.training_pipeline.load_features import load_training_data
from src.training_pipeline.train import HORIZON_CONFIG, compare_models_for_horizon, train_final_models


def run_comparison():
    """Manual/report-generation only — compares all 4 models per horizon.
    Not meant to run on every automated trigger."""
    for label, config in HORIZON_CONFIG.items():
        print(f"\n############ Comparing models at {label} ({config['horizon']}h) ############")
        X, y, feature_cols = load_training_data(config["horizon"])
        compare_models_for_horizon(label, X, y)


def run_training():
    """Automated daily run — retrains and registers ONLY the winning model (Gradient Boosting)."""
    print("Training and registering final tuned models")
    train_final_models(load_training_data)


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        run_comparison()
    run_training()