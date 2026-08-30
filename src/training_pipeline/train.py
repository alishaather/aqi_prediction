import os
import joblib
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.training_pipeline.evaluate import evaluate_with_cv, evaluate_persistence_baseline
from src.utils.model_registry import save_model_to_registry
# Final tuned Gradient Boosting params per forecast horizon
# found via RandomizedSearchCV + TimeSeriesSplit. see report for methodology
HORIZON_CONFIG = {
    "day1": {
        "horizon": 24,
        "params": {"n_estimators": 100, "min_samples_split": 10, "min_samples_leaf": 5, "max_depth": 2, "learning_rate": 0.05},
    },
    "day2": {
        "horizon": 48,
        "params": {"n_estimators": 300, "min_samples_split": 5, "min_samples_leaf": 1, "max_depth": 2, "learning_rate": 0.01},
    },
    "day3": {
        "horizon": 72,
        "params": {"n_estimators": 300, "min_samples_split": 5, "min_samples_leaf": 1, "max_depth": 2, "learning_rate": 0.01},
    },
}


def compare_models_for_horizon(label, X, y):
    print(f"\n--- BASELINE: Naive Persistence ({label}) ---")
    evaluate_persistence_baseline(X, y)

    print(f"\n--- MODEL 1: Linear Regression ({label}) ---")
    evaluate_with_cv(LinearRegression(), X, y)

    print(f"\n--- MODEL 2: Ridge Regression ({label}) ---")
    evaluate_with_cv(Ridge(alpha=10.0), X, y)

    print(f"\n--- MODEL 3: Random Forest ({label}) ---")
    evaluate_with_cv(RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=3, random_state=42), X, y)

    print(f"\n--- MODEL 4: Gradient Boosting - tuned ({label}) ---")
    params = HORIZON_CONFIG[label]["params"]
    evaluate_with_cv(GradientBoostingRegressor(**params, random_state=42), X, y)


def train_final_models(prepare_data_fn):
    """
    prepare_data_fn: a function(horizon) -> (X, y, feature_cols), reading
    from the feature store rather than raw data directly.
    """
    os.makedirs("model_registry", exist_ok=True)

    for label, config in HORIZON_CONFIG.items():
        print(f"\n=== Training Gradient Boosting for {label} ({config['horizon']}h ahead) ===")
        X, y, feature_cols = prepare_data_fn(config["horizon"])

        model = GradientBoostingRegressor(**config["params"], random_state=42)
        evaluate_with_cv(model, X, y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        joblib.dump(model, f"model_registry/gb_{label}.pkl")
        joblib.dump(scaler, f"model_registry/scaler_{label}.pkl")
        print(f"Saved model_registry/gb_{label}.pkl")

        from src.utils.model_registry import save_model_to_registry
        metrics = {"n_estimators": config["params"]["n_estimators"]}
        save_model_to_registry(model, scaler, label, metrics)