import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_with_cv(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        print(f"  Fold {fold+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

    print(f"\nAverage over {n_splits} folds: RMSE={np.mean(rmses):.2f}, MAE={np.mean(maes):.2f}, R2={np.mean(r2s):.2f}")
    return np.mean(rmses)


def evaluate_persistence_baseline(X, y, n_splits=5):
    """Naive baseline: assume AQI stays the same as its last known reading."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, r2s = [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        y_test = y.iloc[test_idx]
        preds = X["us_aqi_lag_1h"].iloc[test_idx]

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        print(f"  Fold {fold+1}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}")

    print(f"\nPersistence baseline average over {n_splits} folds: RMSE={np.mean(rmses):.2f}, MAE={np.mean(maes):.2f}, R2={np.mean(r2s):.2f}")
    return np.mean(rmses)