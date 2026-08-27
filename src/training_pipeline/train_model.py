import os
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Imports from feature_pipeline folder
from src.feature_pipeline.data_loader import fetch_combined_data
from src.feature_pipeline.features import create_feature_pipeline



# def prepare_data():
#     print("STEP 1: Ingesting & Engineering Features")
#     raw_df = fetch_combined_data(past_days=365)
#     print(f"\nRaw data range: {raw_df['time'].min()} to {raw_df['time'].max()}")
#     print(f"Missing values per column:\n{raw_df.isnull().sum()}")
    
#     if raw_df is None or raw_df.empty:
#         print("Error: Could not fetch data.")
#         return None, None, None, None, None, None     
#     df = create_feature_pipeline(raw_df, horizon=72)

#     # Separate Features (X) and Target (y)
#     drop_cols = ["time", "us_aqi","target_aqi"]
#     feature_cols = [c for c in df.columns if c not in drop_cols]
#     print("\n--- Features inside X ---")
#     print(feature_cols)
    
#     X = df[feature_cols]
#     y = df["target_aqi"]

#     # Chronological Train/Test Split (80% train, 20% test) - no shuffling
#     split_idx = int(len(df) * 0.8)
    
#     X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
#     y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
#     print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
#     #######################
#     print("\nTrain target stats:")
#     print(y_train.describe())
#     print("\nTest target stats:")
#     print(y_test.describe())

#     # Feature Scaling
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols,X_test


def prepare_data():
    print("STEP 1: Ingesting & Engineering Features")
    raw_df = fetch_combined_data(past_days=120)
    print(f"\nRaw data range: {raw_df['time'].min()} to {raw_df['time'].max()}")
    print(f"Missing values per column:\n{raw_df.isnull().sum()}")


    plot_aqi_timeline(raw_df) 
    
    if raw_df is None or raw_df.empty:
        print("Error: Could not fetch data.")
        return None, None, None
    df = create_feature_pipeline(raw_df, horizon=72)

    drop_cols = ["time", "us_aqi", "target_aqi"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df["target_aqi"]

    return X, y, feature_cols


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


def evaluate_baseline(X_test_raw, y_test):
    print("\n--- BASELINE: Naive Persistence (AQI now = AQI in 72h) ---")
    preds = X_test_raw["us_aqi_lag_1h"]

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"Baseline MAE  : {mae:.2f}")
    print(f"Baseline RMSE : {rmse:.2f}")
    print(f"Baseline R2   : {r2:.2f}")

    return rmse


def train_linear_regression(X_train, X_test, y_train, y_test,feature_cols):
    print("\n--- MODEL 1: Linear Regression ---")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Linear Regression MAE  : {mae:.2f}")
    print(f"Linear Regression RMSE : {rmse:.2f}")
    print(f"Linear Regression R2   : {r2:.2f}")

    print("\nLinear Regression coefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    return model, rmse

from sklearn.linear_model import Ridge

def train_ridge(X_train, X_test, y_train, y_test, alpha=10.0):
    print(f"\n--- MODEL 2: Ridge Regression (alpha={alpha}) ---")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Ridge MAE  : {mae:.2f}")
    print(f"Ridge RMSE : {rmse:.2f}")
    print(f"Ridge R2   : {r2:.2f}")
    
    return model, rmse

if __name__ == "__main__":
    X, y, feature_cols = prepare_data()
    
    if X is not None:
        print("\n--- MODEL 1: Linear Regression ---")
        evaluate_with_cv(LinearRegression(), X, y)

        print("\n--- MODEL 2: Ridge Regression ---")
        evaluate_with_cv(Ridge(alpha=10.0), X, y)

        print("\n--- MODEL 3: Random Forest ---")
        evaluate_with_cv(RandomForestRegressor(n_estimators=300, max_depth=None,min_samples_leaf=3, random_state=42), X, y)

        print("\n--- MODEL 4: Gradient Boosting Regressor ---")
        evaluate_with_cv(GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42), X, y)