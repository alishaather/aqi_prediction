import pandas as pd 
import numpy as np

def add_time_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Extract time based features from the datetime column
    """
    df = df.copy()
    #extract temporal components
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    #encoding for hours for smooth transition from 23 t0 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    return df

def add_lag_features(df: pd.DataFrame,target_col: str="us_aqi",lags: list = [1,2,24,48]) -> pd.DataFrame:
    """
    Create historical lag features for the target column
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target_col:str="us_aqi", windows: list=[6,24]) -> pd.DataFrame:
    """
    Creates moving averages and standard deviations over specified hour windows.
    """
    df = df.copy()
    
    for window in windows:
        df[f"{target_col}_roll_mean_{window}h"] = df[target_col].rolling(window=window).mean()
        df[f"{target_col}_roll_std_{window}h"] = df[target_col].rolling(window=window).std()
        
    return df


def create_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline function to generate all features and clean NaN rows caused by lagging.
    """
    df = add_time_features(df)
    df = add_lag_features(df, target_col="us_aqi", lags=[1, 2, 24, 48])
    df = add_rolling_features(df, target_col="us_aqi", windows=[6, 24])
    
    # Drop NaNs created by lag shifts and rolling calculations
    df_clean = df.dropna().reset_index(drop=True)
    
    print(f"Features created successfully! Dataset shape: {df_clean.shape}")
    return df_clean