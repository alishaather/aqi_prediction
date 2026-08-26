import pandas as pd
import requests
from typing import Optional

#karachi coordinates 
KARACHI_LAT = 24.8608
KARACHI_LON = 67.0104
TIMEZONE = "Asia/Karachi"


def fetch_aqi_data(city: str="Karachi", past_days:int=30) -> Optional[pd.DataFrame]:
    """
    Fetches historical air quality data from Open-Meteo API for karachi.
    Parameters:
        past days(int): Number of past days to fetch.
    Returns:
        DataFrame with hourly AQI metrics or None if request fails.
    """

    print(f'Fetching AQI data for {city} (past {past_days} days)...')
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params={
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "past_days": past_days,
        "timezone": TIMEZONE
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  #will raise an HTTP error if the response was unsuccessful
        data =response.json()
        df = pd.DataFrame(data["hourly"]) #convert hourly dictionary into pandas dataframe
        df["time"] = pd.to_datetime(df["time"]) #convert time string column into date type objects

        print(f"Successfully fetched {len(df)} rows of AQI data for Karachi!")
        return df

    except Exception as e:
        print(f"Failed to fetch AQI data: {e}")
        return None

    

def fetch_weather_data(city: str = "Karachi", past_days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetches historical weather data from Open-Meteo Weather API for Karachi.
    Parameters:
        city (str): Name of the city (default: "Karachi").
        past_days (int): Number of past days to fetch (default: 30).   
    Returns:
        Optional[pd.DataFrame]: DataFrame with hourly weather metrics or None if request fails.
    """
    print(f"Fetching Weather data for {city} (past {past_days} days)...")
    
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation",
        "past_days": past_days,
        "timezone": TIMEZONE
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])

        print(f"Successfully fetched {len(df)} rows of Weather data for {city}!")
        return df

    except Exception as e:
        print(f"Failed to fetch Weather data: {e}")
        return None


def fetch_combined_data(city: str = "Karachi", past_days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetches both AQI and Weather data for Karachi and merges them on timestamp.
    """
    aqi_df = fetch_aqi_data(city=city, past_days=past_days)
    weather_df = fetch_weather_data(city=city, past_days=past_days)

    if aqi_df is not None and weather_df is not None:
        # Merge both datasets using the time column
        combined_df = pd.merge(aqi_df, weather_df, on="time", how="inner")
        print(f"Successfully combined datasets into {combined_df.shape[0]} rows and {combined_df.shape[1]} columns!")
        return combined_df
    else:
        print("Failed to merge datasets due to an error fetching AQI or Weather data.")
        return None