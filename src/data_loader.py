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

    print(f'Fetching AQI data for {city}...')
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
   

