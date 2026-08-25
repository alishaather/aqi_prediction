import pandas as pd
import requests
from typing import Optional

def fetch_aqi_data(city: str="Karachi") -> Optional[pd.DataFrame]:
    """Fetches aqi data for a given cty nd returns it as pandas dataframe"""
    print(f'Fetching AQI data for {city}...')

    return None