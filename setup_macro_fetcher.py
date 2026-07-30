code = r'''"""
macro_data_fetcher.py - Institutional Macro Economic Data Fetcher
Fetches US-DE Rate Differential, DXY, and VIX from FRED API.
Aligns daily data to H1 timeframe via forward-fill.
"""
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

class MacroDataFetcher:
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # FRED Series IDs
    SERIES = {
        "US_10Y": "DGS10",            # US 10-Year Treasury (Daily)
        "DE_10Y": "IRLTLT01DEM156N",  # Germany 10-Year Bond (Monthly)
        "DXY": "DTWEXBGS",            # Trade Weighted USD Index (Daily)
        "VIX": "VIXCLS",              # CBOE VIX (Daily)
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """Fetches a single FRED series."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            if response.status_code != 200:
                print(f"  Error fetching {series_id}: HTTP {response.status_code}")
                return pd.Series(dtype=float)
                
            data = response.json()
            observations = data.get("observations", [])
            
            if not observations:
                return pd.Series(dtype=float)
                
            dates = []
            values = []
            for obs in observations:
                try:
                    val = float(obs["value"])
                    dates.append(pd.to_datetime(obs["date"]))
                    values.append(val)
                except (ValueError, KeyError):
                    continue
                    
            return pd.Series(values, index=dates, name=series_id)
            
        except Exception as e:
            print(f"  Error fetching {series_id}: {e}")
            return pd.Series(dtype=float)
    
    def fetch_all_macro(self, start_date="2019-01-01", end_date="2024-06-30") -> pd.DataFrame:
        """Fetches all macro series and aligns to daily."""
        print(f"Fetching macro data from FRED ({start_date} to {end_date})...")
        
        all_data = {}
        for name, series_id in self.SERIES.items():
            print(f"  Fetching {name} ({series_id})...")
            series = self.fetch_series(series_id, start_date, end_date)
            if not series.empty:
                all_data[name] = series
            time.sleep(0.5)  # Rate limiting
            
        if not all_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_data)
        df.index.name = "date"
        
        # Calculate US-DE Rate Differential
        if "US_10Y" in df.columns and "DE_10Y" in df.columns:
            df["US_DE_Spread"] = df["US_10Y"] - df["DE_10Y"]
            print("  Calculated US-DE Rate Differential")
            
        # Forward-fill (macro data is lower frequency than H1)
        df.ffill(inplace=True)
        
        return df
    
    def align_to_h1(self, macro_df: pd.DataFrame, h1_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Aligns daily macro data to H1 timeframe via forward-fill."""
        if macro_df.empty:
            return pd.DataFrame(index=h1_index)
            
        # Reindex to H1 timestamps, forward-filling daily values
        aligned = macro_df.reindex(macro_df.index.union(h1_index))
        aligned.ffill(inplace=True)
        aligned = aligned.loc[h1_index]
        
        return aligned


if __name__ == "__main__":
    API_KEY = "YOUR_FRED_API_KEY_HERE"
    
    if API_KEY == "YOUR_FRED_API_KEY_HERE":
        print("ERROR: Please register at https://fredaccount.stlouisfed.org/apikeys")
        print("       and replace YOUR_FRED_API_KEY_HERE with your actual key.")
    else:
        fetcher = MacroDataFetcher(API_KEY)
        macro_df = fetcher.fetch_all_macro("2019-01-01", "2024-06-30")
        
        if not macro_df.empty:
            print(f"\nTotal macro data points: {len(macro_df)}")
            print(f"Columns: {list(macro_df.columns)}")
            print("\nSample Macro Data:")
            print(macro_df.tail(10))
            
            # Save to CSV
            os.makedirs("data", exist_ok=True)
            macro_df.to_csv("data/macro_data.csv")
            print("\nSaved to data/macro_data.csv")
        else:
            print("No macro data fetched. Check API key or network connection.")
'''
open("macro_data_fetcher.py", "w", encoding="utf-8").write(code)
print("SUCCESS: macro_data_fetcher.py created")

import ast
ast.parse(code)
print("AST OK")
