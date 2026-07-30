"""
gdelt_fetcher.py - GDELT Project News Data Fetcher (Direct API)
Fetches geopolitical and macro news tone/volume for EURUSD context.
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import time

class GDELTFetcher:
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def fetch_monthly_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches GDELT timeline tone directly via HTTP request.
        """
        start_dt = start_date + "000000"
        end_dt = end_date + "235959"
        
        params = {
            "query": "euro OR ecb OR eurozone",
            "mode": "timelinetone",
            "startdatetime": start_dt,
            "enddatetime": end_dt,
            "format": "json"
        }
        
        # FIX-INST: Retry mechanism for unstable GDELT servers
        for attempt in range(3):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    if "timeline" in data and len(data["timeline"]) > 0:
                        series = data["timeline"][0]
                        temp_df = pd.DataFrame(series["data"])
                        temp_df["datetime"] = pd.to_datetime(temp_df["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
                        temp_df.set_index("datetime", inplace=True)
                        temp_df.rename(columns={"value": "News_Tone"}, inplace=True)
                        temp_df.drop(columns=["date"], inplace=True, errors="ignore")
                        return temp_df
                    else:
                        return pd.DataFrame()
                else:
                    time.sleep(5)
                    
            except requests.exceptions.ReadTimeout:
                print(f"  Timeout on attempt {attempt+1}. Retrying...", end=" ")
                time.sleep(10)
            except Exception as e:
                print(f"  Error: {e}", end=" ")
                time.sleep(5)
                
        return pd.DataFrame()

    def fetch_full_history(self, start_year=2019, end_year=2024) -> pd.DataFrame:
        """Fetches 5 years of GDELT data by looping through months."""
        print("Fetching GDELT News Data via Direct API (with Retries)...")
        all_data = []
        
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 6, 30)
        
        current = start
        while current < end:
            next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            if next_month > end:
                next_month = end
                
            print(f"  Fetching {current.strftime('%Y-%m')}...", end=" ")
            df = self.fetch_monthly_data(
                current.strftime("%Y%m%d"),
                next_month.strftime("%Y%m%d")
            )
            
            if not df.empty:
                all_data.append(df)
                print("Success")
            else:
                print("Failed/No data")
                
            current = next_month + timedelta(days=1)
            time.sleep(2.0)  # Rate limiting to prevent blacklisting
            
        if not all_data:
            return pd.DataFrame()
            
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)
        
        # Calculate features
        if 'News_Tone' in final_df.columns:
            final_df['News_Tone_MA7'] = final_df['News_Tone'].rolling(7).mean()
            final_df['News_Tone_Z_Score'] = (final_df['News_Tone'] - final_df['News_Tone'].rolling(30).mean()) / (final_df['News_Tone'].rolling(30).std() + 1e-10)
        
        print(f"\n  Successfully processed {len(final_df)} days of GDELT data.")
        return final_df

if __name__ == "__main__":
    fetcher = GDELTFetcher()
    gdelt_df = fetcher.fetch_full_history(2019, 2024)
    
    if not gdelt_df.empty:
        print("\nSample GDELT Data:")
        print(gdelt_df.tail(10))
        
        os.makedirs("data", exist_ok=True)
        gdelt_df.to_csv("data/gdelt_data.csv")
        print("\nSaved to data/gdelt_data.csv")
    else:
        print("No GDELT data fetched.")
