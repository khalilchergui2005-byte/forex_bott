import subprocess
subprocess.check_call(["pip", "install", "gdeltdoc", "-q"])

code = r'''"""
gdelt_fetcher.py - GDELT Project News Data Fetcher
Fetches geopolitical and macro news tone/volume for EURUSD context.
"""
import pandas as pd
import numpy as np
from gdeltdoc import Filters, TimelineSearch, ApiClient
from datetime import datetime, timedelta
import os
import time

class GDELTFetcher:
    def __init__(self):
        self.api = ApiClient()
        
    def fetch_monthly_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches GDELT timeline tone and volume for a specific date range.
        GDELT DOC 2.0 API allows max 365 days per request for timeline.
        We query monthly to be safe and avoid timeouts.
        """
        # Focus on themes and keywords affecting EURUSD
        f = Filters(
            keyword="euro OR ecb OR eurozone OR european central bank",
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            # TimelineSearch returns volume and tone over time
            timeline = TimelineSearch(f).load()
            if timeline.empty:
                return pd.DataFrame()
                
            # timeline has columns: 'date', 'count', 'tone'
            # Rename for clarity
            timeline.rename(columns={
                'count': 'News_Volume',
                'tone': 'News_Tone'
            }, inplace=True)
            
            timeline['date'] = pd.to_datetime(timeline['date'])
            timeline.set_index('date', inplace=True)
            
            return timeline
            
        except Exception as e:
            print(f"  Error fetching GDELT {start_date} to {end_date}: {e}")
            return pd.DataFrame()

    def fetch_full_history(self, start_year=2019, end_year=2024) -> pd.DataFrame:
        """Fetches 5 years of GDELT data by looping through months."""
        print("Fetching GDELT News Data (this may take a moment)...")
        all_data = []
        
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 6, 30)
        
        current = start
        while current < end:
            next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            if next_month > end:
                next_month = end
                
            print(f"  Fetching {current.strftime('%Y-%m')}...")
            df = self.fetch_monthly_data(
                current.strftime("%Y%m%d"),
                next_month.strftime("%Y%m%d")
            )
            
            if not df.empty:
                all_data.append(df)
                
            current = next_month + timedelta(days=1)
            time.sleep(1.5)  # Rate limiting to avoid 429 Too Many Requests
            
        if not all_data:
            return pd.DataFrame()
            
        final_df = pd.concat(all_data)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)
        
        # Calculate 7-day rolling average for smoothing
        final_df['News_Volume_MA7'] = final_df['News_Volume'].rolling(7).mean()
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
'''
open("gdelt_fetcher.py", "w", encoding="utf-8").write(code)
print("SUCCESS: gdelt_fetcher.py created")

import ast
ast.parse(code)
print("AST OK")
