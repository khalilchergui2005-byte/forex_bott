"""
build_historical_data.py - Institutional Multi-Year Data Builder
Downloads, aggregates, and stores real H1 data with volume for backtesting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dukascopy_fetcher import DukascopyFetcher

class HistoricalDataBuilder:
    def __init__(self, instrument="EURUSD"):
        self.instrument = instrument
        self.fetcher = DukascopyFetcher(instrument)
        self.file_path = f"data/{instrument.lower()}_h1_real.csv"
        
        if not os.path.exists("data"):
            os.makedirs("data")
            
    def build_period(self, start_date: str, end_date: str):
        """Downloads and aggregates tick data to H1 and saves to CSV."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        all_h1 = []
        
        # Load existing data to append
        if os.path.exists(self.file_path):
            existing_df = pd.read_csv(self.file_path, index_col=0, parse_dates=True)
            all_h1.append(existing_df)
            print(f"Loaded existing data: {len(existing_df)} rows")
        
        print(f"Building historical data for {self.instrument} from {start_date} to {end_date}...")
        
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                print(f"Fetching {current.date()}...", end="\r")
                ticks = self.fetcher.fetch_period(current.strftime("%Y-%m-%d"), current.strftime("%Y-%m-%d"))
                
                if not ticks.empty:
                    h1 = self.fetcher.aggregate_to_h1(ticks)
                    all_h1.append(h1)
                    
                    # Save periodically to avoid data loss
                    if len(all_h1) % 10 == 0:
                        self._save(all_h1)
                        
                time.sleep(0.2)  # Gentle rate limiting
            
            current += timedelta(days=1)
            
        self._save(all_h1)
        print(f"\nSUCCESS: Historical data saved to {self.file_path}")
        
    def _save(self, dataframes):
        if not dataframes:
            return
        result = pd.concat(dataframes)
        result = result[~result.index.duplicated(keep="last")]
        result.sort_index(inplace=True)
        result.to_csv(self.file_path)
        
    def update_daily_data(self):
        """Updates the CSV with the most recent data."""
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Updating data up to {today}...")
        self.build_period(yesterday, today)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    # Build 1 year of data for EURUSD
    builder_eurusd = HistoricalDataBuilder("EURUSD")
    builder_eurusd.build_period("2023-07-01", "2024-06-30")
    
    # Build 1 year of data for GBPUSD
    builder_gbpusd = HistoricalDataBuilder("GBPUSD")
    builder_gbpusd.build_period("2023-07-01", "2024-06-30")
    
    print("\nData building complete. Ready for institutional backtesting.")
