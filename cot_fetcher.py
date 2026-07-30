"""
cot_fetcher.py - CFTC Commitments of Traders (COT) Data Fetcher
Uses exact verified column names from CFTC legacy futures reports.
"""
import pandas as pd
import numpy as np
import cot_reports as cot
import os
import time

class COTFetcher:
    MARKET_NAME = "EURO FX - CHICAGO MERCANTILE EXCHANGE"
    DATE_COL = "As_of_Date_in_Form_YYYY-MM-DD"
    MARKET_COL = "Market_and_Exchange_Names"
    LONG_COL = "Noncommercial_Positions-Long_(All)"
    SHORT_COL = "Noncommercial_Positions-Short_(All)"
    
    def fetch_cot_data(self, start_year=2019, end_year=2024) -> pd.DataFrame:
        print("Fetching COT data from CFTC...")
        all_data = []
        
        for year in range(start_year, end_year + 1):
            print(f"  Fetching {year}...")
            try:
                time.sleep(2) # Delay to prevent timeout
                df = cot.cot_year(year=year, store_txt=False, verbose=False)
                if df is not None and not df.empty:
                    # Standardize column names (spaces to underscores)
                    df.columns = df.columns.str.strip().str.replace(' ', '_')
                    all_data.append(df)
                    print(f"  Success: {len(df)} records for {year}")
                else:
                    print(f"  Warning: No data for {year}")
            except Exception as e:
                print(f"  Error fetching {year}: {e}")
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Filter for Euro FX
        df = df[df[self.MARKET_COL].str.strip() == self.MARKET_NAME].copy()
        
        if df.empty:
            print("  Warning: Euro FX data not found")
            return pd.DataFrame()
        
        # Clean and process using exact verified columns
        df[self.DATE_COL] = pd.to_datetime(df[self.DATE_COL])
        df.set_index(self.DATE_COL, inplace=True)
        df.sort_index(inplace=True)
        
        # Extract Non-Commercial positions (All)
        df[self.LONG_COL] = pd.to_numeric(df[self.LONG_COL], errors='coerce')
        df[self.SHORT_COL] = pd.to_numeric(df[self.SHORT_COL], errors='coerce')
        
        # Calculate Net Position
        df['COT_Net'] = df[self.LONG_COL] - df[self.SHORT_COL]
        
        # Calculate Z-Score (52-week lookback = 1 year)
        df['COT_Z_Score'] = (df['COT_Net'] - df['COT_Net'].rolling(52).mean()) / (df['COT_Net'].rolling(52).std() + 1e-10)
        
        # Week-over-week change
        df['COT_Change'] = df['COT_Net'].diff(1)
        
        print(f"\n  Successfully processed {len(df)} COT reports.")
        
        return df[['COT_Net', 'COT_Z_Score', 'COT_Change']]

if __name__ == "__main__":
    fetcher = COTFetcher()
    cot_df = fetcher.fetch_cot_data(2019, 2024)
    
    if not cot_df.empty:
        print("\nSample COT Data:")
        print(cot_df.tail(10))
        
        os.makedirs("data", exist_ok=True)
        cot_df.to_csv("data/cot_data.csv")
        print("\nSaved to data/cot_data.csv")
    else:
        print("No COT data fetched.")
