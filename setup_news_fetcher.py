import subprocess
subprocess.check_call(["pip", "install", "finnhub-python", "-q"])

code = r'''"""
economic_calendar_fetcher.py - Institutional News Data Fetcher
Fetches historical economic events (Actual vs Forecast) from Finnhub.
"""
import finnhub
import pandas as pd
from datetime import datetime, timedelta
import os
import time

class EconomicCalendarFetcher:
    def __init__(self, api_key: str):
        self.client = finnhub.Client(api_key=api_key)
        
    def fetch_economic_events(self, start_date: str, end_date: str, country="US"):
        """
        Fetches economic calendar events.
        start_date/end_date format: "YYYY-MM-DD"
        """
        print(f"Fetching economic events for {country} from {start_date} to {end_date}...")
        
        # Finnhub economic calendar endpoint
        # We fetch in chunks to avoid API limits
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_events = []
        current = start
        
        while current < end:
            next_date = current + timedelta(days=30)  # Fetch 30 days at a time
            if next_date > end:
                next_date = end
                
            try:
                # Fetch economic calendar
                events = self.client.economic_calendar(start_date=current.strftime("%Y-%m-%d"), 
                                                       end_date=next_date.strftime("%Y-%m-%d"))
                
                if events:
                    for event in events:
                        # Filter for high impact (NFP, CPI, FOMC, GDP, etc.)
                        if event.get("country", "") == country:
                            all_events.append({
                                "timestamp": pd.to_datetime(event.get("time", 0), unit="s"),
                                "country": event.get("country"),
                                "event": event.get("event"),
                                "impact": event.get("impact"),
                                "actual": event.get("actual"),
                                "forecast": event.get("forecast"),
                                "prev": event.get("prev")
                            })
            except Exception as e:
                print(f"Error fetching {current.date()}: {e}")
                
            current = next_date
            time.sleep(1)  # Rate limiting for free tier (60 calls/min)
            
        df = pd.DataFrame(all_events)
        if not df.empty:
            # Filter for high impact events only (Finnhub uses 'high', 'medium', 'low')
            df = df[df["impact"] == "high"]
            df.dropna(subset=["actual", "forecast"], inplace=True)
            df.sort_values("timestamp", inplace=True)
            
        return df

if __name__ == "__main__":
    # ENTER YOUR FINNHUB API KEY HERE
    API_KEY = "YOUR_API_KEY_HERE"
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please register at finnhub.io and replace YOUR_API_KEY_HERE with your actual key.")
    else:
        fetcher = EconomicCalendarFetcher(API_KEY)
        
        # Fetch 5 years of US high-impact news
        news_df = fetcher.fetch_economic_events("2019-01-01", "2024-06-30", country="US")
        
        if not news_df.empty:
            print(f"\nTotal high-impact news events fetched: {len(news_df)}")
            print("\nSample News Data:")
            print(news_df.head())
            
            # Save to CSV
            os.makedirs("data", exist_ok=True)
            news_df.to_csv("data/us_high_impact_news.csv", index=False)
            print("\nSaved to data/us_high_impact_news.csv")
        else:
            print("No news data fetched.")
'''
open("economic_calendar_fetcher.py", "w", encoding="utf-8").write(code)
print("SUCCESS: economic_calendar_fetcher.py created")

import ast
ast.parse(code)
print("AST OK")
