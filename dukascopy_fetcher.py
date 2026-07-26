"""
dukascopy_fetcher.py - Institutional Tick Data Fetcher
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import lzma
import struct

class DukascopyFetcher:
    BASE_URL = "https://datafeed.dukascopy.com/datafeed/{instrument}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    
    def __init__(self, instrument="EURUSD"):
        self.instrument = instrument.upper()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        })
    
    def _fetch_hour_ticks(self, date: datetime, hour: int) -> pd.DataFrame:
        url = self.BASE_URL.format(
            instrument=self.instrument,
            year=date.year,
            month=date.month - 1,
            day=date.day,
            hour=hour
        )
        
        try:
            response = self.session.get(url, timeout=15)
            if response.status_code != 200 or len(response.content) == 0:
                return pd.DataFrame()
            
            decompressed = lzma.decompress(response.content)
            tick_size = 20
            num_ticks = len(decompressed) // tick_size
            
            if num_ticks == 0:
                return pd.DataFrame()
                
            fmt = ">iiid"
            ticks = struct.iter_unpack(fmt, decompressed)
            
            data = []
            base_time = datetime(date.year, date.month, date.day, hour)
            
            for t in ticks:
                ms_offset, ask, bid, volume = t
                timestamp = base_time + timedelta(milliseconds=ms_offset)
                data.append({
                    "timestamp": timestamp,
                    "ask": ask / 100000.0,
                    "bid": bid / 100000.0,
                    "volume": volume
                })
                
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return pd.DataFrame()
    
    def fetch_period(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_ticks = []
        current = start
        
        while current <= end:
            if current.weekday() < 5:
                print(f"Fetching {self.instrument} ticks for {current.date()}...")
                for hour in range(24):
                    df = self._fetch_hour_ticks(current, hour)
                    if not df.empty:
                        all_ticks.append(df)
                    time.sleep(0.1)  # Gentle rate limiting
            current += timedelta(days=1)
        
        if not all_ticks:
            return pd.DataFrame()
        
        result = pd.concat(all_ticks, ignore_index=True)
        result["mid"] = (result["ask"] + result["bid"]) / 2
        result["spread"] = result["ask"] - result["bid"]
        result.set_index("timestamp", inplace=True)
        return result
    
    def aggregate_to_h1(self, ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame()
        
        h1 = pd.DataFrame()
        h1["open"] = ticks["mid"].resample("1h").first()
        h1["high"] = ticks["mid"].resample("1h").max()
        h1["low"] = ticks["mid"].resample("1h").min()
        h1["close"] = ticks["mid"].resample("1h").last()
        h1["volume"] = ticks["volume"].resample("1h").sum()
        h1["avg_spread"] = ticks["spread"].resample("1h").mean()
        
        h1.dropna(inplace=True)
        return h1

if __name__ == "__main__":
    fetcher = DukascopyFetcher("EURUSD")
    
    print("Fetching 1 day of Dukascopy EURUSD tick data...")
    ticks = fetcher.fetch_period("2024-06-26", "2024-06-26")
    
    if not ticks.empty:
        print(f"\nTotal ticks fetched: {len(ticks)}")
        print(f"Date range: {ticks.index[0]} to {ticks.index[-1]}")
        
        print(f"\nAggregating to H1 candles...")
        h1 = fetcher.aggregate_to_h1(ticks)
        print(f"Total H1 candles: {len(h1)}")
        print(f"\nSample H1 with REAL Volume:")
        print(h1.head())
    else:
        print("No data fetched. Check network connection or URL format.")
