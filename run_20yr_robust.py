import os
import pandas as pd
from datetime import datetime, timedelta
import time
import requests
import lzma
import struct
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\forex_bot_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_last_date(filepath):
    """Reads the last line of the CSV to resume download."""
    if not os.path.exists(filepath):
        return None
    try:
        # Efficient way to read the last line in Windows
        with open(filepath, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        return pd.to_datetime(last_line.split(',')[0])
    except:
        return None

class RobustDukascopyFetcher:
    BASE_URL = "https://datafeed.dukascopy.com/datafeed/{instrument}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    
    def __init__(self, instrument="EURUSD"):
        self.instrument = instrument.upper()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        
    def _fetch_hour_ticks(self, date, hour):
        url = self.BASE_URL.format(
            instrument=self.instrument,
            year=date.year,
            month=date.month - 1,
            day=date.day,
            hour=hour
        )
        
        for attempt in range(5): # 5 retries per hour
            try:
                response = self.session.get(url, timeout=15)
                if response.status_code == 200 and len(response.content) > 0:
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
                elif response.status_code == 404:
                    return pd.DataFrame()
            except Exception:
                time.sleep(2) # Wait before retry
        
        return pd.DataFrame() # Failed after 5 attempts
        
    def fetch_day(self, date):
        all_ticks = []
        for hour in range(24):
            df = self._fetch_hour_ticks(date, hour)
            if not df.empty:
                all_ticks.append(df)
        if not all_ticks:
            return pd.DataFrame()
        result = pd.concat(all_ticks, ignore_index=True)
        result["mid"] = (result["ask"] + result["bid"]) / 2
        result["spread"] = result["ask"] - result["bid"]
        result.set_index("timestamp", inplace=True)
        return result

    def aggregate_to_h1(self, ticks):
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

def build_historical_data(instrument, start_date_str, end_date_str):
    file_path = os.path.join(DATA_DIR, f"{instrument.lower()}_h1_real.csv")
    
    last_date = get_last_date(file_path)
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    
    if last_date:
        print(f"[{instrument}] Resuming from last recorded date: {last_date.date()}")
        start_date = last_date + timedelta(days=1)
    else:
        print(f"[{instrument}] Starting fresh download.")
        with open(file_path, "w") as f:
            f.write("timestamp,open,high,low,close,volume,avg_spread\n")
            
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current = start_date
    
    fetcher = RobustDukascopyFetcher(instrument)
    days_processed = 0
    
    print(f"[{instrument}] Downloading from {current.date()} to {end_date.date()}...")
    
    while current <= end_date:
        if current.weekday() < 5:
            ticks = fetcher.fetch_day(current)
            if not ticks.empty:
                h1 = fetcher.aggregate_to_h1(ticks)
                h1.to_csv(file_path, mode="a", header=False)
                
            days_processed += 1
            if days_processed % 30 == 0:
                print(f"[{instrument}] Processed {days_processed} days... Last date: {current.date()}")
                
            time.sleep(0.05)
            
        current += timedelta(days=1)
        
    print(f"SUCCESS: [{instrument}] historical data complete up to {end_date.date()}.")

if __name__ == "__main__":
    print("========================================")
    print("   20-YEAR DUKASCOPY DOWNLOAD (ROBUST)")
    print("========================================")
    
    build_historical_data("EURUSD", "2003-01-01", "2024-06-30")
    build_historical_data("GBPUSD", "2003-01-01", "2024-06-30")
    
    print("ALL DOWNLOADS COMPLETE.")
