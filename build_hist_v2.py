import os
import pandas as pd
from datetime import datetime, timedelta
import time
from dukascopy_fetcher import DukascopyFetcher

# مسار التخزين الجديد في القرص D
DATA_DIR = r"D:\forex_bot_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class HistoricalDataBuilderV2:
    def __init__(self, instrument="EURUSD"):
        self.instrument = instrument
        self.fetcher = DukascopyFetcher(instrument)
        self.file_path = os.path.join(DATA_DIR, f"{instrument.lower()}_h1_real.csv")
        
    def build_period(self, start_date: str, end_date: str):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # كتابة الهيدر إذا كان الملف جديداً
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                f.write("timestamp,open,high,low,close,volume,avg_spread\n")
        
        print(f"Building historical data for {self.instrument} from {start_date} to {end_date}...")
        print(f"Saving to: {self.file_path}")
        
        current = start
        days_processed = 0
        
        while current <= end:
            if current.weekday() < 5:  # Skip weekends
                # جلب بيانات يوم واحد
                ticks = self.fetcher.fetch_period(current.strftime("%Y-%m-%d"), current.strftime("%Y-%m-%d"))
                
                if not ticks.empty:
                    # تجميعها على H1 فوراً
                    h1 = self.fetcher.aggregate_to_h1(ticks)
                    
                    # إلحاقها بالملف مباشرة (Append Mode) بدون تحميل البيانات السابقة في الذاكرة
                    h1.to_csv(self.file_path, mode="a", header=False)
                    
                days_processed += 1
                if days_processed % 10 == 0:
                    print(f"Processed {days_processed} days... Last date: {current.date()}")
                    
                time.sleep(0.1) # تجنب الحظر
            
            current += timedelta(days=1)
            
        print(f"\nSUCCESS: Historical data for {self.instrument} saved to {self.file_path}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    print("=" * 60)
    print("   BUILDING 5-YEAR INSTITUTIONAL DATA (D: Drive - Append Mode)")
    print("=" * 60)
    
    # Build EURUSD
    print("\n[1/2] Building EURUSD historical data...")
    builder_eurusd = HistoricalDataBuilderV2("EURUSD")
    builder_eurusd.build_period("2019-01-01", "2024-06-30")
    
    # Build GBPUSD
    print("\n[2/2] Building GBPUSD historical data...")
    builder_gbpusd = HistoricalDataBuilderV2("GBPUSD")
    builder_gbpusd.build_period("2019-01-01", "2024-06-30")
    
    print("\nAll historical data successfully built.")
