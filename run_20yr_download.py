import warnings
warnings.filterwarnings("ignore")
from build_historical_data import HistoricalDataBuilder
import time

print("=" * 60)
print("   EXPANDING DATA TO 20+ YEARS (2003 - 2024)")
print("=" * 60)

# 1. Build EURUSD (2003 to 2024)
print("\n[1/2] Expanding EURUSD historical data (2003-2024)...")
builder_eurusd = HistoricalDataBuilder("EURUSD")
builder_eurusd.build_period("2003-01-01", "2024-06-30")

print("\nEURUSD data expansion complete.\n")

# 2. Build GBPUSD (2003 to 2024)
print("[2/2] Expanding GBPUSD historical data (2003-2024)...")
builder_gbpusd = HistoricalDataBuilder("GBPUSD")
builder_gbpusd.build_period("2003-01-01", "2024-06-30")

print("\nGBPUSD data expansion complete.")
print("\nAll historical data (20+ years) successfully built and saved to D:\\forex_bot_data\\.")
