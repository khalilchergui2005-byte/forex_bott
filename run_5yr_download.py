import warnings
warnings.filterwarnings("ignore")
from build_historical_data import HistoricalDataBuilder
import time

print("=" * 60)
print("   BUILDING 5-YEAR INSTITUTIONAL DATA (2019 - 2024)")
print("=" * 60)

# 1. Build EURUSD (5 years)
print("\n[1/2] Building EURUSD historical data...")
builder_eurusd = HistoricalDataBuilder("EURUSD")
builder_eurusd.build_period("2019-01-01", "2024-06-30")

print("\nEURUSD data building complete.\n")

# 2. Build GBPUSD (5 years)
print("[2/2] Building GBPUSD historical data...")
builder_gbpusd = HistoricalDataBuilder("GBPUSD")
builder_gbpusd.build_period("2019-01-01", "2024-06-30")

print("\nGBPUSD data building complete.")
print("\nAll historical data successfully built and saved to data/ directory.")
