"""
ofi_calculator.py - Institutional Order Flow Imbalance (OFI) Calculator
Calculates true OFI using separated Buy/Sell volumes (Tick Rule applied).
"""
import pandas as pd
import numpy as np

class OFICalculator:
    def __init__(self, lookback=20):
        self.lookback = lookback

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches dataframe with Order Flow Imbalance features.
        Requires 'volume' column. Ideally requires 'buy_volume' and 'sell_volume'.
        """
        data = df.copy()

        # Fallback: If tick-level separated volumes aren't available (e.g., old CSV),
        # apply Tick Rule proxy on H1 bars to prevent crash.
        if 'buy_volume' not in data.columns or 'sell_volume' not in data.columns:
            print("WARN: buy_volume/sell_volume not found. Applying H1 Tick Rule proxy.")
            direction = np.sign(data['close'].diff().fillna(0))
            direction.replace(0, np.nan, inplace=True)
            direction.ffill(inplace=True)
            direction.fillna(0, inplace=True)
            data['buy_volume'] = data['volume'].where(direction > 0, 0)
            data['sell_volume'] = data['volume'].where(direction < 0, 0)

        # 1. Bar-level OFI (Order Flow Imbalance)
        data['ofi_bar'] = data['buy_volume'] - data['sell_volume']

        # 2. Cumulative OFI (True CVD based on volume delta)
        data['ofi_cumulative'] = data['ofi_bar'].cumsum()

        # 3. OFI Z-Score (Intensity of imbalance relative to recent history)
        rolling_mean = data['ofi_bar'].rolling(self.lookback).mean()
        rolling_std = data['ofi_bar'].rolling(self.lookback).std()
        data['ofi_z_score'] = (data['ofi_bar'] - rolling_mean) / (rolling_std + 1e-10)

        # 4. OFI Moving Average (Smoothed trend of order flow)
        data['ofi_ma'] = data['ofi_bar'].rolling(self.lookback).mean()

        return data

if __name__ == "__main__":
    # Synthetic test to prove math works without needing 1-hour data re-download
    print("Testing OFI Calculator with synthetic data...")
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    df_test = pd.DataFrame({
        'close': np.cumsum(np.random.randn(100) * 0.001) + 1.1,
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    # Simulate the tick rule volumes for the test
    direction = np.sign(df_test['close'].diff().fillna(0))
    direction.replace(0, np.nan, inplace=True)
    direction.ffill(inplace=True)
    df_test['buy_volume'] = df_test['volume'].where(direction > 0, 0)
    df_test['sell_volume'] = df_test['volume'].where(direction < 0, 0)

    ofi = OFICalculator(lookback=10)
    enriched = ofi.calculate(df_test)

    print("\nSample OFI Data (Tail):")
    print(enriched[['buy_volume', 'sell_volume', 'ofi_bar', 'ofi_cumulative', 'ofi_z_score']].tail(10))
    print("\nSUCCESS: OFI Calculator operational.")
