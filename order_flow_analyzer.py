"""
order_flow_analyzer.py - Institutional Order Flow Analyzer
Analyzes REAL volume data to extract Cumulative Volume Delta (CVD) and Exhaustion.
"""
import pandas as pd
import numpy as np

class OrderFlowAnalyzer:
    def __init__(self, vol_lookback=100, spike_mult=1.5, exhaustion_body_ratio=0.4):
        self.vol_lookback = vol_lookback
        self.spike_mult = spike_mult
        self.exhaustion_body_ratio = exhaustion_body_ratio

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches OHLCV dataframe with Order Flow metrics.
        Requires 'open', 'high', 'low', 'close', 'volume' columns.
        """
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column for Order Flow analysis.")
            
        data = df.copy()
        
        # 1. Calculate Volume Delta (Buy vs Sell pressure proxy)
        # If close > open, volume is considered buy volume. If close < open, sell volume.
        direction = np.sign(data["close"] - data["open"])
        data["delta"] = direction * data["volume"]
        
        # 2. Cumulative Volume Delta (CVD)
        data["cvd"] = data["delta"].cumsum()
        
        # 3. Volume Spike Detection (Relative to recent history)
        data["vol_ma"] = data["volume"].rolling(self.vol_lookback, min_periods=20).mean()
        data["vol_spike"] = data["volume"] > (data["vol_ma"] * self.spike_mult)
        
        # 4. Exhaustion / Absorption Detection
        # High volume but small body (price didn't move much despite huge volume)
        body = abs(data["close"] - data["open"])
        range_ = data["high"] - data["low"]
        data["body_ratio"] = body / (range_ + 1e-10)
        
        data["exhaustion"] = data["vol_spike"] & (data["body_ratio"] < self.exhaustion_body_ratio)
        
        # Clean up temporary columns used for logic
        data.drop(columns=["vol_ma"], inplace=True)
        
        return data

# Test the analyzer on our real fetched data
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    print("Testing OrderFlowAnalyzer on fetched Dukascopy H1 data...")
    
    # Fetch 1 day of data
    from dukascopy_fetcher import DukascopyFetcher
    fetcher = DukascopyFetcher("EURUSD")
    ticks = fetcher.fetch_period("2024-06-26", "2024-06-26")
    
    if ticks.empty:
        print("Failed to fetch ticks for testing.")
    else:
        h1 = fetcher.aggregate_to_h1(ticks)
        
        analyzer = OrderFlowAnalyzer()
        h1_enriched = analyzer.analyze(h1)
        
        print("\n=== H1 Data with Order Flow Metrics ===")
        print(h1_enriched[["close", "volume", "delta", "cvd", "exhaustion"]].head(10))
        
        print("\n=== Exhaustion Signals ===")
        exhaustion_signals = h1_enriched[h1_enriched["exhaustion"] == True]
        if not exhaustion_signals.empty:
            print(exhaustion_signals[["close", "volume", "delta"]])
        else:
            print("No exhaustion signals in this day (normal for 1 day).")
