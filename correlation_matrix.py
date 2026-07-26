"""
correlation_matrix.py - Institutional Cross-Asset Correlation & Divergence
Reads real H1 data for multiple pairs, calculates rolling correlation, and detects divergence.
"""
import pandas as pd
import numpy as np
import os

class CorrelationMatrix:
    def __init__(self, base_pair="EURUSD", correlated_pair="GBPUSD", lookback=50, data_dir="data"):
        self.base_pair = base_pair
        self.correlated_pair = correlated_pair
        self.lookback = lookback
        self.data_dir = data_dir
        
    def _load_data(self, pair: str) -> pd.DataFrame:
        file_path = os.path.join(self.data_dir, f"{pair.lower()}_h1_real.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found for {pair}: {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float)
        
    def analyze(self) -> pd.DataFrame:
        """Merges pairs, calculates rolling correlation, and detects divergence."""
        # 1. Load and align data
        df_base = self._load_data(self.base_pair).add_prefix(f"{self.base_pair}_")
        df_corr = self._load_data(self.correlated_pair).add_prefix(f"{self.correlated_pair}_")
        
        # Inner join to keep only matching timestamps
        merged = pd.merge(df_base, df_corr, left_index=True, right_index=True, how="inner")
        
        # 2. Calculate Returns
        base_ret = merged[f"{self.base_pair}_close"].pct_change()
        corr_ret = merged[f"{self.correlated_pair}_close"].pct_change()
        
        # 3. Rolling Correlation
        merged["rolling_corr"] = base_ret.rolling(self.lookback).corr(corr_ret)
        
        # 4. Divergence Detection (Rolling Highs/Lows over lookback)
        base_high = merged[f"{self.base_pair}_close"].rolling(self.lookback).max()
        base_low = merged[f"{self.base_pair}_close"].rolling(self.lookback).min()
        corr_high = merged[f"{self.correlated_pair}_close"].rolling(self.lookback).max()
        corr_low = merged[f"{self.correlated_pair}_close"].rolling(self.lookback).min()
        
        # Bullish Divergence: Base pair breaks high, but correlated pair fails to break high
        merged["bull_div"] = (merged[f"{self.base_pair}_close"] >= base_high) & (merged[f"{self.correlated_pair}_close"] < corr_high)
        
        # Bearish Divergence: Base pair breaks low, but correlated pair fails to break low
        merged["bear_div"] = (merged[f"{self.base_pair}_close"] <= base_low) & (merged[f"{self.correlated_pair}_close"] > corr_low)
        
        merged.dropna(inplace=True)
        return merged

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    print("=" * 60)
    print("   CORRELATION & DIVERGENCE ANALYSIS (EURUSD vs GBPUSD)")
    print("=" * 60)
    
    cm = CorrelationMatrix(base_pair="EURUSD", correlated_pair="GBPUSD", lookback=50)
    df = cm.analyze()
    
    print("Total aligned rows:", len(df))
    print("Average Rolling Correlation:", round(df['rolling_corr'].mean(), 2))
    
    print("\n=== Divergence Signals ===")
    bull_divs = df[df["bull_div"] == True]
    bear_divs = df[df["bear_div"] == True]
    
    print("Bullish Divergences (False Breakout Up):", len(bull_divs))
    print("Bearish Divergences (False Breakout Down):", len(bear_divs))
    
    if not bull_divs.empty:
        print("\nSample Bullish Divergence (EURUSD made high, GBPUSD didn't):")
        print(bull_divs[["EURUSD_close", "GBPUSD_close", "rolling_corr"]].head())
