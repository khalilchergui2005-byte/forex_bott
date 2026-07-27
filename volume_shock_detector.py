"""
volume_shock_detector.py - Institutional Liquidity Shock Detector
Calculates Volume Z-Score and detects sudden liquidity injections (News/Fat Finger).
"""
import pandas as pd
import numpy as np

class VolumeShockDetector:
    def __init__(self, lookback=24, z_threshold=3.0, body_ratio=0.6):
        self.lookback = lookback  # 24 hours to capture daily volume baseline
        self.z_threshold = z_threshold
        self.body_ratio = body_ratio

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriches dataframe with Volume Z-Score and Shock signals.
        Requires 'open', 'high', 'low', 'close', 'volume' columns.
        """
        if "volume" not in df.columns:
            raise ValueError("DataFrame must contain 'volume' column for Shock Detection.")
            
        data = df.copy()
        
        # 1. Calculate Volume Z-Score (Relative to past 24 hours)
        vol_mean = data["volume"].rolling(self.lookback).mean()
        vol_std = data["volume"].rolling(self.lookback).std()
        data["vol_z_score"] = (data["volume"] - vol_mean) / (vol_std + 1e-10)
        
        # 2. Calculate Candle Body Ratio (Directional Conviction)
        body = abs(data["close"] - data["open"])
        range_ = data["high"] - data["low"]
        data["body_ratio"] = body / (range_ + 1e-10)
        
        # 3. Detect Liquidity Shock
        # High Volume AND Strong Directional Body
        data["vol_shock"] = (data["vol_z_score"] > self.z_threshold) & (data["body_ratio"] > self.body_ratio)
        
        # 4. Determine Shock Direction (1 for Bullish, -1 for Bearish)
        data["shock_dir"] = 0
        data.loc[data["vol_shock"] & (data["close"] > data["open"]), "shock_dir"] = 1
        data.loc[data["vol_shock"] & (data["close"] < data["open"]), "shock_dir"] = -1
        
        return data

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import os
    
    DATA_DIR = r"D:\forex_bot_data"
    print("Testing Volume Shock Detector on EURUSD H1 (5 Years)...")
    
    file_path = os.path.join(DATA_DIR, "eurusd_h1_real.csv")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    detector = VolumeShockDetector()
    enriched_df = detector.detect(df)
    
    shocks = enriched_df[enriched_df["vol_shock"] == True]
    
    print("\n" + "="*50)
    print("   LIQUIDITY SHOCK DETECTION RESULTS")
    print("="*50)
    print(f"Total H1 candles analyzed: {len(enriched_df)}")
    print(f"Total Liquidity Shocks Detected (Z > 3.0): {len(shocks)}")
    
    if len(shocks) > 0:
        print("\nSample Shocks (Sudden Volume Injections):")
        print(shocks[["close", "volume", "vol_z_score", "shock_dir"]].head(10))
    print("="*50)
