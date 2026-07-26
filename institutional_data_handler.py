"""
institutional_data_handler.py - Institutional Data Gateway
Loads real Dukascopy data, applies all 3 smart modules (Regime, Order Flow, Correlation),
and returns a fully enriched DataFrame ready for the smart engine.
"""
import pandas as pd
import os
from regime_detector import RegimeDetector
from order_flow_analyzer import OrderFlowAnalyzer
from correlation_matrix import CorrelationMatrix

DATA_DIR = r"D:\forex_bot_data"

def prepare_institutional_data(base_pair="EURUSD", correlated_pair="GBPUSD") -> pd.DataFrame:
    print("=" * 60)
    print("   PREPARING INSTITUTIONAL DATA GATEWAY")
    print("=" * 60)
    
    # 1. Correlation & Divergence (Merges the two pairs)
    print("\n[1/3] Applying Correlation & Divergence Matrix...")
    cm = CorrelationMatrix(base_pair=base_pair, correlated_pair=correlated_pair, lookback=50, data_dir=DATA_DIR)
    df = cm.analyze()
    
    # We need the base pair OHLCV columns for the next steps
    df["open"] = df[f"{base_pair}_open"]
    df["high"] = df[f"{base_pair}_high"]
    df["low"] = df[f"{base_pair}_low"]
    df["close"] = df[f"{base_pair}_close"]
    df["volume"] = df[f"{base_pair}_volume"]
    
    # 2. Order Flow Analysis (CVD, Exhaustion)
    print("[2/3] Applying Order Flow Analyzer...")
    ofa = OrderFlowAnalyzer()
    df = ofa.analyze(df)
    
    # 3. Regime Detection (Trend, Range, Volatile)
    print("[3/3] Applying Regime Detector...")
    rd = RegimeDetector()
    regime_df = rd.detect_history(df[["open", "high", "low", "close"]])
    
    # Merge regime columns into main df
    df["regime"] = regime_df["regime"]
    df["regime_name"] = regime_df["regime_name"]
    df["regime_confidence"] = regime_df["confidence"]
    
    # Final cleanup of NaNs created by rolling windows
    df.dropna(inplace=True)
    
    print("\nSUCCESS: Institutional Data Prepared.")
    print(f"Total enriched rows: {len(df)}")
    print(f"Columns available: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    df = prepare_institutional_data()
    
    print("\n=== Sample Enriched Institutional Data ===")
    print(df[["close", "volume", "regime_name", "cvd", "rolling_corr", "bull_div"]].tail(10))
