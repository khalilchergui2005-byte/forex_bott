code = r'''"""
spread_calculator.py - Dynamic Spread & Z-Score Calculator
Calculates rolling hedge ratio, spread, and Z-score for Pairs Trading.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = r"D:\forex_bot_data"

def load_pair_data(pair: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"{pair.lower()}_h1_real.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def calculate_spread(pair1="EURUSD", pair2="GBPUSD", lookback=100) -> pd.DataFrame:
    print(f"Calculating dynamic spread for {pair1} & {pair2}...")
    
    df1 = load_pair_data(pair1).add_prefix(f"{pair1}_")
    df2 = load_pair_data(pair2).add_prefix(f"{pair2}_")
    
    df = pd.merge(df1, df2, left_index=True, right_index=True, how="inner")
    
    s1 = df[f"{pair1}_close"]
    s2 = df[f"{pair2}_close"]
    
    # 1. Rolling Hedge Ratio (Beta) = Cov(Y, X) / Var(X)
    # Using rolling covariance and variance for vectorized speed
    cov = s1.rolling(lookback).cov(s2)
    var = s2.rolling(lookback).var()
    df["hedge_ratio"] = cov / var
    
    # 2. Calculate Spread = Y - (Beta * X)
    df["spread"] = s1 - (df["hedge_ratio"] * s2)
    
    # 3. Calculate Z-Score of the Spread
    rolling_mean = df["spread"].rolling(lookback).mean()
    rolling_std = df["spread"].rolling(lookback).std()
    df["z_score"] = (df["spread"] - rolling_mean) / (rolling_std + 1e-10)
    
    # Drop NaNs from rolling windows
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    
    df = calculate_spread()
    
    print("\n" + "="*50)
    print("   SPREAD & Z-SCORE CALCULATION RESULTS")
    print("="*50)
    print(f"Total data points: {len(df)}")
    print(f"Average Hedge Ratio: {df['hedge_ratio'].mean():.4f}")
    print(f"Z-Score Stats: Min={df['z_score'].min():.2f}, Max={df['z_score'].max():.2f}, Mean={df['z_score'].mean():.2f}")
    
    # Count how many times Z-Score exceeded +- 2.0 (Entry signals)
    entry_signals = df[(df["z_score"] > 2.0) | (df["z_score"] < -2.0)]
    print(f"Potential Entry Signals (|Z-Score| > 2.0): {len(entry_signals)}")
    
    print("\nSample Data (Tail):")
    print(df[[f"EURUSD_close", f"GBPUSD_close", "hedge_ratio", "spread", "z_score"]].tail(10))
'''
open("spread_calculator.py", "w", encoding="utf-8").write(code)
print("SUCCESS: spread_calculator.py created")

import ast
ast.parse(code)
print("AST OK")
