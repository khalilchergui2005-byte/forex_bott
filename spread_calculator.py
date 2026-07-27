"""
spread_calculator.py - Dynamic Spread, Z-Score & Cointegration Gate
Calculates rolling hedge ratio, spread, Z-score, and Rolling P-Value.
"""
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\forex_bot_data"

def load_pair_data(pair: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"{pair.lower()}_h1_real.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df[["open", "high", "low", "close", "volume"]].astype(float)

def calculate_spread(pair1="EURUSD", pair2="GBPUSD", lookback=140, coint_window=1000) -> pd.DataFrame:
    print(f"Calculating dynamic spread & Cointegration Gate for {pair1} & {pair2}...")
    
    df1 = load_pair_data(pair1).add_prefix(f"{pair1}_")
    df2 = load_pair_data(pair2).add_prefix(f"{pair2}_")
    
    df = pd.merge(df1, df2, left_index=True, right_index=True, how="inner")
    
    s1 = df[f"{pair1}_close"]
    s2 = df[f"{pair2}_close"]
    
    # 1. Rolling Hedge Ratio (Beta)
    cov = s1.rolling(lookback).cov(s2)
    var = s2.rolling(lookback).var()
    df["hedge_ratio"] = (cov / (var + 1e-10)).clip(0.5, 1.5)  # FIX: Institutional Clipping to prevent math blowups
    
    # 2. Calculate Spread
    df["spread"] = s1 - (df["hedge_ratio"] * s2)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 3. Calculate Z-Score
    rolling_mean = df["spread"].rolling(lookback).mean()
    rolling_std = df["spread"].rolling(lookback).std()
    df["z_score"] = (df["spread"] - rolling_mean) / (rolling_std + 1e-10)
    
    # 4. Rolling Cointegration P-Value (Traffic Light)
    # Calculated every 24 hours to maintain performance, then forward-filled
    print("Calculating Rolling P-Value (This may take a moment)...")
    p_values = []
    for i in range(len(df)):
        if i < coint_window:
            p_values.append(np.nan)
        else:
            # Calculate every 24 bars to speed up, otherwise reuse previous
            if i % 24 == 0 or i == len(df) - 1:
                y = s1.iloc[i-coint_window:i]
                x = s2.iloc[i-coint_window:i]
                X = sm.add_constant(x)
                try:
                    model = sm.OLS(y, X).fit()
                    residuals = y - model.predict(X)
                    adf_result = adfuller(residuals, autolag="AIC")
                    p_values.append(adf_result[1])
                except:
                    p_values.append(np.nan)
            else:
                p_values.append(p_values[-1])
                
    df["coint_p_value"] = p_values
    df["coint_valid"] = df["coint_p_value"] < 0.05
    
    # Drop NaNs from rolling windows
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    df = calculate_spread()
    print("\nSample Data with Cointegration Gate:")
    print(df[[f"EURUSD_close", f"GBPUSD_close", "z_score", "coint_p_value", "coint_valid"]].tail(10))
