code = r'''"""
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
    df["hedge_ratio"] = cov / var
    
    # 2. Calculate Spread
    df["spread"] = s1 - (df["hedge_ratio"] * s2)
    
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
'''
open("spread_calculator.py", "w", encoding="utf-8").write(code)
print("SUCCESS: spread_calculator.py upgraded with Cointegration Gate")

# Update strategy to respect the gate
strat_code = r'''"""
strategy_pairs_trading.py - Statistical Arbitrage (Pairs Trading)
Logic: Trades the spread based on Z-Score mean reversion, ONLY if cointegration is valid.
"""
import pandas as pd
import numpy as np

def generate_signals(df, entry_z=1.7, exit_z=0.0, stop_z=4.5):
    df = df.copy()
    
    if "z_score" not in df.columns:
        df["Pairs_Signal"] = 0
        return df
        
    df["Pairs_Signal"] = 0
    in_trade = False
    trade_type = 0
    
    for i in range(len(df)):
        z = df["z_score"].iloc[i]
        coint_valid = df["coint_valid"].iloc[i] if "coint_valid" in df.columns else True
        
        if not in_trade:
            # Entry Conditions: Z-Score threshold AND Cointegration must be valid (Green Light)
            if coint_valid:
                if z > entry_z:
                    df.iloc[i, df.columns.get_loc("Pairs_Signal")] = -1
                    in_trade = True
                    trade_type = -1
                elif z < -entry_z:
                    df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 1
                    in_trade = True
                    trade_type = 1
        else:
            # Exit Condition (Z-Score reverts to exit_z) OR Cointegration breaks (Red Light)
            if (trade_type == 1 and z >= exit_z) or (trade_type == -1 and z <= exit_z) or not coint_valid:
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 0
                in_trade = False
                trade_type = 0
            # Stop Loss Condition
            elif (trade_type == 1 and z <= -stop_z) or (trade_type == -1 and z >= stop_z):
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = 0
                in_trade = False
                trade_type = 0
            else:
                df.iloc[i, df.columns.get_loc("Pairs_Signal")] = trade_type
                
    return df
'''
open("strategy_pairs_trading.py", "w", encoding="utf-8").write(strat_code)
print("SUCCESS: strategy_pairs_trading.py updated to respect Cointegration Gate")

import ast
ast.parse(open("spread_calculator.py", encoding="utf-8").read())
ast.parse(open("strategy_pairs_trading.py", encoding="utf-8").read())
print("AST OK for both files")
