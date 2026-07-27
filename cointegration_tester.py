"""
cointegration_tester.py - Engle-Granger Cointegration Test
Verifies if EURUSD and GBPUSD are cointegrated over 5 years of real data.
"""
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

DATA_DIR = r"D:\forex_bot_data"

def load_pair_data(pair: str) -> pd.Series:
    file_path = os.path.join(DATA_DIR, f"{pair.lower()}_h1_real.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df["close"]

def test_cointegration(pair1="EURUSD", pair2="GBPUSD"):
    print(f"Testing Cointegration between {pair1} and {pair2}...")
    
    s1 = load_pair_data(pair1)
    s2 = load_pair_data(pair2)
    
    # Align timestamps
    df = pd.DataFrame({pair1: s1, pair2: s2}).dropna()
    
    # 1. OLS Regression to find Hedge Ratio
    X = sm.add_constant(df[pair2])
    model = sm.OLS(df[pair1], X).fit()
    hedge_ratio = model.params[pair2]
    intercept = model.params["const"]
    
    # 2. Calculate Residuals (Spread)
    residuals = df[pair1] - (hedge_ratio * df[pair2] + intercept)
    
    # 3. ADF Test on Residuals
    adf_result = adfuller(residuals, autolag="AIC")
    p_value = adf_result[1]
    
    print("\n" + "="*50)
    print("   COINTEGRATION TEST RESULTS (Engle-Granger)")
    print("="*50)
    print(f"Data points used: {len(df)}")
    print(f"Hedge Ratio (Beta): {hedge_ratio:.4f}")
    print(f"Intercept (Alpha): {intercept:.4f}")
    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-Value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("\nConclusion: REJECTED Null Hypothesis. The pairs are COINTEGRATED.")
        print("-> The spread is stationary. Pairs Trading is viable.")
    else:
        print("\nConclusion: FAILED to reject Null Hypothesis. The pairs are NOT cointegrated.")
        print("-> The spread has a unit root. Pairs Trading is risky.")
    print("="*50)
    
    return hedge_ratio, p_value

if __name__ == "__main__":
    test_cointegration()
