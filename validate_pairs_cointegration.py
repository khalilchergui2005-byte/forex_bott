"""
validate_pairs_cointegration.py - Rolling Cointegration Test
Checks if the statistical relationship between EURUSD and GBPUSD holds over time.
"""
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\forex_bot_data"

def load_pair_data(pair: str) -> pd.Series:
    file_path = os.path.join(DATA_DIR, f"{pair.lower()}_h1_real.csv")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df["close"]

def test_rolling_cointegration(pair1="EURUSD", pair2="GBPUSD", window=1000):
    print(f"Testing Rolling Cointegration ({window}-hour window) between {pair1} and {pair2}...")
    
    s1 = load_pair_data(pair1)
    s2 = load_pair_data(pair2)
    
    df = pd.DataFrame({pair1: s1, pair2: s2}).dropna()
    
    p_values = []
    dates = []
    
    # Loop through the data in steps to calculate rolling p-value
    # We use steps of 500 to speed up the process over 33,000 rows
    for i in range(window, len(df), 500):
        end_idx = i
        start_idx = i - window
        
        y = df[pair1].iloc[start_idx:end_idx]
        x = df[pair2].iloc[start_idx:end_idx]
        
        # OLS Regression
        X = sm.add_constant(x)
        try:
            model = sm.OLS(y, X).fit()
            residuals = y - model.predict(X)
            
            # ADF Test on residuals
            adf_result = adfuller(residuals, autolag="AIC")
            p_val = adf_result[1]
            
            p_values.append(p_val)
            dates.append(df.index[end_idx])
        except:
            pass
            
    result_df = pd.DataFrame({"date": dates, "p_value": p_values}).set_index("date")
    
    # Analyze results
    valid_periods = result_df[result_df["p_value"] < 0.05]
    broken_periods = result_df[result_df["p_value"] >= 0.05]
    
    print("\n" + "="*50)
    print("   ROLLING COINTEGRATION TEST RESULTS")
    print("="*50)
    print(f"Total windows tested: {len(result_df)}")
    print(f"Windows with VALID cointegration (p < 0.05): {len(valid_periods)} ({len(valid_periods)/len(result_df)*100:.1f}%)")
    print(f"Windows with BROKEN cointegration (p >= 0.05): {len(broken_periods)} ({len(broken_periods)/len(result_df)*100:.1f}%)")
    
    if len(broken_periods) > 0:
        print("\nWARNING: Cointegration broke in the following periods:")
        print(broken_periods.head())
    else:
        print("\nConclusion: The pairs remained cointegrated throughout the 5 years.")
    print("="*50)

if __name__ == "__main__":
    test_rolling_cointegration()
