import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
from order_flow_analyzer import OrderFlowAnalyzer
from regime_detector import RegimeDetector

DATA_DIR_D = r"D:\forex_bot_data"
DATA_DIR_L = "data"

print("=" * 60)
print("   RESTORING 20-YEAR DATA (Left Join + ffill)")
print("=" * 60)

# 1. Load Raw Data (The 20-Year Foundation)
print("\n[1/5] Loading Raw Tick Data (20 Years)...")
df_base = pd.read_csv(os.path.join(DATA_DIR_D, "eurusd_h1_real.csv"), index_col=0, parse_dates=True)
df_corr = pd.read_csv(os.path.join(DATA_DIR_D, "gbpusd_h1_real.csv"), index_col=0, parse_dates=True)
df = pd.merge(df_base.add_prefix("EURUSD_"), df_corr.add_prefix("GBPUSD_"), left_index=True, right_index=True, how="inner")

# Standard columns
df["open"] = df["EURUSD_open"]
df["high"] = df["EURUSD_high"]
df["low"] = df["EURUSD_low"]
df["close"] = df["EURUSD_close"]
df["volume"] = df["EURUSD_volume"]
df["buy_volume"] = df["EURUSD_buy_volume"]
df["sell_volume"] = df["EURUSD_sell_volume"]
df["avg_spread"] = df["EURUSD_avg_spread"]

# 2. Apply Order Flow & Regime
print("[2/5] Applying Order Flow & Regime...")
ofa = OrderFlowAnalyzer()
df = ofa.analyze(df)

rd = RegimeDetector()
regime_df = rd.detect_history(df[["open", "high", "low", "close"]])
df["regime"] = regime_df["regime"]
df["regime_name"] = regime_df["regime_name"]
df["regime_confidence"] = regime_df["confidence"]

# 3. Add GMM Probabilities
print("[3/5] Merging GMM Regime Probabilities...")
df_gmm = pd.read_csv(os.path.join(DATA_DIR_L, "regime_gmm_probs.csv"), index_col=0, parse_dates=True)
df = df.join(df_gmm, how="left")

# 4. Add Microstructure Features
print("[4/5] Merging Microstructure Features...")
df_micro = pd.read_csv(os.path.join(DATA_DIR_L, "microstructure_features_20y.csv"), index_col=0, parse_dates=True)
df = df.join(df_micro, how="left")

# 5. Add Macro & COT (with ffill/bfill to preserve 20 years)
print("[5/5] Merging Macro & COT (with ffill to preserve 2003-2018)...")
df_macro = pd.read_csv(os.path.join(DATA_DIR_L, "macro_data.csv"), index_col=0, parse_dates=True)
df_cot = pd.read_csv(os.path.join(DATA_DIR_L, "cot_data.csv"), index_col=0, parse_dates=True)

# Left join to keep all 20-year rows, then fill NaNs
df = df.join([df_macro, df_cot], how="left")
df.ffill(inplace=True)
df.bfill(inplace=True) # Fill the very first rows if needed

# Final clean (only drop rows where core H1 data is missing, which shouldn't happen)
df.dropna(subset=["close", "vpin", "ofi_z", "regime_state_1_prob"], inplace=True)

print(f"\nFinal Shape: {df.shape}")
print(f"Date Range: {df.index[0]} to {df.index[-1]}")

# Save the full 20-year dataset
df.to_csv(os.path.join(DATA_DIR_L, "institutional_data_20y.csv"))

# Recreate Train and Holdout Splits
holdout_start = "2023-01-01"
train_df = df[df.index < holdout_start]
holdout_df = df[df.index >= holdout_start]

train_df.to_csv(os.path.join(DATA_DIR_L, "institutional_data_train.csv"))
holdout_df.to_csv(os.path.join(DATA_DIR_L, "holdout_data.csv"))

print(f"\nTrain Set Shape: {train_df.shape}")
print(f"Holdout Set Shape: {holdout_df.shape}")
print("\n✅ SUCCESS: 20-Year Data Restored and Split Correctly.")
