"""
edge_diagnostics.py - Statistical Edge Test (Council Atom 7.1)
Tests if classical indicators have ANY predictive power.
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from data_loader import load_hourly
from parameter_controller import params

print("=" * 60)
print("   EDGE DIAGNOSTICS - Predictive Power Test")
print("=" * 60)

# 1. Load and prepare data
print("\n[1/3] Loading data...")
df = load_hourly()

# 2. Calculate indicators (exactly as in strategies)
print("[2/3] Calculating indicators...")
# ATR
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
# RSI
delta = df['close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
# Bollinger Bands position
df['BB_mid'] = df['close'].rolling(20).mean()
df['BB_std'] = df['close'].rolling(20).std()
df['BB_position'] = (df['close'] - (df['BB_mid'] - 2*df['BB_std'])) / (4*df['BB_std'])
# MACD
ema_fast = df['close'].ewm(span=12, adjust=False).mean()
ema_slow = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_fast - ema_slow
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']
# ADX
high_low = df['high'] - df['low']
high_close = np.abs(df['high'] - df['close'].shift())
low_close = np.abs(df['low'] - df['close'].shift())
true_range = np.max(pd.concat([high_low, high_close, low_close], axis=1), axis=1)
atr_adx = true_range.rolling(14).mean()
plus_dm = df['high'].diff().clip(lower=0)
minus_dm = (-df['low'].diff()).clip(lower=0)
plus_di = 100 * (plus_dm.rolling(14).mean() / atr_adx)
minus_di = 100 * (minus_dm.rolling(14).mean() / atr_adx)
dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
df['ADX'] = dx.rolling(14).mean()
# ATR Ratio
df['ATR_ratio'] = df['ATR'] / df['close']

# Forward returns (next candle)
df['forward_return'] = df['close'].pct_change().shift(-1)

# Clean
df.dropna(inplace=True)

# 3. Test predictive power
print("[3/3] Running statistical tests...")
indicators = ['RSI', 'BB_position', 'MACD_hist', 'ADX', 'ATR_ratio']
results = []
for ind in indicators:
    corr, pval = spearmanr(df[ind], df['forward_return'])
    results.append({'Indicator': ind, 'Spearman_Corr': corr, 'P_Value': pval})

# Bonferroni correction
n_tests = len(results)
for r in results:
    r['P_Value_Corrected'] = min(r['P_Value'] * n_tests, 1.0)
    r['Significant'] = 'YES' if r['P_Value_Corrected'] < 0.05 else 'NO'

# Print
print("\n" + "=" * 60)
print("   RESULTS - Predictive Power of Indicators")
print("=" * 60)
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
print(f"\nBonferroni correction applied (n_tests={n_tests}).")
print("If NO indicator is Significant, classical indicators have NO edge.")
print("=" * 60)
