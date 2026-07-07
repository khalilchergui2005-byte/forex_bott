"""
edge_scanner.py - Systematic Edge Discovery (Council Atom 8.9)
Scans multiple feature classes for predictive power using CPCV.
"""
import pandas as pd
import numpy as np
from data_loader import load_hourly
from backtest_engine import dynamic_backtest
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   SYSTEMATIC EDGE SCANNER (CPCV-based)")
print("=" * 60)

# 1. Load and prepare data
print("\n[1/3] Loading and preparing data...")
df = load_hourly()
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 10000
df['body'] = abs(df['close'] - df['open']) / df['ATR']
df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['ATR']
df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['ATR']
df.dropna(inplace=True)

# Define feature sets to test (all conditions now use safe thresholds)
feature_sets = {
    'Low_ATR': ('ATR_ratio', 'percentile_below', 0.33),
    'High_ATR': ('ATR_ratio', 'percentile_above', 0.66),
    'Positive_Gap': ('gap', 'above_value', 5.0),
    'Negative_Gap': ('gap', 'below_value', -5.0),
    'Small_Body': ('body', 'percentile_below', 0.5),
    'Large_Body': ('body', 'percentile_above', 0.5),
    'Long_Upper_Shadow': ('upper_shadow', 'above_value', 1.0),
    'Long_Lower_Shadow': ('lower_shadow', 'above_value', 1.0),
    'Hour_00': ('hour', 'eq', 0),
    'Hour_13': ('hour', 'eq', 13),
}

# CPCV Setup
n_groups = 5
group_size = len(df) // n_groups
groups = [df.iloc[i*group_size:(i+1)*group_size].copy() for i in range(n_groups)]
test_combinations = list(combinations(range(n_groups), 2))
print(f"   Testing {len(feature_sets)} features across {len(test_combinations)} CPCV paths.")

# 3. Scan features
print("\n[3/3] Scanning features...")
results = []
for name, (col, condition, threshold) in feature_sets.items():
    pf_list = []
    for test_idx in test_combinations:
        test_data = pd.concat([groups[i] for i in test_idx]).copy()
        test_data['Signal'] = 0
        
        # Generate signal safely
        if condition == 'percentile_below':
            limit = test_data[col].quantile(threshold)
            test_data.loc[test_data[col] <= limit, 'Signal'] = 1
        elif condition == 'percentile_above':
            limit = test_data[col].quantile(threshold)
            test_data.loc[test_data[col] >= limit, 'Signal'] = 1
        elif condition == 'above_value':
            test_data.loc[test_data[col] >= threshold, 'Signal'] = 1
        elif condition == 'below_value':
            test_data.loc[test_data[col] <= threshold, 'Signal'] = 1
        elif condition == 'eq':
            test_data.loc[test_data[col] == threshold, 'Signal'] = 1
        
        if test_data['Signal'].sum() < 5:
            pf_list.append(0.0)
            continue
        
        trades, equity = dynamic_backtest(
            data=test_data,
            strategy_cols=['Signal'],
            risk_percent=0.005
        )
        if trades and len(trades) > 5:
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t <= 0]
            pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
            pf_list.append(pf)
        else:
            pf_list.append(0.0)
    
    avg_pf = np.mean(pf_list)
    pct_gt_1 = sum(1 for p in pf_list if p>1.0)/len(pf_list)*100 if pf_list else 0
    results.append({'Feature': name, 'Avg_PF': avg_pf, 'Pct_PF_gt_1': pct_gt_1})

# 4. Display results
print("\n" + "=" * 60)
print("   SCAN RESULTS (Sorted by Avg PF)")
print("=" * 60)
df_res = pd.DataFrame(results).sort_values('Avg_PF', ascending=False)
print(df_res.to_string(index=False))
print(f"\n{'='*60}")
print("Features with Avg_PF > 1.1 and Pct > 70% are candidates for further investigation.")
