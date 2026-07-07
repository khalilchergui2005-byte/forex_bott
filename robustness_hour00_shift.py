"""
robustness_hour00_shift.py - Robustness test for Hour_00 (Candidate 3)
Tests the edge by shifting the entry time ±1 and ±2 hours.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from data_loader import load_hourly
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   ROBUSTNESS TEST - Hour_00 (Time Shift)")
print("=" * 60)

# 1. Load data and pre-calculate ATR
print("\n[1/3] Loading and preparing data...")
full_data = load_hourly()
full_data['ATR'] = (full_data['high'] - full_data['low']).rolling(14).mean()
full_data.dropna(inplace=True)
print(f"   Clean candles: {len(full_data)}")

# 2. CPCV Setup
n_groups = 5
group_size = len(full_data) // n_groups
groups = [full_data.iloc[i*group_size:(i+1)*group_size].copy() for i in range(n_groups)]
test_combinations = list(combinations(range(n_groups), 2))

shifts = {'Hour_23': 23, 'Hour_00': 0, 'Hour_01': 1, 'Hour_02': 2}

# 3. Run CPCV for each shift
print(f"\n[2/3] Testing {len(shifts)} time shifts across {len(test_combinations)} CPCV paths...")
all_results = {}
for label, hour in shifts.items():
    pf_results = []
    for test_idx in test_combinations:
        test_data = pd.concat([groups[i] for i in test_idx]).copy()
        test_data['Signal'] = 0
        test_data.loc[test_data.index.hour == hour, 'Signal'] = 1
        
        if test_data['Signal'].sum() < 5:
            pf_results.append(0.0)
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
            pf_results.append(pf)
        else:
            pf_results.append(0.0)
    
    avg_pf = np.mean(pf_results)
    std_pf = np.std(pf_results)
    all_results[label] = {'avg_pf': avg_pf, 'pct_gt_1': sum(1 for p in pf_results if p>1.0)/len(pf_results)*100}

# 4. Display comparison
print(f"\n{'='*60}")
print("   ROBUSTNESS COMPARISON (Hour_00)")
print(f"{'='*60}")
print(f"   {'Hour':<12} {'Avg PF':<10} {'% PF>1':<10}")
print(f"   {'-'*40}")
for label in ['Hour_23', 'Hour_00', 'Hour_01', 'Hour_02']:
    r = all_results[label]
    print(f"   {label:<12} {r['avg_pf']:<10.3f} {r['pct_gt_1']:<10.1f}%")
print(f"{'='*60}")
print("If edge is REAL, Hour_00 should have the best PF.")
