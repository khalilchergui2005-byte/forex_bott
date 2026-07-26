"""
robustness_shadow_thresholds.py - Robustness test for Long Lower Shadow (Atom 8.4)
Tests the edge across three threshold values (0.8, 1.0, 1.2) using CPCV.
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
print("   ROBUSTNESS TEST - Long Lower Shadow (3 Thresholds)")
print("=" * 60)

# 1. Load data
print("\n[1/3] Loading H1 data...")
full_data = load_hourly()
full_data['ATR'] = (full_data['high'] - full_data['low']).rolling(14).mean()
full_data['lower_shadow'] = (full_data[['open', 'close']].min(axis=1) - full_data['low']) / full_data['ATR']
full_data.dropna(inplace=True)
print(f"   Clean candles: {len(full_data)}")

# 2. CPCV Setup
n_groups = 5
group_size = len(full_data) // n_groups
groups = [full_data.iloc[i*group_size:(i+1)*group_size].copy() for i in range(n_groups)]
test_combinations = list(combinations(range(n_groups), 2))
thresholds = [0.8, 1.0, 1.2]

# 3. Run CPCV for each threshold
print(f"\n[2/3] Testing {len(thresholds)} thresholds across {len(test_combinations)} CPCV paths...")
all_results = {}
for threshold in thresholds:
    print(f"\n--- Testing threshold >= {threshold} ---")
    pf_results = []
    for idx, test_idx in enumerate(test_combinations):
        test_data = pd.concat([groups[i] for i in test_idx]).copy()
        test_data['Signal'] = 0
        test_data.loc[test_data['lower_shadow'] >= threshold, 'Signal'] = 1
        
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
    n_trials = len(test_combinations)
    expected_max = np.sqrt(2 * np.log(n_trials))
    dsr = (avg_pf - 1.0) / max(std_pf, 0.001) * (1 - expected_max / np.sqrt(n_trials))
    all_results[threshold] = {'avg_pf': avg_pf, 'std_pf': std_pf, 'dsr': dsr, 'pct_gt_1': sum(1 for p in pf_results if p>1.0)/len(pf_results)*100}

# 4. Display comparison
print(f"\n{'='*60}")
print("   ROBUSTNESS COMPARISON")
print(f"{'='*60}")
print(f"   {'Threshold':<12} {'Avg PF':<10} {'DSR':<10} {'% PF>1':<10}")
print(f"   {'-'*50}")
for t in thresholds:
    r = all_results[t]
    print(f"   {t:<12} {r['avg_pf']:<10.3f} {r['dsr']:<10.3f} {r['pct_gt_1']:<10.1f}%")
print(f"{'='*60}")
print("If edge is REAL, PF and DSR should remain positive and similar across thresholds.")
print("If they collapse at lower/higher thresholds, the edge is likely statistical noise.")
