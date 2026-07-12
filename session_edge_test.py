"""
session_edge_test.py - Trading Session Edge Test (Council Atom 7.4)
Tests buy signals at the start of major trading sessions (UTC).
"""
import pandas as pd
import numpy as np
from itertools import combinations
from data_loader import load_hourly
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print("=" * 60)
    print("   TRADING SESSION EDGE TEST (CPCV-Based)")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/3] Loading and preparing data...")
    df = load_hourly()
    # Pre-calculate ATR (required by backtest engine)
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    df.dropna(inplace=True)
    
    # Define session start hours (UTC)
    sessions = {
        'Tokyo_Open': 0,      # 00:00 UTC (9:00 AM Tokyo)
        'London_Open': 8,     # 08:00 UTC (9:00 AM London)
        'NY_Open': 13,        # 13:00 UTC (9:00 AM New York)
        'Day_Start': 7,       # 07:00 UTC (Start of European business day)
    }
    
    # 2. CPCV Setup
    n_groups = 5
    group_size = len(df) // n_groups
    groups = [df.iloc[i*group_size:(i+1)*group_size].copy() for i in range(n_groups)]
    test_combinations = list(combinations(range(n_groups), 2))
    
    print(f"[2/3] Testing {len(sessions)} session opens across {len(test_combinations)} CPCV paths...")
    
    # 3. Run CPCV for each session
    results = {}
    for session_name, start_hour in sessions.items():
        pf_list = []
        for test_idx in test_combinations:
            test_data = pd.concat([groups[i] for i in test_idx]).copy()
            test_data['Signal'] = 0
            test_data.loc[test_data.index.hour == start_hour, 'Signal'] = 1
            
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
        pct_gt_1 = sum(1 for p in pf_list if p>1.0)/len(pf_list)*100
        results[session_name] = {'Avg_PF': avg_pf, '% PF>1': pct_gt_1}
    
    # 4. Display results
    print(f"\n{'='*60}")
    print("   SESSION EDGE RESULTS (Sorted by Avg PF)")
    print(f"{'='*60}")
    print(f"   {'Session':<15} {'Avg PF':<10} {'% PF>1':<10}")
    print(f"   {'-'*35}")
    for session_name in sorted(results, key=lambda x: results[x]['Avg_PF'], reverse=True):
        r = results[session_name]
        print(f"   {session_name:<15} {r['Avg_PF']:<10.3f} {r['% PF>1']:<10.1f}%")
    print(f"{'='*60}")
    print("If any session shows Avg_PF > 1.1 and % PF>1 > 70%, it is a candidate for strategy.")
