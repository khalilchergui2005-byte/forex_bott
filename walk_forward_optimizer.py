"""
walk_forward_optimizer.py - Walk-Forward Test for Long Lower Shadow Strategy
Tests ONLY the Signal column across 5 time windows.
"""
import optuna
import pandas as pd
import numpy as np
from data_loader import load_hourly
from strategy_long_lower_shadow import generate_long_lower_shadow_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   WALK-FORWARD TEST - Long Lower Shadow Strategy")
print("=" * 60)

# 1. Load and prepare data
print("\n[1/3] Loading data...")
full_data = load_hourly()
full_data = generate_long_lower_shadow_signals(full_data)
print(f"   Total candles: {len(full_data)}, Signals: {full_data['Signal'].sum()}")

# 2. Walk-Forward Setup (5 windows)
WINDOW_TRAIN = 2000
WINDOW_TEST = 1000
WINDOW_STEP = 500
NUM_WINDOWS = 5

results = []
for w in range(NUM_WINDOWS):
    start_train = w * WINDOW_STEP
    end_train = start_train + WINDOW_TRAIN
    start_test = end_train
    end_test = start_test + WINDOW_TEST
    
    if end_test > len(full_data):
        break
    
    test_data = full_data.iloc[start_test:end_test].copy()
    
    trades, equity = dynamic_backtest(
        data=test_data,
        strategy_cols=['Signal'],
        risk_percent=0.005
    )
    
    if trades and len(trades) > 5:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
        ret = (equity[-1]/params.initial_capital - 1)*100
        dd = (pd.Series(equity)/pd.Series(equity).cummax() - 1).min()*100
        results.append({'window': w+1, 'pf': pf, 'return': ret, 'dd': dd, 'trades': len(trades)})
        print(f"   Window {w+1}: PF={pf:.2f}, Return={ret:.2f}%, MaxDD={dd:.2f}%, Trades={len(trades)}")
    else:
        results.append({'window': w+1, 'pf': 0.0, 'return': 0.0, 'dd': 0.0, 'trades': 0})
        print(f"   Window {w+1}: No trades!")

# 3. Final Report
if results:
    df_results = pd.DataFrame(results)
    avg_pf = df_results['pf'].mean()
    std_pf = df_results['pf'].std()
    print(f"\n{'='*60}")
    print(f"   WALK-FORWARD FINAL REPORT")
    print(f"{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\n   Average PF: {avg_pf:.2f} (±{std_pf:.2f})")
    print(f"   Stability: {'STABLE' if avg_pf > 1.0 and std_pf < 0.5 else 'UNSTABLE'}")
    print(f"{'='*60}")
