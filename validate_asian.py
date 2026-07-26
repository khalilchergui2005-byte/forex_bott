import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_hourly
from backtest_engine import dynamic_backtest
from monte_carlo_test import MonteCarloTest
from strategy_asian_breakout import generate_signals as gen_asian

print("=" * 60)
print("   VALIDATION SUITE - Asian Range Breakout (Session Edge)")
print("=" * 60)

print("\nLoading data and generating signals...")
data = load_hourly()
data = gen_asian(data)
print(f"Total Signals -> Buy: {(data['Asian_Breakout_Signal']==1).sum()}, Sell: {(data['Asian_Breakout_Signal']==-1).sum()}")

print("\n[1/2] Running Monte Carlo Simulation (1000 paths)...")
res = dynamic_backtest(data=data, strategy_cols=['Asian_Breakout_Signal'], log_trades=False)
if len(res) == 3:
    trades, equity_curve, _ = res
else:
    trades, equity_curve = res

print(f"Total trades: {len(trades)}")
if len(trades) > 30:
    mc = MonteCarloTest(n_sims=1000)
    report = mc.run(trades)
    print("\n=== Monte Carlo Report ===")
    print(f"Classification: {report.classification}")
    print(f"Real PF: {report.real_pf:.2f}")
    print(f"P-value: {report.p_value_pf:.3f}")
    print(f"Max DD: {report.real_max_dd_pct:.2f}%")
else:
    print("Not enough trades for Monte Carlo.")

print("\n[2/2] Running Walk-Forward Test (5 windows)...")
WINDOW_TRAIN = 2000
WINDOW_TEST = 1000
WINDOW_STEP = 500
NUM_WINDOWS = 5
wf_results = []
for w in range(NUM_WINDOWS):
    start_train = w * WINDOW_STEP
    end_train = start_train + WINDOW_TRAIN
    start_test = end_train
    end_test = start_test + WINDOW_TEST
    if end_test > len(data): break
    
    test_data = data.iloc[start_test:end_test].copy()
    res_w = dynamic_backtest(data=test_data, strategy_cols=['Asian_Breakout_Signal'], risk_percent=0.005, log_trades=False)
    t = res_w[0] if isinstance(res_w, tuple) else res_w
    
    if t and len(t) > 5:
        wins = [x for x in t if x > 0]
        losses = [x for x in t if x <= 0]
        pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
        wf_results.append(pf)
        print(f"   Window {w+1}: PF={pf:.2f}, Trades={len(t)}")
    else:
        wf_results.append(0.0)
        print(f"   Window {w+1}: No trades!")

if wf_results:
    avg_pf = np.mean(wf_results)
    std_pf = np.std(wf_results)
    print(f"\n   Walk-Forward Avg PF: {avg_pf:.2f} (±{std_pf:.2f})")
    print(f"   Stability: {'STABLE' if avg_pf > 1.0 and std_pf < 0.5 else 'UNSTABLE'}")
