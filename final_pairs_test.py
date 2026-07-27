# --- 1. تحديث spread_calculator.py بـ lookback=170 ---
f = 'spread_calculator.py'
t = open(f, encoding='utf-8').read()
t = t.replace("def calculate_spread(pair1=\"EURUSD\", pair2=\"GBPUSD\", lookback=100)", "def calculate_spread(pair1=\"EURUSD\", pair2=\"GBPUSD\", lookback=170)")
open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: Updated spread_calculator.py with lookback=170")

# --- 2. تحديث strategy_pairs_trading.py بالإعدادات المثلى ---
f = 'strategy_pairs_trading.py'
t = open(f, encoding='utf-8').read()
t = t.replace("def generate_signals(df, entry_z=2.0, exit_z=0.0, stop_z=4.0)", "def generate_signals(df, entry_z=2.5, exit_z=0.2, stop_z=5.0)")
open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: Updated strategy_pairs_trading.py with optimal parameters")

# --- 3. تشغيل الاختبار النهائي ---
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from spread_calculator import calculate_spread
from strategy_pairs_trading import generate_signals as gen_pairs
from pairs_backtest_engine import dynamic_pairs_backtest
from monte_carlo_test import MonteCarloTest

print("\n" + "=" * 60)
print("   FINAL VALIDATION - Pairs Trading (Optimized Institutional)")
print("=" * 60)

print("\n[1/4] Calculating Spread & Z-Score (5 Years Real Data, Lookback=170)...")
df = calculate_spread()

print("\n[2/4] Generating Pairs Trading Signals (Entry=2.5, Exit=0.2, Stop=5.0)...")
df = gen_pairs(df)
print(f"Total Signals -> Long Spread: {(df['Pairs_Signal']==1).sum()}, Short Spread: {(df['Pairs_Signal']==-1).sum()}")

print("\n[3/4] Running Pairs Backtest for Monte Carlo...")
res = dynamic_pairs_backtest(data=df, log_trades=False)
if len(res) == 3:
    trades, equity_curve, _ = res
else:
    trades, equity_curve = res

print(f"Total trades executed: {len(trades)}")
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

print("\n[4/4] Running Walk-Forward Test (5 windows)...")
WINDOW_STEP = 500
WINDOW_TEST = 1000
NUM_WINDOWS = 5
wf_results = []

total_len = len(df)
start_idx = total_len - (WINDOW_TEST * NUM_WINDOWS)

for w in range(NUM_WINDOWS):
    start_test = start_idx + (w * WINDOW_STEP)
    end_test = start_test + WINDOW_TEST
    if end_test > total_len: break
    
    # Slice the pre-calculated dataframe
    test_data = df.iloc[start_test:end_test].copy()
    # Re-generate signals for the sliced window to reset state
    test_data = gen_pairs(test_data)
    
    res_w = dynamic_pairs_backtest(data=test_data, log_trades=False)
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
