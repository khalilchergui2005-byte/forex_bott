code = r'''import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from institutional_data_handler import prepare_institutional_data
from volume_shock_detector import VolumeShockDetector
from strategy_news_deviation import generate_signals as gen_news
from backtest_engine import dynamic_backtest
from monte_carlo_test import MonteCarloTest

print("=" * 60)
print("   VALIDATION SUITE - News Deviation (Liquidity Shock)")
print("=" * 60)

print("\n[1/4] Preparing Institutional Data (5 Years Real Data)...")
inst_data = prepare_institutional_data()

print("\n[2/4] Detecting Volume Shocks (News Events)...")
detector = VolumeShockDetector()
inst_data = detector.detect(inst_data)

print("\n[3/4] Generating News Deviation Signals (Hold 4h)...")
inst_data = gen_news(inst_data)
print(f"Total Signals -> Buy: {(inst_data['News_Signal']==1).sum()}, Sell: {(inst_data['News_Signal']==-1).sum()}")

print("\nRunning Baseline Backtest for Monte Carlo...")
res = dynamic_backtest(data=inst_data, strategy_cols=['News_Signal'], log_trades=False)
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

print("\n[4/4] Running Walk-Forward Test (5 Continuous Windows)...")
# Using continuous slicing to preserve rolling history for Smart Gate
wf_results = []
total_len = len(inst_data)
WINDOW_TEST = total_len // 5
WINDOW_STEP = WINDOW_TEST

for w in range(5):
    start_test = w * WINDOW_STEP
    end_test = start_test + WINDOW_TEST
    if end_test > total_len: break
    
    test_data = inst_data.iloc[start_test:end_test].copy()
    # Re-generate signals for the sliced window to reset trade state
    test_data = gen_news(test_data)
    
    res_w = dynamic_backtest(data=test_data, strategy_cols=['News_Signal'], log_trades=False)
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
'''
open("validate_news.py", "w", encoding="utf-8").write(code)
print("SUCCESS: validate_news.py created")

import ast
ast.parse(code)
print("AST OK")
