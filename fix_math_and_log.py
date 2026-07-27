# --- 1. إصلاح الفساد الرياضي في spread_calculator.py ---
f = 'spread_calculator.py'
t = open(f, encoding='utf-8').read()

# إضافة حماية ضد القسمة على صفر
t = t.replace("df[\"hedge_ratio\"] = cov / var", "df[\"hedge_ratio\"] = cov / (var + 1e-10)")

# تحويل قيم Infinity إلى NaN لتُحذف بواسطة dropna
t = t.replace("df[\"spread\"] = s1 - (df[\"hedge_ratio\"] * s2)", "df[\"spread\"] = s1 - (df[\"hedge_ratio\"] * s2)\n    df.replace([np.inf, -np.inf], np.nan, inplace=True)")

open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: Fixed mathematical corruption (Division by zero & Infinity) in spread_calculator.py")

# --- 2. إصلاح سكريبت الاختبار لتفعيل سجل الصفقات ---
test_code = r'''import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from spread_calculator import calculate_spread
from strategy_pairs_trading import generate_signals as gen_pairs
from pairs_backtest_engine import dynamic_pairs_backtest
from monte_carlo_test import MonteCarloTest

print("=" * 60)
print("   FINAL VALIDATION - Pairs Trading (Math Fixed & Log Enabled)")
print("=" * 60)

print("\n[1/4] Calculating Spread & Cointegration Gate (5 Years Continuously)...")
df = calculate_spread()

print("\n[2/4] Generating Pairs Trading Signals (On Full Dataset)...")
df = gen_pairs(df)
print(f"Total Signals -> Long Spread: {(df['Pairs_Signal']==1).sum()}, Short Spread: {(df['Pairs_Signal']==-1).sum()}")

print("\n[3/4] Running Pairs Backtest for Monte Carlo...")
# تفعيل log_trades=True للحصول على سجل الصفقات
res = dynamic_pairs_backtest(data=df, log_trades=True)
if len(res) == 3:
    trades, equity_curve, trade_log = res
else:
    trades, equity_curve = res
    trade_log = []

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

print("\n[4/4] Running Continuous Walk-Forward Test (5 Annual Windows)...")
df_trades = pd.DataFrame(trade_log)

wf_results = []
if not df_trades.empty and 'exit_time' in df_trades.columns:
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    df_trades.set_index('exit_time', inplace=True)
    
    total_period = df.index[-1] - df.index[0]
    period_delta = total_period / 5
    
    for w in range(5):
        start_date = df.index[0] + (w * period_delta)
        end_date = start_date + period_delta
        
        period_trades = df_trades.loc[start_date:end_date, 'pnl'].values
        
        if len(period_trades) > 5:
            wins = [t for t in period_trades if t > 0]
            losses = [t for t in period_trades if t <= 0]
            pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else 0.0
            wf_results.append(pf)
            print(f"   Period {w+1} ({start_date.year}-{end_date.year}): PF={pf:.2f}, Trades={len(period_trades)}")
        else:
            wf_results.append(0.0)
            print(f"   Period {w+1} ({start_date.year}-{end_date.year}): No trades!")
else:
    print("No trade log available for Walk-Forward analysis.")

if wf_results:
    avg_pf = np.mean(wf_results)
    std_pf = np.std(wf_results)
    print(f"\n   Walk-Forward Avg PF: {avg_pf:.2f} (±{std_pf:.2f})")
    print(f"   Stability: {'STABLE' if avg_pf > 1.0 and std_pf < 0.5 else 'UNSTABLE'}")
'''
open("validate_pairs_cont.py", "w", encoding="utf-8").write(test_code)
print("SUCCESS: validate_pairs_cont.py updated to enable trade log")

import ast
ast.parse(open("spread_calculator.py", encoding="utf-8").read())
ast.parse(open("validate_pairs_cont.py", encoding="utf-8").read())
print("AST OK for both files")
