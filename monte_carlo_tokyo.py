"""
monte_carlo_tokyo.py - FTMO Challenge Monte Carlo on Tokyo Open Edge (Atom 7.6)
Estimates success probability using bootstrapped trade sequences.
"""
import pandas as pd
import numpy as np
from data_loader import load_hourly
from backtest_engine import dynamic_backtest
from parameter_controller import params
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   FTMO MONTE CARLO - Tokyo Open Edge")
print("=" * 60)

# 1. Generate real trade sequence
print("\n[1/4] Generating real trade history...")
df = load_hourly()
df['Signal'] = 0
df.loc[df.index.hour == 0, 'Signal'] = 1  # Tokyo Open
trades, equity, _ = dynamic_backtest(
    data=df,
    strategy_cols=['Signal'],
    risk_percent=0.005,
    log_trades=True
)

if not trades or len(trades) < 20:
    print("   Not enough trades for simulation. Exiting.")
    exit()

print(f"   Total real trades: {len(trades)}")
print(f"   Win rate: {sum(1 for t in trades if t>0)/len(trades)*100:.1f}%")
print(f"   Avg win: ${np.mean([t for t in trades if t>0]):.2f}")
print(f"   Avg loss: ${np.mean([abs(t) for t in trades if t<=0]):.2f}")

# 2. Monte Carlo Setup
N_SIMULATIONS = 2000
TARGET_PROFIT = 0.10  # 10% profit target (FTMO)
MAX_DRAWDOWN = 0.10   # 10% max drawdown (FTMO)
INITIAL_CAPITAL = 100000.0  # FTMO Challenge capital

print(f"\n[2/4] Running {N_SIMULATIONS} Monte Carlo simulations...")
print(f"   Target: +{TARGET_PROFIT*100:.0f}% | Max DD: -{MAX_DRAWDOWN*100:.0f}%")
print(f"   Initial Capital: ${INITIAL_CAPITAL:,.0f}")

# 3. Run simulations
success_count = 0
failure_count = 0

for sim in range(N_SIMULATIONS):
    # Resample trades with replacement (bootstrap)
    sim_trades = np.random.choice(trades, size=len(trades), replace=True)
    
    balance = INITIAL_CAPITAL
    peak = balance
    failed = False
    
    for trade in sim_trades:
        balance += trade
        peak = max(peak, balance)
        dd = (balance / peak) - 1
        
        if dd <= -MAX_DRAWDOWN:
            failure_count += 1
            failed = True
            break
        
        if (balance / INITIAL_CAPITAL - 1) >= TARGET_PROFIT:
            success_count += 1
            failed = True
            break
    
    if not failed:
        # Ended without hitting either limit
        if (balance / INITIAL_CAPITAL - 1) >= TARGET_PROFIT:
            success_count += 1

# 4. Results
success_rate = success_count / N_SIMULATIONS * 100
failure_rate = failure_count / N_SIMULATIONS * 100

print(f"\n{'='*60}")
print("   FTMO MONTE CARLO RESULTS - Tokyo Open Edge")
print(f"{'='*60}")
print(f"   Simulations: {N_SIMULATIONS}")
print(f"   ✅ Success (hit +10% before -10%): {success_count} ({success_rate:.1f}%)")
print(f"   ❌ Failed (hit -10%): {failure_count} ({failure_rate:.1f}%)")
print(f"   📊 Expected return (avg): ${np.mean([sum(np.random.choice(trades, size=len(trades), replace=True)) for _ in range(500)]):.2f}")
print(f"{'='*60}")
if success_rate >= 70:
    print("   VERDICT: READY for FTMO Challenge!")
elif success_rate >= 40:
    print("   VERDICT: PROMISING - Needs minor improvements.")
else:
    print("   VERDICT: NOT READY - Requires significant edge improvement.")
