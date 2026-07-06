"""
prop_firm_sim.py - Simulates a Prop Firm Challenge environment.
Capital: 100,000$ | Min Order: 500$ | Max Daily Loss: 5% | Max Total DD: 10%
"""
import pandas as pd
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

# --- Override settings for Prop Firm Simulation ---
print("Configuring Prop Firm Environment (100K Challenge)...")
params.initial_capital = 100000.0
params.min_order_value = 500.0  # Realistic minimum for 100k accounts
params.max_daily_loss_pct = 0.05  # 5% daily loss limit (Typical FTMO/TFT)
params.max_total_dd_pct = 0.10   # 10% max total drawdown

# --- Load and prepare data ---
print("Loading H1 data...")
df = load_hourly()
df = generate_signals(df)
df = generate_momentum_signals(df)
df['Final_Mom'] = df['MO_Signal']

# --- Run simulation with both strategies ---
print("Running Prop Firm Simulation...")
trades, equity, trade_log = dynamic_backtest(
    data=df,
    strategy_cols=['Final', 'Final_Mom'],
    risk_percent=0.005,  # 0.5% per trade
    max_daily_loss_pct=0.05,
    max_total_dd_pct=0.10,
    log_trades=True
)

# --- Results ---
print("\n" + "=" * 60)
print("   PROP FIRM CHALLENGE RESULT (100,000$ Account)")
print("=" * 60)
if trades and len(trades) > 0:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    total_ret = (equity[-1]/params.initial_capital - 1)*100
    max_dd = (pd.Series(equity)/pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    print(f"   💰 Final Balance: {equity[-1]:,.2f} $")
    print(f"   📈 Total Return: {total_ret:.2f}%")
    print(f"   📉 Max Drawdown: {max_dd:.2f}%")
    print(f"   🔢 Total Trades: {n}")
    print(f"   ✅ Win Rate: {wr:.1f}%")
    print(f"   ⚖️ Profit Factor: {pf:.2f}")
    print(f"   🎯 Challenge Status: {'PASSED' if total_ret >= 10.0 and max_dd > -10.0 else 'FAILED'}")
else:
    print("   No trades generated. Check signals and data.")

print("=" * 60)
