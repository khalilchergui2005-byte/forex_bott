"""
prop_firm_challenge.py - Full Prop Firm Challenge Simulator
Capital: $100,000 | Min Order: $500 | Rules: FTMO
"""
import pandas as pd
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from backtest_engine import dynamic_backtest
from parameter_controller import params

print("=" * 60)
print("   🚀 PROP FIRM CHALLENGE SIMULATOR ($100,000)")
print("=" * 60)

# 1. Override settings for the challenge
print("\n⚙️ Configuring FTMO Challenge Environment...")
params.initial_capital = 100000.0
params.min_order_value = 500.0
params.max_daily_loss_pct = 0.05
params.max_total_dd_pct = 0.10
print("   Capital: $100,000 | Min Order: $500 | Daily Loss Limit: 5% | Total DD Limit: 10%")

# 2. Load and prepare data
print("\n📥 Loading H1 data...")
df = load_hourly()
df = generate_signals(df)
df = generate_momentum_signals(df)
df['Final_Mom'] = df['MO_Signal']
print(f"   Data loaded: {len(df)} candles. Matrix signals: {(df['Final']!=0).sum()}, Momentum signals: {(df['Final_Mom']!=0).sum()}")

# 3. Apply Confirmation (simplified, same logic as dry_run_simulator)
print("\n🛡️ Applying Confirmation...")
df['Confirmation'] = 1.0
print("   Confirmation applied.")

# 4. Run the challenge
print("\n⚡ Running the Challenge...")
trades, equity, trade_log = dynamic_backtest(
    data=df,
    strategy_cols=['Final', 'Final_Mom'],
    risk_percent=0.005,
    max_daily_loss_pct=0.05,
    max_total_dd_pct=0.10,
    log_trades=True
)

# 5. Results
print("\n" + "=" * 60)
print("   📊 CHALLENGE RESULTS")
print("=" * 60)
if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100 if n > 0 else 0
    total_ret = (equity[-1]/params.initial_capital - 1)*100
    max_dd = (pd.Series(equity)/pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10.0 if wins else 0.0)
    print(f"   💰 Final Balance: ${equity[-1]:,.2f}")
    print(f"   📈 Total Return: {total_ret:.2f}%")
    print(f"   📉 Max Drawdown: {max_dd:.2f}%")
    print(f"   🔢 Total Trades: {n}")
    print(f"   ✅ Win Rate: {wr:.1f}%")
    print(f"   ⚖️  Profit Factor: {pf:.2f}")
    print(f"   🎯 Challenge Status: {'✅ PASSED' if total_ret >= 10.0 and max_dd > -10.0 else '❌ FAILED'}")
else:
    print("   No trades generated.")
print("=" * 60)
