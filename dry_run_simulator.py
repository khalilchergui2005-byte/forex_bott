"""
dry_run_simulator.py - Dynamic Portfolio (3 strategies + adaptive weighting)
Project: Viral Entity - Strongest Trading Firm
"""
import pandas as pd
from pathlib import Path
from data_loader import load_hourly
from strategy_matrix import generate_signals
from strategy_momentum import generate_momentum_signals
from strategy_range_hunter import generate_range_signals
from backtest_engine import dynamic_backtest
from unified_filters import apply_time_filter, apply_weekly_bias_filter
from parameter_controller import params
from weekly_analyzer import analyze_week, apply_weekly_rules, get_effective_risk, recalculate_filters

print("=" * 60)
print("   🦅 DYNAMIC PORTFOLIO (Matrix + Momentum + Range Hunter)")
print("=" * 60)

# 1. Load data
print("\n📥 Loading H1 data...")
df = load_hourly()
print(f"   ✅ Loaded {len(df)} candles")

# 2. Generate signals from ALL THREE strategies
print("\n🧠 Generating signals...")
df = generate_signals(df)          # Matrix
df = generate_momentum_signals(df)  # Momentum
df = generate_range_signals(df)     # Range Hunter
df['Final_Mom'] = df['MO_Signal']
df['Final_Range'] = df['Range_Signal']
print(f"   Matrix: {(df['Final']!=0).sum()}, Momentum: {(df['Final_Mom']!=0).sum()}, Range: {(df['Final_Range']!=0).sum()}")

# 3. Confirmation layer
print("\n🛡️ Calculating Confirmation...")
time_ok = apply_time_filter(df)
day_ok = apply_weekly_bias_filter(df)
vol_ok = (df['VolClass'] == 2)
df['Confirmation'] = 1.0
df.loc[time_ok, 'Confirmation'] += 0.5
df.loc[day_ok, 'Confirmation'] += 0.5
df.loc[vol_ok, 'Confirmation'] += 0.5
df['Confirmation'] = df['Confirmation'].clip(upper=2.0)

# 4. Dynamic weighting based on recent performance
print("\n⚖️ Calculating dynamic strategy weights...")
try:
    past_log = pd.read_csv('logs/dry_run_log.csv')
    # Simple dynamic weight: recent win rate per strategy
    if 'strategy' in past_log.columns and len(past_log) > 50:
        recent = past_log.tail(100)
        perf = recent.groupby('strategy')['net_pnl'].apply(lambda x: (x > 0).mean())
        # Default weights
        weights = {'Final': 0.5, 'Final_Mom': 0.3, 'Final_Range': 0.2}
        for strat in perf.index:
            if strat in weights:
                weights[strat] = max(0.1, min(0.6, perf[strat]))  # Cap between 10% and 60%
        # Normalize
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
        print(f"   Weights: {weights}")
    else:
        weights = {'Final': 0.5, 'Final_Mom': 0.3, 'Final_Range': 0.2}
except FileNotFoundError:
    weights = {'Final': 0.5, 'Final_Mom': 0.3, 'Final_Range': 0.2}
    print("   No log file found. Using default weights.")

# 5. Weekly analyzer
print("\n🔄 Analyzing past performance...")
try:
    dynamic_filters = recalculate_filters(past_log)
    good_days = dynamic_filters['good_days']
    good_hours = dynamic_filters['good_hours']
    analysis = analyze_week(past_log)
    print(f"   {analysis['message']}")
    weekly_rules = analysis['rules']
except (FileNotFoundError, NameError):
    good_days = [1, 2, 4]
    good_hours = [7, 8, 12, 13]
    weekly_rules = {}
    print("   ⚠️ No past log. Fresh start.")

df['time_ok'] = df['hour'].isin(good_hours)
df['day_ok'] = df['weekday'].isin(good_days)
if weekly_rules:
    df = apply_weekly_rules(df, weekly_rules)

base_risk = 0.005
effective_risk = get_effective_risk(base_risk, weekly_rules)

# 6. Run with three strategies
print(f"\n⚙️ Running simulation (risk: {effective_risk*100:.2f}%)...")
trades, equity, trade_log = dynamic_backtest(
    df, params.initial_capital, risk_percent=effective_risk, log_trades=True,
    strategy_cols=['Final', 'Final_Mom', 'Final_Range']
)

# 7. Results
print("\n📊 Simulation Results:")
if trades:
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    n = len(trades)
    wr = len(wins)/n*100
    ret = (equity[-1]/params.initial_capital - 1)*100
    dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min() * 100
    pf = abs(sum(wins)/sum(losses)) if losses and sum(losses)!=0 else (10 if wins else 0)
    print(f"   💰 Final Balance: {equity[-1]:,.2f} $")
    print(f"   📈 Total Return: {ret:.2f}%")
    print(f"   📉 Max Drawdown: {dd:.2f}%")
    print(f"   🔢 Total Trades: {n}")
    print(f"   ✅ Win Rate: {wr:.1f}%")
    print(f"   ⚖️  Profit Factor: {pf:.2f}")
else:
    print("   ⚠️ No trades.")

if trade_log:
    log_df = pd.DataFrame(trade_log)
    Path("logs").mkdir(exist_ok=True)
    log_df.to_csv("logs/dry_run_log.csv", index=False)
    print(f"\n📋 {len(log_df)} trades logged.")

print("\n🎉 Dynamic Portfolio simulation complete.")
print("=" * 60)
