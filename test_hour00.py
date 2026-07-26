import pandas as pd
from data_loader import load_hourly
from backtest_engine import dynamic_backtest

try:
    from strategy_hour00 import generate_hour00_signals as gen_hour00
except ImportError:
    from strategy_hour00 import generate_signals as gen_hour00

data = load_hourly()
data = gen_hour00(data)
print("Signals generated. Number of 1s:", (data['Signal']==1).sum(), "Number of -1s:", (data['Signal']==-1).sum())

trades, equity, log = dynamic_backtest(data=data, strategy_cols=['Signal'], log_trades=True)
print(f"Total trades executed: {len(trades)}")
if len(trades) > 0:
    print(f"First trade PnL: {trades[0]}")
else:
    print("No trades were executed by the engine.")
