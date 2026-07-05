"""
backtest_engine.py - المحرك الديناميكي الموحد
يستخدم DataHub للحصول على البيانات.
"""
import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException
from data_loader import load_hourly

def dynamic_backtest(data=None, initial_balance=10000, risk_percent=0.005, max_daily_loss_pct=0.02, max_total_dd_pct=0.10, log_trades=False):
    if data is None:
        data = load_hourly()
    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_total_drawdown_pct': max_total_dd_pct,
    })
    rm.min_order_value = 50.0
    trades, equity_curve, trade_log = [], [initial_balance], []
    pos, entry_price, entry_time = 0, 0.0, None
    avg_atr = data['ATR'].mean()
    for i in range(len(data)):
        price, atr, signal = data['close'].iloc[i], data['ATR'].iloc[i], data['Final'].iloc[i]
        current_time = data.index[i] if hasattr(data.index[i], 'strftime') else i
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        stop_multiplier = max(1.5, min(2.5, 2.5 - (0.5 * vol_ratio)))
        if pos != 0:
            stop_loss = entry_price - (pos * stop_multiplier * atr)
            if signal == -pos or (pos == 1 and price <= stop_loss) or (pos == -1 and price >= stop_loss):
                pnl = (price - entry_price) * pos * rm.calculate_position_size(entry_price, stop_loss, 1.0)
                rm.update_pnl(pnl)
                trades.append(pnl)
                if log_trades:
                    trade_log.append({'entry_time': entry_time, 'exit_time': current_time, 'direction': 'LONG' if pos == 1 else 'SHORT', 'entry_price': entry_price, 'exit_price': price, 'net_pnl': pnl, 'balance_after': float(rm.current_balance)})
                pos = 0
                continue
        if pos == 0 and signal != 0:
            try:
                rm.check_circuit_breaker()
                stop_loss = price - (signal * stop_multiplier * atr)
                size = rm.calculate_position_size(price, stop_loss, 1.0)
                if size > 0:
                    pos, entry_price, entry_time = signal, price, current_time
            except CircuitBreakerException:
                break
        equity_curve.append(float(rm.current_balance))
    return (trades, equity_curve, trade_log) if log_trades else (trades, equity_curve)
