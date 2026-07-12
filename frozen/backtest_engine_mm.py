import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException

def dynamic_backtest_mm(data, initial_balance, risk_percent=0.005, max_daily_loss_pct=0.02, max_total_dd_pct=0.10, cost_per_100k=7):
    """
    محرك اختبار خلفي ديناميكي لصانع السوق.
    """
    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_total_drawdown_pct': max_total_dd_pct,
    })
    rm.min_order_value = 10.0  # وضع الاختبار
    
    trades = []
    equity_curve = [initial_balance]
    pos = 0
    entry_price = 0.0
    direction = 0
    avg_atr = data['ATR'].mean()
    
    for i in range(1, len(data)):
        price = data['close'].iloc[i]
        atr = data['ATR'].iloc[i]
        active = data['MM_Active'].iloc[i]
        
        # حساب مضاعفات ديناميكية
        if avg_atr > 0:
            vol_ratio = atr / avg_atr
        else:
            vol_ratio = 1.0
        # الهدف والوقف ديناميكيان
        spread_target = 0.5 * atr
        stop_multiplier = 2.0 - (0.5 * vol_ratio)
        stop_multiplier = max(1.2, min(2.0, stop_multiplier))
        
        if pos != 0:
            target = entry_price + direction * spread_target
            stop = entry_price - direction * stop_multiplier * atr
            close_trade = False
            if direction == 1:
                if price >= target or price <= stop: close_trade = True
            else:
                if price <= target or price >= stop: close_trade = True
            if not active:
                close_trade = True
                
            if close_trade:
                pnl = (price - entry_price) * direction * rm.calculate_position_size(entry_price, stop, 1.0)
                cost = (rm.calculate_position_size(entry_price, stop, 1.0) / 100_000) * cost_per_100k
                net_pnl = pnl - cost
                rm.update_pnl(net_pnl)
                trades.append(net_pnl)
                pos = 0
                
        if pos == 0 and active:
            try:
                rm.check_circuit_breaker()
                stop_loss = price - (1 * stop_multiplier * atr)
                size = rm.calculate_position_size(price, stop_loss, 1.0)
                if size > 0:
                    direction = 1 if i % 2 == 0 else -1
                    entry_price = price
                    pos = direction
            except CircuitBreakerException:
                break
                
        equity_curve.append(float(rm.current_balance))
        
    return trades, equity_curve
