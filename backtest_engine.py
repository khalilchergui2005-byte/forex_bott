import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException

def dynamic_backtest(data, initial_balance, risk_percent=0.005, max_daily_loss_pct=0.02, max_total_dd_pct=0.10):
    """
    محرك اختبار خلفي ديناميكي مؤسسي.
    """
    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_total_drawdown_pct': max_total_dd_pct,
    })
    # وضع الاختبار: خفض الحد الأدنى لحجم الصفقة
    rm.min_order_value = 50.0
    
    trades = []
    equity_curve = [initial_balance]
    pos = 0
    entry_price = 0.0
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        atr = data['ATR'].iloc[i]
        signal = data['Final'].iloc[i]
        
        if pos != 0:
            stop_loss = entry_price - (pos * 2.0 * atr)
            if signal == -pos or (pos == 1 and price <= stop_loss) or (pos == -1 and price >= stop_loss):
                pnl = (price - entry_price) * pos * rm.calculate_position_size(entry_price, stop_loss, 1.0)
                rm.update_pnl(pnl)
                trades.append(pnl)
                pos = 0
                continue
                
        if pos == 0 and signal != 0:
            try:
                rm.check_circuit_breaker()
                stop_loss = price - (signal * 2.0 * atr)
                size = rm.calculate_position_size(price, stop_loss, 1.0)
                if size > 0:
                    pos = signal
                    entry_price = price
            except CircuitBreakerException:
                break
        
        equity_curve.append(float(rm.current_balance))
        
    return trades, equity_curve
