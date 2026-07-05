import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException

def dynamic_backtest(data, initial_balance, risk_percent=0.005, max_daily_loss_pct=0.02, max_total_dd_pct=0.10):
    """
    محرك اختبار خلفي ديناميكي مؤسسي.
    لا يحتوي على أي قيم ثابتة. مضاعف الوقف ديناميكي.
    """
    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_total_drawdown_pct': max_total_dd_pct,
    })
    rm.min_order_value = 50.0
    
    trades = []
    equity_curve = [initial_balance]
    pos = 0
    entry_price = 0.0
    
    # حساب متوسط التقلب التاريخي مرة واحدة
    avg_atr = data['ATR'].mean()
    
    for i in range(len(data)):
        price = data['close'].iloc[i]
        atr = data['ATR'].iloc[i]
        signal = data['Final'].iloc[i]
        
        # مضاعف وقف ديناميكي: يعتمد على نسبة التقلب الحالي إلى المتوسط
        if avg_atr > 0:
            volatility_ratio = atr / avg_atr
        else:
            volatility_ratio = 1.0
        # إذا التقلب أعلى من المتوسط → نضيق الوقف (1.5x). إذا أقل → نوسع (2.5x)
        stop_multiplier = 2.5 - (0.5 * volatility_ratio)
        stop_multiplier = max(1.5, min(2.5, stop_multiplier))
        
        if pos != 0:
            stop_loss = entry_price - (pos * stop_multiplier * atr)
            if signal == -pos or (pos == 1 and price <= stop_loss) or (pos == -1 and price >= stop_loss):
                pnl = (price - entry_price) * pos * rm.calculate_position_size(entry_price, stop_loss, 1.0)
                rm.update_pnl(pnl)
                trades.append(pnl)
                pos = 0
                continue
                
        if pos == 0 and signal != 0:
            try:
                rm.check_circuit_breaker()
                stop_loss = price - (signal * stop_multiplier * atr)
                size = rm.calculate_position_size(price, stop_loss, 1.0)
                if size > 0:
                    pos = signal
                    entry_price = price
            except CircuitBreakerException:
                break
        
        equity_curve.append(float(rm.current_balance))
        
    return trades, equity_curve
