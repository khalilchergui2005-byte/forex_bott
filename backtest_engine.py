"""
backtest_engine.py - المحرك الديناميكي الموحد + نظام التأكيد الهجومي
المشروع: الكيان الفيروسي - أقوى شركة تداول
يدعم: خروج جزئي، هدف ديناميكي، حماية Breakeven، ومضاعف رافعة ذكي.
"""
import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException
from data_loader import load_hourly

def dynamic_backtest(data=None, initial_balance=10000, risk_percent=0.005,
                     max_daily_loss_pct=0.02, max_total_dd_pct=0.10, log_trades=False):
    """
    المحرك الموحد مع نظام التداول الهجومي:
    - يضرب نسبة المخاطرة الأساسية في درجة التأكيد (Confirmation) لكل إشارة.
    - يخرج جزئياً ويحرك الوقف إلى نقطة الدخول.
    """
    if data is None:
        data = load_hourly()

    # التحقق من وجود عمود التأكيد
    use_confirmation = 'Confirmation' in data.columns

    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_total_drawdown_pct': max_total_dd_pct,
    })
    rm.min_order_value = 50.0

    trades = []
    equity_curve = [initial_balance]
    trade_log = []

    pos = 0
    entry_price = 0.0
    entry_time = None
    remaining_units = 0.0
    partial_exit_done = False

    avg_atr = data['ATR'].mean()

    for i in range(len(data)):
        price = data['close'].iloc[i]
        atr = data['ATR'].iloc[i]
        signal = data['Final'].iloc[i]
        current_time = data.index[i] if hasattr(data.index[i], 'strftime') else i

        # حساب نسبة المخاطرة الذكية
        if use_confirmation and signal != 0:
            conf_score = data['Confirmation'].iloc[i]
            # القيم المعتمدة: 0.25% إلى 1.0% بناءً على درجة التأكيد
            effective_risk = risk_percent * conf_score
            effective_risk = max(0.0025, min(0.01, effective_risk))  # حدود الأمان 0.25% - 1.0%
        else:
            effective_risk = risk_percent

        # حساب مضاعف الوقف الديناميكي
        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        stop_multiplier = max(1.5, min(2.5, 2.5 - (0.5 * vol_ratio)))

        # مضاعف الهدف (Take Profit) يعتمد على جودة الإشارة
        signal_quality = conf_score / 2.0 if use_confirmation and signal != 0 else 0.7
        take_profit_mult = 2.0 + (signal_quality * 2.0)  # بين 2.0 و 4.0
        target_price = entry_price + (pos * take_profit_mult * atr) if pos != 0 else 0

        # --- إدارة الصفقة المفتوحة ---
        if pos != 0:
            stop_loss = entry_price - (pos * stop_multiplier * atr)

            # 1. هل ضرب الهدف الأول (للخروج الجزئي)؟
            if not partial_exit_done:
                if (pos == 1 and price >= target_price) or (pos == -1 and price <= target_price):
                    half_units = remaining_units / 2
                    pnl = (price - entry_price) * pos * half_units
                    rm.update_pnl(pnl)
                    trades.append(pnl)
                    if log_trades:
                        trade_log.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': 'LONG' if pos == 1 else 'SHORT',
                            'entry_price': entry_price, 'exit_price': price,
                            'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                            'type': 'PARTIAL_TP'
                        })
                    remaining_units = half_units
                    partial_exit_done = True
                    stop_loss = entry_price  # حماية Breakeven
                    continue

            # 2. هل ضرب الهدف الثاني (الخروج الكامل)؟
            second_target = entry_price + (pos * 6.0 * atr)
            if partial_exit_done:
                if (pos == 1 and price >= second_target) or (pos == -1 and price <= second_target):
                    pnl = (price - entry_price) * pos * remaining_units
                    rm.update_pnl(pnl)
                    trades.append(pnl)
                    if log_trades:
                        trade_log.append({
                            'entry_time': entry_time, 'exit_time': current_time,
                            'direction': 'LONG' if pos == 1 else 'SHORT',
                            'entry_price': entry_price, 'exit_price': price,
                            'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                            'type': 'FULL_TP'
                        })
                    pos = 0
                    remaining_units = 0.0
                    partial_exit_done = False
                    continue

            # 3. هل ضرب وقف الخسارة أو انعكست الإشارة؟
            hit_stop = (pos == 1 and price <= stop_loss) or (pos == -1 and price >= stop_loss)
            signal_reverse = (signal == -pos)

            if hit_stop or signal_reverse:
                pnl = (price - entry_price) * pos * remaining_units
                rm.update_pnl(pnl)
                trades.append(pnl)
                if log_trades:
                    trade_log.append({
                        'entry_time': entry_time, 'exit_time': current_time,
                        'direction': 'LONG' if pos == 1 else 'SHORT',
                        'entry_price': entry_price, 'exit_price': price,
                        'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                        'type': 'STOP_LOSS' if hit_stop else 'SIGNAL_REVERSE'
                    })
                pos = 0
                remaining_units = 0.0
                partial_exit_done = False
                continue

        # --- فتح صفقة جديدة ---
        if pos == 0 and signal != 0:
            try:
                rm.check_circuit_breaker()
                stop_loss = price - (signal * stop_multiplier * atr)
                # استخدام effective_risk لحساب الحجم
                size = rm.calculate_position_size(price, stop_loss, effective_risk)
                if size > 0:
                    pos = signal
                    entry_price = price
                    entry_time = current_time
                    remaining_units = float(size)
                    partial_exit_done = False
            except CircuitBreakerException:
                break

        equity_curve.append(float(rm.current_balance))

    if log_trades:
        return trades, equity_curve, trade_log
    return trades, equity_curve
