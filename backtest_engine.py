"""
backtest_engine.py - محرك المحفظة الموحد (Portfolio Engine)
المشروع: الكيان الفيروسي - أقوى شركة تداول
يدعم: استراتيجيات متعددة، خروج جزئي، هدف ديناميكي، Sniping، والمزيد.
"""
import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException
from data_loader import load_hourly

def dynamic_backtest(data=None, initial_balance=10000, risk_percent=0.005,
                     max_daily_loss_pct=0.02, max_total_dd_pct=0.10, log_trades=False,
                     strategy_cols=None):
    """
    محرك المحفظة الموحد.
    يمكنه التداول بناءً على استراتيجية واحدة (عمود 'Final') أو عدة استراتيجيات.
    """
    if data is None:
        data = load_hourly()

    # تحديد الأعمدة التي سيتم التداول عليها
    if strategy_cols is None:
        strategy_cols = ['Final']

    # إضافة عمود 'Final' افتراضي إذا لم يكن موجوداً
    for col in strategy_cols:
        if col not in data.columns:
            data[col] = 0

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

    # لكل استراتيجية، نحتفظ بمركزها الخاص
    positions = {col: {'pos': 0, 'entry': 0.0, 'time': None, 'units': 0.0, 'partial': False}
                 for col in strategy_cols}

    avg_atr = data['ATR'].mean()

    for i in range(len(data)):
        price = data['close'].iloc[i]
        atr = data['ATR'].iloc[i]
        current_time = data.index[i] if hasattr(data.index[i], 'strftime') else i

        vol_ratio = atr / avg_atr if avg_atr > 0 else 1.0
        stop_multiplier = max(1.5, min(2.5, 2.5 - (0.5 * vol_ratio)))

        # --- معالجة جميع الاستراتيجيات ---
        for col in strategy_cols:
            signal = data[col].iloc[i]
            pos_data = positions[col]

            # حساب نسبة المخاطرة الذكية
            conf_score = 1.0
            if use_confirmation and signal != 0:
                conf_score = data['Confirmation'].iloc[i]
                effective_risk = risk_percent * conf_score
                effective_risk = max(0.0025, min(0.01, effective_risk))
            else:
                effective_risk = risk_percent

            signal_quality = conf_score / 2.0 if use_confirmation and signal != 0 else 0.7
            take_profit_mult = 2.0 + (signal_quality * 2.0)
            target_price = pos_data['entry'] + (pos_data['pos'] * take_profit_mult * atr) if pos_data['pos'] != 0 else 0

            # إدارة الصفقة المفتوحة
            if pos_data['pos'] != 0:
                stop_loss = pos_data['entry'] - (pos_data['pos'] * stop_multiplier * atr)

                # هدف أول (خروج جزئي)
                if not pos_data['partial']:
                    if (pos_data['pos'] == 1 and price >= target_price) or (pos_data['pos'] == -1 and price <= target_price):
                        half_units = pos_data['units'] / 2
                        pnl = (price - pos_data['entry']) * pos_data['pos'] * half_units
                        rm.update_pnl(pnl)
                        trades.append(pnl)
                        if log_trades:
                            trade_log.append({
                                'strategy': col, 'entry_time': pos_data['time'], 'exit_time': current_time,
                                'direction': 'LONG' if pos_data['pos'] == 1 else 'SHORT',
                                'entry_price': pos_data['entry'], 'exit_price': price,
                                'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                                'type': 'PARTIAL_TP'
                            })
                        pos_data['units'] = half_units
                        pos_data['partial'] = True
                        stop_loss = pos_data['entry']
                        continue

                # هدف ثاني (خروج كامل)
                second_target = pos_data['entry'] + (pos_data['pos'] * 6.0 * atr)
                if pos_data['partial']:
                    if (pos_data['pos'] == 1 and price >= second_target) or (pos_data['pos'] == -1 and price <= second_target):
                        pnl = (price - pos_data['entry']) * pos_data['pos'] * pos_data['units']
                        rm.update_pnl(pnl)
                        trades.append(pnl)
                        if log_trades:
                            trade_log.append({
                                'strategy': col, 'entry_time': pos_data['time'], 'exit_time': current_time,
                                'direction': 'LONG' if pos_data['pos'] == 1 else 'SHORT',
                                'entry_price': pos_data['entry'], 'exit_price': price,
                                'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                                'type': 'FULL_TP'
                            })
                        pos_data['pos'] = 0
                        pos_data['units'] = 0.0
                        pos_data['partial'] = False
                        continue

                # وقف خسارة أو انعكاس إشارة
                hit_stop = (pos_data['pos'] == 1 and price <= stop_loss) or (pos_data['pos'] == -1 and price >= stop_loss)
                signal_reverse = (signal == -pos_data['pos'])
                if hit_stop or signal_reverse:
                    pnl = (price - pos_data['entry']) * pos_data['pos'] * pos_data['units']
                    rm.update_pnl(pnl)
                    trades.append(pnl)
                    if log_trades:
                        trade_log.append({
                            'strategy': col, 'entry_time': pos_data['time'], 'exit_time': current_time,
                            'direction': 'LONG' if pos_data['pos'] == 1 else 'SHORT',
                            'entry_price': pos_data['entry'], 'exit_price': price,
                            'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                            'type': 'STOP_LOSS' if hit_stop else 'SIGNAL_REVERSE'
                        })
                    pos_data['pos'] = 0
                    pos_data['units'] = 0.0
                    pos_data['partial'] = False
                    continue

            # فتح صفقة جديدة
            if pos_data['pos'] == 0 and signal != 0:
                try:
                    rm.check_circuit_breaker()
                    stop_loss = price - (signal * stop_multiplier * atr)
                    size = rm.calculate_position_size(price, stop_loss, effective_risk)
                    if size > 0:
                        # Sniping Mode
                        if use_confirmation and conf_score >= 2.0:
                            SNIPING_MULTIPLIER = 1.5
                            size = int(size * SNIPING_MULTIPLIER)
                        pos_data['pos'] = signal
                        pos_data['entry'] = price
                        pos_data['time'] = current_time
                        pos_data['units'] = float(size)
                        pos_data['partial'] = False
                except CircuitBreakerException:
                    break

        equity_curve.append(float(rm.current_balance))

    if log_trades:
        return trades, equity_curve, trade_log
    return trades, equity_curve
