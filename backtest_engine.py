"""
backtest_engine.py - Unified portfolio engine + Meta-Labeling Filter
Project: Viral Entity - Strongest Trading Firm
Supports: Multiple strategies, partial exit, fixed target, Sniping, Turbo Volatility.
Filter layers (SmartScore, Fear, Shock&Awe) act as gatekeepers, NOT position size multipliers.
"""
import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException
from data_loader import load_hourly
from parameter_controller import params

def dynamic_backtest(data=None, initial_balance=None, risk_percent=0.005,
                     max_daily_loss_pct=0.02, max_total_dd_pct=0.10, log_trades=False,
                     strategy_cols=None):
    if data is None:
        data = load_hourly()

    if initial_balance is None:
        initial_balance = params.initial_capital

    if 'ATR' not in data.columns:
        data['ATR'] = (data['high'] - data['low']).rolling(14).mean()

    if strategy_cols is None:
        strategy_cols = ['Final']

    for col in strategy_cols:
        if col not in data.columns:
            data[col] = 0

    use_confirmation = 'Confirmation' in data.columns
    use_volatility = 'VolClass' in data.columns

    data['avg_atr'] = data['ATR'].expanding(min_periods=30).mean()

    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': max_daily_loss_pct,
        'max_total_drawdown_pct': max_total_dd_pct,
    })
    rm.min_order_value = params.min_order_value

    trades = []
    equity_curve = [initial_balance]
    trade_log = []

    positions = {col: {'pos': 0, 'entry': 0.0, 'time': None, 'units': 0.0, 'partial': False,
                       'target': 0.0, 'stop_loss': 0.0}
                 for col in strategy_cols}

    # Shock & Awe state
    consecutive_wins = 0

    for i in range(len(data)):
        price = data['close'].iloc[i]
        atr = data['ATR'].iloc[i]
        current_time = data.index[i] if hasattr(data.index[i], 'strftime') else i

        # Fear multiplier calculation (used ONLY for filtering, NOT for sizing)
        fear_mult = 0.5 if (atr / price) > params.fear_threshold else 1.0

        avg_atr_val = data['avg_atr'].iloc[i]
        if pd.isna(avg_atr_val) or avg_atr_val == 0:
            avg_atr_val = data['ATR'].iloc[:i+1].mean()
        vol_ratio = atr / avg_atr_val if avg_atr_val > 0 else 1.0
        stop_multiplier = max(params.stop_min_mult, min(params.stop_max_mult, params.stop_max_mult - (params.stop_volatility_factor * vol_ratio)))

        is_volatile = False
        if use_volatility:
            is_volatile = (data['VolClass'].iloc[i] == 2)

        if is_volatile:
            tp_mult = params.turbo_tp_mult
            sl_mult = params.turbo_sl_mult
        else:
            tp_mult = params.normal_tp_mult
            sl_mult = stop_multiplier

        for col in strategy_cols:
            signal = data[col].iloc[i]
            pos_data = positions[col]

            conf_score = 1.0
            if use_confirmation and signal != 0:
                conf_score = data['Confirmation'].iloc[i]
                effective_risk = risk_percent * conf_score
                effective_risk = max(0.0025, min(0.01, effective_risk))
            else:
                effective_risk = risk_percent

            # Shock & Awe overlay (for sizing)
            if consecutive_wins >= params.shock_awe_threshold:
                effective_risk *= params.shock_awe_multiplier
                effective_risk = min(effective_risk, 0.015)

            if pos_data['pos'] != 0:
                stop_loss = pos_data['stop_loss']
                target_price = pos_data['target']

                if not pos_data['partial']:
                    if (pos_data['pos'] == 1 and price >= target_price) or (pos_data['pos'] == -1 and price <= target_price):
                        half_units = pos_data['units'] * params.partial_exit_ratio
                        pnl = (price - pos_data['entry']) * pos_data['pos'] * half_units
                        rm.update_pnl(pnl)
                        trades.append(pnl)
                        if pnl > 0:
                            consecutive_wins += 1
                        else:
                            consecutive_wins = 0
                        if log_trades:
                            trade_log.append({
                                'strategy': col, 'entry_time': pos_data['time'], 'exit_time': current_time,
                                'direction': 'LONG' if pos_data['pos'] == 1 else 'SHORT',
                                'entry_price': pos_data['entry'], 'exit_price': price,
                                'net_pnl': pnl, 'balance_after': float(rm.current_balance),
                                'type': 'PARTIAL_TP'
                            })
                        pos_data['units'] -= half_units
                        pos_data['partial'] = True
                        pos_data['stop_loss'] = pos_data['entry']
                        continue

                second_target = pos_data['entry'] + (pos_data['pos'] * params.second_target_atr_mult * atr)
                if pos_data['partial']:
                    if (pos_data['pos'] == 1 and price >= second_target) or (pos_data['pos'] == -1 and price <= second_target):
                        pnl = (price - pos_data['entry']) * pos_data['pos'] * pos_data['units']
                        rm.update_pnl(pnl)
                        trades.append(pnl)
                        if pnl > 0:
                            consecutive_wins += 1
                        else:
                            consecutive_wins = 0
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

                hit_stop = (pos_data['pos'] == 1 and price <= stop_loss) or (pos_data['pos'] == -1 and price >= stop_loss)
                signal_reverse = (signal == -pos_data['pos'])
                if hit_stop or signal_reverse:
                    pnl = (price - pos_data['entry']) * pos_data['pos'] * pos_data['units']
                    rm.update_pnl(pnl)
                    trades.append(pnl)
                    if pnl > 0:
                        consecutive_wins += 1
                    else:
                        consecutive_wins = 0
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

            if pos_data['pos'] == 0 and signal != 0:
                # --- Meta-Labeling Filter (Council Recommendation) ---
                # These act as GATEKEEPERS only. Position size is NOT modified.
                smart_score_ok = (conf_score >= params.min_smart_score_filter)
                fear_ok = (fear_mult >= params.max_fear_level_filter)
                shock_ok = (not params.require_shock_awe_filter) or (consecutive_wins >= params.shock_awe_threshold)

                if not (smart_score_ok and fear_ok and shock_ok):
                    continue  # Gate rejects this trade
                # -------------------------------------------------

                try:
                    rm.check_circuit_breaker()
                    signal_quality = conf_score / 2.0 if use_confirmation and signal != 0 else 0.7
                    take_profit_mult = tp_mult + (signal_quality * 1.0)
                    initial_stop = price - (signal * sl_mult * atr)
                    initial_target = price + (signal * take_profit_mult * atr)
                    size = rm.calculate_position_size(price, initial_stop, effective_risk)
                    if size > 0:
                        if use_confirmation and conf_score >= params.sniping_threshold:
                            size = int(size * params.sniping_multiplier)
                        pos_data['pos'] = signal
                        pos_data['entry'] = price
                        pos_data['time'] = current_time
                        pos_data['units'] = float(size)
                        pos_data['partial'] = False
                        pos_data['target'] = initial_target
                        pos_data['stop_loss'] = initial_stop
                except CircuitBreakerException:
                    break

        equity_curve.append(float(rm.current_balance))

    if log_trades:
        return trades, equity_curve, trade_log
    return trades, equity_curve
