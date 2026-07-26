"""
backtest_engine.py - Unified portfolio engine + Meta-Labeling Filter
Project: Viral Entity - Strongest Trading Firm
Supports: Multiple strategies, partial exit, fixed target, Sniping, Turbo Volatility.
Filter layers (SmartScore, Fear, Shock&Awe) act as gatekeepers, NOT position size multipliers.
"""
import pandas as pd
from risk_manager import RiskManager, CircuitBreakerException
from risk_guardian import RiskGuardian  # Phase 4: Guardian integration
from data_loader import load_hourly
import random
from parameter_controller import params

def dynamic_backtest(data=None, initial_balance=None, risk_percent=0.005,
                     max_daily_loss_pct=0.02, max_total_drawdown_pct=0.10, log_trades=False,
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
        'max_total_drawdown_pct': max_total_drawdown_pct,
    })
    guardian = RiskGuardian(initial_equity=initial_balance)  # Phase 4: Initialize Guardian
    # BUG-C02 FIX: removed fake rm.min_order_value (dead attribute)

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

            # BUG-C01 FIX: signal_quality before effective_risk
            signal_quality = conf_score / 2.0 if use_confirmation and signal != 0 else 0.7

            if use_confirmation and signal != 0:
                effective_risk = risk_percent * signal_quality
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
                        commission = (half_units / 100000) * params.commission_per_lot
                        slippage = params.slippage_pips * params.pip_size
                        pnl = (price - pos_data['pos'] * slippage - pos_data['entry']) * pos_data['pos'] * half_units - commission
                        rm.update_pnl(pnl)
                        guardian.record_trade(pnl, current_time=current_time)  # Phase 4: Settle trade
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
                        commission = (pos_data['units'] / 100000) * params.commission_per_lot
                        slippage = params.slippage_pips * params.pip_size
                        pnl = (price - pos_data['pos'] * slippage - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission
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
                    commission = (pos_data['units'] / 100000) * params.commission_per_lot
                    slippage = params.slippage_pips * params.pip_size
                    pnl = (price - pos_data['pos'] * slippage - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission
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
                # --- Temporal Edge Filter (Council Atom 7.4) ---
                current_hour = data.index[i].hour if hasattr(data.index[i], 'hour') else current_time.hour
                if current_hour in params.blocked_hours:
                    continue  # Blocked hour, skip trade
                # ---------------------------------------------
                # --- Meta-Labeling Filter (Council Recommendation) ---
                # These act as GATEKEEPERS only. Position size is NOT modified.
                smart_score_ok = (conf_score >= params.min_smart_score_filter)
                fear_ok = (fear_mult >= params.max_fear_level_filter)
                shock_ok = (not params.require_shock_awe_filter) or (consecutive_wins >= params.shock_awe_threshold)

                if not (smart_score_ok and fear_ok and shock_ok):
                    continue  # Gate rejects this trade
                # -------------------------------------------------
                
                # --- Institutional Governor & Immune System ---
                is_mean_reversion = (col == "Signal" or "Absorption" in col)
                is_trend = ("Trend" in col or "trend" in col)
                is_breakout = ("Breakout" in col or "Asian" in col or "London" in col)
                if "regime_name" in data.columns:
                    current_regime = data["regime_name"].iloc[i]
                    if current_regime == "EXTREME_VOLATILE": continue
                    if is_mean_reversion:
                        if current_regime not in ["RANGE", "QUIET"]: continue
                    elif is_trend:
                        if current_regime not in ["TREND_UP", "TREND_DOWN"]: continue
                    elif is_breakout:
                        if current_regime not in ["VOLATILE", "TREND_UP", "TREND_DOWN"]: continue
                # Identify strategy type by column name
                is_mean_reversion = (col == 'Signal' or 'Absorption' in col or 'Reversion' in col)
                is_trend = ('Trend' in col or 'trend' in col)
                is_breakout = ('Breakout' in col or 'Asian' in col or 'London' in col)

                # 1. Regime Filter (Strategy-Specific)
                if 'regime_name' in data.columns:
                    current_regime = data['regime_name'].iloc[i]
                    if is_mean_reversion:
                        if current_regime not in ['RANGE', 'QUIET']:
                            continue
                    elif is_trend:
                        if current_regime not in ['TREND_UP', 'TREND_DOWN']:
                            continue
                    elif is_breakout:
                        if current_regime not in ['VOLATILE', 'TREND_UP', 'TREND_DOWN']:
                            continue

                # 2. Divergence Filter (Disabled for Mean Reversion)
                if not is_mean_reversion:
                    if signal == 1 and 'bull_div' in data.columns and data['bull_div'].iloc[i]:
                        continue
                    if signal == -1 and 'bear_div' in data.columns and data['bear_div'].iloc[i]:
                        continue

                # 3. Exhaustion Filter (Disabled for Mean Reversion)
                if not is_mean_reversion:
                    if 'exhaustion' in data.columns and data['exhaustion'].iloc[i]:
                        continue
                # ---------------------------------------------

                try:
                    rm.check_circuit_breaker(current_time=current_time)
                    # ARCH-02 FIX: Strategy-Specific Trade Management
                    curr_tp = tp_mult
                    curr_sl = sl_mult
                    if hasattr(params, 'strategy_sl_tp') and col in params.strategy_sl_tp:
                        strat_cfg = params.strategy_sl_tp[col]
                        curr_tp = strat_cfg.get('tp_mult', tp_mult)
                        curr_sl = strat_cfg.get('sl_mult', sl_mult)
                    take_profit_mult = curr_tp + (signal_quality * 1.0)
                    initial_stop = price - (signal * curr_sl * atr)
                    initial_target = price + (signal * take_profit_mult * atr)
                    size = rm.calculate_position_size(price, initial_stop, signal_quality=1.0, risk_fraction=effective_risk)
                    if size > 0:
                        # --- Phase 4: Guardian Risk Check ---
                        guardian_decision = guardian.check_signal(
                            current_time=current_time,
                            strategy_name=col,
                            proposed_size=size,
                            proposed_notional=size * price,
                            side='long' if signal == 1 else 'short'
                        )
                        if not getattr(guardian_decision, 'approved', False):
                            continue  # Guardian rejected the trade
                        # ---------------------------------------
                        # Atom 4.5: Simulate order rejection
                        if params.rejection_rate > 0 and random.random() < params.rejection_rate:
                            continue
                        if use_confirmation and conf_score >= params.sniping_threshold:
                            size = int(size * params.sniping_multiplier)
                            # Atom 4.5: Simulate partial fill
                            if params.partial_fill_probability > 0 and random.random() < params.partial_fill_probability:
                                fill_ratio = random.uniform(params.partial_fill_min_ratio, 1.0)
                                size = int(size * fill_ratio)
                                if size <= 0:
                                    continue
                        pos_data['pos'] = signal
                        half_spread = (params.spread_pips * params.pip_size) / 2
                        slippage = params.slippage_pips * params.pip_size
                        pos_data['entry'] = price + signal * half_spread + signal * slippage
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


