"""
pairs_backtest_engine.py - Dedicated Backtest Engine for Statistical Arbitrage
Handles simultaneous dual-leg execution (Buy EUR, Sell GBP) and Combined PnL.
"""
import pandas as pd
import numpy as np
from risk_manager import RiskManager, CircuitBreakerException
from risk_guardian import RiskGuardian
from parameter_controller import params
import warnings
warnings.filterwarnings("ignore")

def dynamic_pairs_backtest(data, initial_balance=None, risk_percent=0.005, log_trades=False):
    if initial_balance is None:
        initial_balance = params.initial_capital
        
    rm = RiskManager(initial_balance, config={
        'base_risk_percent': risk_percent,
        'max_daily_loss_pct': params.max_daily_loss_pct,
        'max_total_drawdown_pct': params.max_total_drawdown_pct,
    })
    guardian = RiskGuardian(initial_equity=initial_balance)
    
    trades = []
    equity_curve = [initial_balance]
    trade_log = []
    
    # State variables for Pairs Trade
    in_trade = False
    trade_type = 0  # 1 for Long Spread, -1 for Short Spread
    entry_eur = 0.0
    entry_gbp = 0.0
    hedge_ratio = 0.0
    entry_time = None
    
    for i in range(len(data)):
        curr_time = data.index[i]
        eur_close = data["EURUSD_close"].iloc[i]
        gbp_close = data["GBPUSD_close"].iloc[i]
        signal = data["Pairs_Signal"].iloc[i]
        current_hedge = data["hedge_ratio"].iloc[i]
        
        try:
            rm.check_circuit_breaker(current_time=curr_time)
        except CircuitBreakerException:
            break
            
        # 1. Manage Open Trade (Exit Logic)
        if in_trade:
            if signal == 0: # Exit signal triggered
                # Calculate Combined PnL (Mathematical Fix)
                # Long Spread (trade_type=1): Buy EUR, Sell GBP
                # PnL = (EUR_exit - EUR_entry) - hedge_ratio * (GBP_exit - GBP_entry)
                # Short Spread (trade_type=-1): Sell EUR, Buy GBP
                # PnL = -(EUR_exit - EUR_entry) + hedge_ratio * (GBP_exit - GBP_entry)
                eur_diff = eur_close - entry_eur
                gbp_diff = gbp_close - entry_gbp
                
                spread_diff = trade_type * (eur_diff - (hedge_ratio * gbp_diff))
                
                # Estimate position size (simplified for backtest: fixed risk per trade)
                # FIX-INST: Fixed institutional size (0.1 lots) to survive spread divergences
                size_eur = 10000
                size_gbp = size_eur * hedge_ratio
                combined_pnl = spread_diff * size_eur
                commission = ((size_eur + size_gbp) / 100000) * params.commission_per_lot
                slippage = ((size_eur + size_gbp) * params.slippage_pips * params.pip_size)
                combined_pnl -= (commission + slippage)
                
                combined_pnl = spread_diff * size_eur - commission - slippage
                
                rm.update_pnl(combined_pnl)
                guardian.record_trade(combined_pnl, current_time=curr_time)
                trades.append(combined_pnl)
                
                if log_trades:
                    trade_log.append({
                        'entry_time': entry_time, 'exit_time': curr_time,
                        'type': 'PAIRS_EXIT', 'pnl': combined_pnl, 
                        'balance_after': float(rm.current_balance)
                    })
                
                in_trade = False
                trade_type = 0
                
        # 2. Check for New Entry
        if not in_trade and signal != 0:
            # Guardian Check
            guardian_decision = guardian.check_signal(
                current_time=curr_time,
                strategy_name="Pairs_Trading",
                proposed_size=100000,
                proposed_notional=100000 * eur_close,
                side='long' if signal == 1 else 'short'
            )
            
            if getattr(guardian_decision, 'approved', False):
                in_trade = True
                trade_type = signal
                entry_eur = eur_close
                entry_gbp = gbp_close
                hedge_ratio = current_hedge
                entry_time = curr_time
                
        equity_curve.append(float(rm.current_balance))
        
    if log_trades:
        return trades, equity_curve, trade_log
    return trades, equity_curve
