f = 'pairs_backtest_engine.py'
t = open(f, encoding='utf-8').read()

old_pnl = """                # Calculate Combined PnL
                # Long Spread (Buy EUR, Sell GBP): PnL = (EUR_exit - EUR_entry) - hedge * (GBP_exit - GBP_entry)
                # Short Spread (Sell EUR, Buy GBP): PnL = -(EUR_exit - EUR_entry) + hedge * (GBP_exit - GBP_entry)
                eur_pnl = trade_type * (eur_close - entry_eur)
                gbp_pnl = -trade_type * hedge_ratio * (gbp_close - entry_gbp)
                
                # Estimate position size (simplified for backtest: fixed risk per trade)
                # Assume we trade 1 standard lot (100k) of EUR, and hedge_ratio lots of GBP
                size = 100000 * (risk_percent / 0.01) # Rough scaling based on risk
                commission = (size / 100000) * params.commission_per_lot * 2 # 2 legs
                slippage = params.slippage_pips * params.pip_size * size * 2 # 2 legs
                
                combined_pnl = (eur_pnl - gbp_pnl) * size - commission - slippage"""

new_pnl = """                # Calculate Combined PnL (Mathematical Fix)
                # Long Spread (trade_type=1): Buy EUR, Sell GBP
                # PnL = (EUR_exit - EUR_entry) - hedge_ratio * (GBP_exit - GBP_entry)
                # Short Spread (trade_type=-1): Sell EUR, Buy GBP
                # PnL = -(EUR_exit - EUR_entry) + hedge_ratio * (GBP_exit - GBP_entry)
                eur_diff = eur_close - entry_eur
                gbp_diff = gbp_close - entry_gbp
                
                spread_diff = trade_type * (eur_diff - (hedge_ratio * gbp_diff))
                
                # Estimate position size (simplified for backtest: fixed risk per trade)
                size = 100000 * (risk_percent / 0.01) 
                commission = (size / 100000) * params.commission_per_lot * 2 # 2 legs
                slippage = params.slippage_pips * params.pip_size * size * 2 # 2 legs
                
                combined_pnl = spread_diff * size - commission - slippage"""

if old_pnl in t:
    t = t.replace(old_pnl, new_pnl)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Fixed Combined PnL mathematical formula")
else:
    print("ERROR: Exact PnL block not found")

import ast
ast.parse(t)
print("AST OK")
