f = 'pairs_backtest_engine.py'
t = open(f, encoding='utf-8').read()

old_pnl_block = """                # Estimate position size (simplified for backtest: fixed risk per trade)
                size = 100000 * (risk_percent / 0.01) # Rough scaling based on risk
                commission = (size / 100000) * params.commission_per_lot * 2 # 2 legs
                slippage = params.slippage_pips * params.pip_size * size * 2 # 2 legs
                
                combined_pnl = spread_diff * size - commission - slippage"""

new_pnl_block = """                # FIX-INST: Fixed institutional size (0.1 lots) to survive spread divergences
                size_eur = 10000
                size_gbp = size_eur * hedge_ratio
                
                combined_pnl = spread_diff * size_eur
                
                # Realistic costs for 2 legs
                commission = ((size_eur + size_gbp) / 100000) * params.commission_per_lot
                slippage = ((size_eur + size_gbp) * params.slippage_pips * params.pip_size)
                
                combined_pnl -= (commission + slippage)"""

if old_pnl_block in t:
    t = t.replace(old_pnl_block, new_pnl_block)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Fixed Position Sizing in pairs_backtest_engine.py (0.1 lots)")
else:
    print("ERROR: Exact PnL block not found")

import ast
ast.parse(t)
print("AST OK")
