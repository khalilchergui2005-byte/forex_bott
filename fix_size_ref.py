f = 'pairs_backtest_engine.py'
t = open(f, encoding='utf-8').read()

old_broken = """                combined_pnl = spread_diff * size - commission - slippage"""
new_fixed = """                combined_pnl = spread_diff * size_eur - commission - slippage"""

if old_broken in t:
    t = t.replace(old_broken, new_fixed)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Fixed size reference to size_eur")
else:
    print("WARN: Broken line not found. Trying alternative fix.")
    t = t.replace("spread_diff * size", "spread_diff * size_eur")
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Applied alternative fix")

import ast
ast.parse(t)
print("AST OK")
