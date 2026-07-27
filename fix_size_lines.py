f = 'pairs_backtest_engine.py'
lines = open(f, encoding='utf-8').read().split('\n')
new_lines = []
skip = False

for line in lines:
    if 'size = 100000' in line:
        indent = len(line) - len(line.lstrip())
        sp = ' ' * indent
        new_lines.append(sp + '# FIX-INST: Fixed institutional size (0.1 lots) to survive spread divergences')
        new_lines.append(sp + 'size_eur = 10000')
        new_lines.append(sp + 'size_gbp = size_eur * hedge_ratio')
        new_lines.append(sp + 'combined_pnl = spread_diff * size_eur')
        new_lines.append(sp + 'commission = ((size_eur + size_gbp) / 100000) * params.commission_per_lot')
        new_lines.append(sp + 'slippage = ((size_eur + size_gbp) * params.slippage_pips * params.pip_size)')
        new_lines.append(sp + 'combined_pnl -= (commission + slippage)')
        skip = True
    elif skip and ('commission' in line or 'slippage' in line or 'combined_pnl' in line or 'size' in line):
        continue
    else:
        skip = False
        new_lines.append(line)

t = '\n'.join(new_lines)
open(f, 'w', encoding='utf-8').write(t)

import ast
ast.parse(t)
print("SUCCESS: Fixed Position Sizing using line-by-line replacement")
print("AST OK")
