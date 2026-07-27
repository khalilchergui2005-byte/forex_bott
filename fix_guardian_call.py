f = 'pairs_backtest_engine.py'
t = open(f, encoding='utf-8').read()

old_line = "guardian.settle_trade(combined_pnl, current_time=curr_time)"
new_line = "guardian.record_trade(combined_pnl, current_time=curr_time)"

if old_line in t:
    t = t.replace(old_line, new_line)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Fixed guardian method call to record_trade")
else:
    print("ERROR: Line not found")

import ast
ast.parse(t)
print("AST OK")
