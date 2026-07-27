f = 'spread_calculator.py'
t = open(f, encoding='utf-8').read()

old_line = 'df["hedge_ratio"] = cov / (var + 1e-10)'
new_line = 'df["hedge_ratio"] = (cov / (var + 1e-10)).clip(0.5, 1.5)  # FIX: Institutional Clipping to prevent math blowups'

if old_line in t:
    t = t.replace(old_line, new_line)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Applied Institutional Clipping to Hedge Ratio (0.5 to 1.5)")
else:
    print("ERROR: Line not found")

import ast
ast.parse(t)
print("AST OK")
