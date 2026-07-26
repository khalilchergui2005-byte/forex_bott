# --- 1. تحديث parameter_controller.py بالإعدادات المثلى ---
f = 'parameter_controller.py'
t = open(f, encoding='utf-8').read()

old_sltp = "'Asian_Breakout_Signal': {'tp_mult': 3.0, 'sl_mult': 1.5, 'second_target_mult': 6.0},  # Session Breakout (Asian)"
new_sltp = "'Asian_Breakout_Signal': {'tp_mult': 2.5, 'sl_mult': 2.0, 'second_target_mult': 5.0},  # OPT-INST: Asian Breakout"

if old_sltp in t:
    t = t.replace(old_sltp, new_sltp)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Updated parameter_controller.py with Asian Breakout Optimal Parameters")
else:
    print("WARN: Asian SL/TP pattern not found, trying regex.")
    import re
    t = re.sub(r"'Asian_Breakout_Signal': \{[^}]+\}", new_sltp, t)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Updated via regex")

import ast
ast.parse(t)
print("AST OK")
