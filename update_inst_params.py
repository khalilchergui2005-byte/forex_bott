# --- 1. تحديث parameter_controller.py بالإعدادات المثلى ---
f = 'parameter_controller.py'
t = open(f, encoding='utf-8').read()

# تحديث BB Mult
t = t.replace('rh_bb_mult: float = 2.0', 'rh_bb_mult: float = 2.6  # OPT-INST: Optimized for 5Y data')

# تحديث SL/TP الخاص بـ Signal
old_sltp = "'Signal': {'tp_mult': 2.0, 'sl_mult': 2.0, 'second_target_mult': 4.0},  # Optimized for Mean Reversion (Range Hunter, Shadow)"
new_sltp = "'Signal': {'tp_mult': 1.5, 'sl_mult': 2.5, 'second_target_mult': 3.0},  # OPT-INST: Optimized for 5Y Mean Reversion"

if old_sltp in t:
    t = t.replace(old_sltp, new_sltp)
else:
    # Fallback if the exact string isn't found, try a broader replace
    import re
    t = re.sub(r"'Signal': \{[^}]+\}", new_sltp, t)

open(f, 'w', encoding='utf-8').write(t)
print("SUCCESS: Updated parameter_controller.py with Institutional Optimal Parameters")

import ast
ast.parse(t)
print("AST OK")
