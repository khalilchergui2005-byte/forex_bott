import re

f = 'parameter_controller.py'
t = open(f, encoding='utf-8').read()

# سنقوم باستبدال كافة الأسطر المتراكمة بين 'strategy_sl_tp' و نهاية القاموس
# بشكل نظيف تماماً يحتوي على الاستراتيجيات الحالية فقط.
pattern = r"strategy_sl_tp: Dict\[str, Dict\[str, float\]\] = field\(default_factory=lambda: \{.*?\}\)"
replacement = """strategy_sl_tp: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'Signal': {'tp_mult': 2.0, 'sl_mult': 2.0, 'second_target_mult': 4.0},  # Mean Reversion
        'Asian_Breakout_Signal': {'tp_mult': 2.5, 'sl_mult': 2.0, 'second_target_mult': 5.0},  # Session Breakout
        'Absorption_Signal': {'tp_mult': 3.0, 'sl_mult': 1.5, 'second_target_mult': 6.0},  # Smart Money
        'Pairs_Signal': {'tp_mult': 0, 'sl_mult': 0, 'second_target_mult': 0}  # Pairs Trading (Exit on Z=0)
    })"""

t_new, n = re.subn(pattern, replacement, t, flags=re.DOTALL)

if n > 0:
    open(f, 'w', encoding='utf-8').write(t_new)
    print("SUCCESS: Cleaned and rebuilt strategy_sl_tp dictionary")
else:
    print("ERROR: Could not find the strategy_sl_tp pattern")

import ast
ast.parse(t_new)
print("AST OK")
