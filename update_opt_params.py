f = 'parameter_controller.py'
t = open(f, encoding='utf-8').read()

# تحديث إعدادات استراتيجية Signal بالإعدادات المثلى التي وجدها المحسن
old_config = "'Signal': {'tp_mult': 1.5, 'sl_mult': 1.0, 'second_target_mult': 3.0}"
new_config = "'Signal': {'tp_mult': 2.0, 'sl_mult': 2.0, 'second_target_mult': 4.0}  # Optimized for Mean Reversion"

if old_config in t:
    t = t.replace(old_config, new_config)
    open(f, 'w', encoding='utf-8').write(t)
    import ast
    ast.parse(t)
    print("parameter_controller.py Updated with Optimized SL/TP & AST OK")
else:
    print("Pattern not found or already updated.")
