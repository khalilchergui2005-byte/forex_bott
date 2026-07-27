f = 'regime_detector.py'
t = open(f, encoding='utf-8').read()

# استبدال bfill() ب dropna() لمنع تسريب المستقبل
old_line = "        df.bfill(inplace=True)"
new_line = "        # FIX-INST: Removed bfill() to prevent look-ahead bias. Drop NaNs instead.\n        df.dropna(inplace=True)"

if old_line in t:
    t = t.replace(old_line, new_line)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Fixed Look-ahead bias in regime_detector.py (Replaced bfill with dropna)")
else:
    print("ERROR: bfill line not found")

import ast
ast.parse(t)
print("AST OK")
