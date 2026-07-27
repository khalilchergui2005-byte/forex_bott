f = 'backtest_engine.py'
t = open(f, encoding='utf-8').read()

# البحث عن النمط المحدد لمسار Full TP
old_exit_2_strict = "pnl = (price - pos_data['pos']* slippage - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission"

if old_exit_2_strict in t:
    new_exit_2_strict = """exit_price = price - pos_data['pos'] * (half_spread_exit + slippage)
                        pnl = (exit_price - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission"""
    t = t.replace(old_exit_2_strict, new_exit_2_strict)
    print("SUCCESS: Fixed Exit 2 (Full TP) - Added missing spread via strict match")
else:
    print("ERROR: Strict Full TP pattern not found. Checking variations...")
    # Fallback to line-by-line replacement if the strict match fails
    lines = t.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'second_target' in line and 'pos_data' in line:
            # This is likely the Full TP condition block, inject the spread fix on the next lines
            new_lines.append(line)
            # Read the next lines until we find the pnl calculation
            while i < len(lines) - 1 and 'pnl' not in lines[i+1]:
                new_lines.append(lines[i+1])
                i += 1
            # Now we are at the line before pnl
            new_lines.append("                        real_spread_exit = data['avg_spread'].iloc[i] if 'avg_spread' in data.columns else params.spread_pips * params.pip_size")
            new_lines.append("                        half_spread_exit = real_spread_exit / 2")
            # The next line is the pnl calculation, replace it
            if i < len(lines) - 1:
                pnl_line = lines[i+1]
                indent = len(pnl_line) - len(pnl_line.lstrip())
                sp = ' ' * indent
                new_lines.append(sp + "exit_price = price - pos_data['pos'] * (half_spread_exit + slippage)")
                new_lines.append(sp + "pnl = (exit_price - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission")
                i += 1
        else:
            new_lines.append(line)
        i += 1
    t = '\n'.join(new_lines)
    print("SUCCESS: Applied fallback line-by-line fix for Full TP")

open(f, 'w', encoding='utf-8').write(t)

import ast
ast.parse(t)
print("AST OK")
