f = 'meta_labeler.py'
t = open(f, encoding='utf-8').read()

# البحث عن الكود القديم لحساب TP و SL
old_barriers = """        if signal == 1:  # Long
                tp = entry + (tp_mult * atr)
                sl = entry - (sl_mult * atr)
            elif signal == -1:  # Short
                tp = entry - (tp_mult * atr)
                sl = entry + (sl_mult * atr)"""

# الكود الجديد: إضافة التكاليف إلى TP
new_barriers = """        # FIX-INST: Widen TP barrier by transaction costs to prevent learning false positives
            # Round-trip cost = Spread (entry+exit) + Slippage (entry+exit) + Commission
            # Assuming avg ATR ~20 pips, 2.7 pips cost = 0.135 ATR (using a safe fixed 0.15 ATR proxy)
            # Or better: calculate from avg_spread if available
            cost_in_price = data["avg_spread"].iloc[i] if "avg_spread" in data.columns else 0.00010  # Fallback 1.0 pip
            cost_in_price += 0.00005  # Slippage 0.5 pip
            cost_in_price += 0.00007  # Commission ~0.7 pip
            cost_in_atr = cost_in_price / (atr + 1e-10)
            
            if signal == 1:  # Long
                tp = entry + ((tp_mult + cost_in_atr) * atr)
                sl = entry - (sl_mult * atr)  # SL unchanged, realized loss is worse by cost
            elif signal == -1:  # Short
                tp = entry - ((tp_mult + cost_in_atr) * atr)
                sl = entry + (sl_mult * atr)"""

if old_barriers in t:
    t = t.replace(old_barriers, new_barriers)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Fixed Triple Barrier to account for transaction costs")
else:
    print("ERROR: Old barriers block not found")

import ast
ast.parse(t)
print("AST OK")
