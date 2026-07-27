f = 'backtest_engine.py'
t = open(f, encoding='utf-8').read()

# ==========================================
# 1. إصلاح حساب الدخول (Entry) - استخدام السبريد الحقيقي
# ==========================================
old_entry = """                        half_spread = (params.spread_pips * params.pip_size) / 2
                        slippage = params.slippage_pips * params.pip_size
                        pos_data['entry'] = price + signal * half_spread + signal * slippage"""

new_entry = """                        # FIX-INST: Use real avg_spread from Dukascopy instead of flat constant
                        real_spread = data['avg_spread'].iloc[i] if 'avg_spread' in data.columns else params.spread_pips * params.pip_size
                        half_spread = real_spread / 2
                        slippage = params.slippage_pips * params.pip_size
                        pos_data['entry'] = price + signal * half_spread + signal * slippage"""

if old_entry in t:
    t = t.replace(old_entry, new_entry)
    print("SUCCESS: Fixed Entry to use real avg_spread")
else:
    print("ERROR: Entry block not found")

# ==========================================
# 2. إصلاح حساب الإغلاق (Exit) - إضافة السبريد المفقود
# ==========================================
# هناك 3 مسارات إغلاق، كلها تحتوي على نفس النمط الخاطئ
# النمط الحالي: pnl = (price - pos_data['pos'] * slippage - pos_data['entry']) * pos_data['pos'] * ...
# النمط الصحيح: pnl = (price - pos_data['pos'] * (half_spread_exit + slippage) - pos_data['entry']) * pos_data['pos'] * ...

# المسار 1: Partial TP
old_exit_1 = """                        commission = (half_units / 100000) * params.commission_per_lot
                        slippage = params.slippage_pips * params.pip_size
                        pnl = (price - pos_data['pos'] * slippage - pos_data['entry']) * pos_data['pos'] * half_units - commission"""

new_exit_1 = """                        # FIX-INST: Added missing spread cost at exit (was only charged at entry)
                        commission = (half_units / 100000) * params.commission_per_lot
                        real_spread_exit = data['avg_spread'].iloc[i] if 'avg_spread' in data.columns else params.spread_pips * params.pip_size
                        half_spread_exit = real_spread_exit / 2
                        slippage = params.slippage_pips * params.pip_size
                        exit_price = price - pos_data['pos'] * (half_spread_exit + slippage)
                        pnl = (exit_price - pos_data['entry']) * pos_data['pos'] * half_units - commission"""

if old_exit_1 in t:
    t = t.replace(old_exit_1, new_exit_1)
    print("SUCCESS: Fixed Exit 1 (Partial TP) - Added missing spread")
else:
    print("ERROR: Exit 1 block not found")

# المسار 2: Full TP
old_exit_2 = """                        commission = (pos_data['units'] / 100000) * params.commission_per_lot
                        slippage = params.slippage_pips * params.pip_size
                        pnl = (price - pos_data['pos']* slippage - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission"""

new_exit_2 = """                        # FIX-INST: Added missing spread cost at exit (was only charged at entry)
                        commission = (pos_data['units'] / 100000) * params.commission_per_lot
                        real_spread_exit = data['avg_spread'].iloc[i] if 'avg_spread' in data.columns else params.spread_pips * params.pip_size
                        half_spread_exit = real_spread_exit / 2
                        slippage = params.slippage_pips * params.pip_size
                        exit_price = price - pos_data['pos'] * (half_spread_exit + slippage)
                        pnl = (exit_price - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission"""

if old_exit_2 in t:
    t = t.replace(old_exit_2, new_exit_2)
    print("SUCCESS: Fixed Exit 2 (Full TP) - Added missing spread")
else:
    print("ERROR: Exit 2 block not found")

# المسار 3: Stop Loss / Signal Reverse
old_exit_3 = """                    commission = (pos_data['units'] / 100000) * params.commission_per_lot
                    slippage = params.slippage_pips * params.pip_size
                    pnl = (price - pos_data['pos'] * slippage - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission"""

new_exit_3 = """                    # FIX-INST: Added missing spread cost at exit (was only charged at entry)
                    commission = (pos_data['units'] / 100000) * params.commission_per_lot
                    real_spread_exit = data['avg_spread'].iloc[i] if 'avg_spread' in data.columns else params.spread_pips * params.pip_size
                    half_spread_exit = real_spread_exit / 2
                    slippage = params.slippage_pips * params.pip_size
                    exit_price = price - pos_data['pos'] * (half_spread_exit + slippage)
                    pnl = (exit_price - pos_data['entry']) * pos_data['pos'] * pos_data['units'] - commission"""

if old_exit_3 in t:
    t = t.replace(old_exit_3, new_exit_3)
    print("SUCCESS: Fixed Exit 3 (Stop Loss/Reverse) - Added missing spread")
else:
    print("ERROR: Exit 3 block not found")

open(f, 'w', encoding='utf-8').write(t)

import ast
ast.parse(t)
print("AST OK")
