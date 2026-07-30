f = 'deflated_sharpe_calculator.py'
t = open(f, encoding='utf-8').read()

# 1. تحديث قائمة الاستراتيجيات لإضافة النموذج الجديد
old_strategies = '''    # All 16 strategy trials with their PF values
    strategies = [
        ("Range Hunter", 0.77),
        ("Asian Breakout", 0.81),
        ("Pairs Trading", 0.60),
        ("News Deviation", 0.21),
        ("Liquidity Absorption", 0.63),
        ("Long Lower Shadow", 1.25),
        ("High ATR", 1.07),
        ("Hour00", 1.02),
        ("Matrix", 0.91),
        ("Momentum", 0.88),
        ("Donchian", 0.98),
        ("MA Crossover", 1.00),
        ("Vol Compression", 0.89),
        ("Weekend Gap Fade", 1.23),
        ("Liquidity Sweep", 0.76),
        ("ML Meta-Labeler", 1.05),  # Our "best"
    ]'''
    
new_strategies = '''    # All 17 strategy trials with their PF values
    strategies = [
        ("Range Hunter", 0.77),
        ("Asian Breakout", 0.81),
        ("Pairs Trading", 0.60),
        ("News Deviation", 0.21),
        ("Liquidity Absorption", 0.63),
        ("Long Lower Shadow", 1.25),
        ("High ATR", 1.07),
        ("Hour00", 1.02),
        ("Matrix", 0.91),
        ("Momentum", 0.88),
        ("Donchian", 0.98),
        ("MA Crossover", 1.00),
        ("Vol Compression", 0.89),
        ("Weekend Gap Fade", 1.23),
        ("Liquidity Sweep", 0.76),
        ("ML Meta-Labeler (5Y)", 1.05),
        ("ML Continuous (20Y)", 2.33),  # NEW: Precision 63.64%, RRR 1.33 -> PF~2.33
    ]'''

if old_strategies in t:
    t = t.replace(old_strategies, new_strategies)
    print("SUCCESS: Added 20Y Continuous Model to DSR trials")

# 2. تحديث عدد العائدات (T) من 12 إلى 1255 (OOS bars)
old_t = 'num_returns=12,  # 12 trades in OOS'
new_t = 'num_returns=1255,  # 1255 OOS bars in 20Y test set'

if old_t in t:
    t = t.replace(old_t, new_t)
    print("SUCCESS: Updated OOS sample size (T=1255)")

open(f, 'w', encoding='utf-8').write(t)

import ast
ast.parse(t)
print("AST OK")
