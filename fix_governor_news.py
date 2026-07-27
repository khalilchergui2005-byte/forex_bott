f = 'backtest_engine.py'
t = open(f, encoding='utf-8').read()

old_defs = """is_mean_reversion = (col == 'Signal' or 'Absorption' in col)
is_trend = ('Trend' in col or 'trend' in col or 'News' in col)
is_breakout = ('Breakout' in col or 'Asian' in col or 'London' in col)"""

new_defs = """is_mean_reversion = (col == 'Signal' or 'Absorption' in col)
is_trend = ('Trend' in col or 'trend' in col)
is_news = ('News' in col)
is_breakout = ('Breakout' in col or 'Asian' in col or 'London' in col)"""

if old_defs in t:
    t = t.replace(old_defs, new_defs)
    print("SUCCESS: Added is_news definition")

old_rules = """                    # 2. Governor: Strategy-Specific Regime Enforcement
                    if is_mean_reversion:
                        if current_regime not in ['RANGE', 'QUIET']:
                            continue
                    elif is_trend:
                        if current_regime not in ['TREND_UP', 'TREND_DOWN']:
                            continue
                    elif is_breakout:
                        if current_regime not in ['VOLATILE', 'TREND_UP', 'TREND_DOWN']:
                            continue"""

new_rules = """                    # 2. Governor: Strategy-Specific Regime Enforcement
                    if is_mean_reversion:
                        if current_regime not in ['RANGE', 'QUIET']:
                            continue
                    elif is_trend:
                        if current_regime not in ['TREND_UP', 'TREND_DOWN']:
                            continue
                    elif is_news:
                        # News strategies thrive on volatility and momentum
                        if current_regime not in ['VOLATILE', 'TREND_UP', 'TREND_DOWN']:
                            continue
                    elif is_breakout:
                        if current_regime not in ['VOLATILE', 'TREND_UP', 'TREND_DOWN']:
                            continue"""

if old_rules in t:
    t = t.replace(old_rules, new_rules)
    print("SUCCESS: Governor updated to allow News in VOLATILE regimes")

open(f, 'w', encoding='utf-8').write(t)

import ast
ast.parse(t)
print("AST OK")
