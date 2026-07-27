f = 'meta_labeler.py'
t = open(f, encoding='utf-8').read()

old_signal = '''    def create_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a simple momentum base signal (Buy if ret_5 > 0, Sell if ret_5 < 0)."""
        data = df.copy()
        data["base_signal"] = 0
        data.loc[data["ret_5"] > 0, "base_signal"] = 1
        data.loc[data["ret_5"] < 0, "base_signal"] = -1
        return data'''

new_signal = '''    def create_base_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a logical base signal (Trend + Order Flow Confirmation)."""
        data = df.copy()
        data["base_signal"] = 0
        
        # Buy: Trend is UP and Order Flow (CVD) is positive
        buy_cond = (data["regime_name"] == "TREND_UP") & (data["cvd"] > 0)
        data.loc[buy_cond, "base_signal"] = 1
        
        # Sell: Trend is DOWN and Order Flow (CVD) is negative
        sell_cond = (data["regime_name"] == "TREND_DOWN") & (data["cvd"] < 0)
        data.loc[sell_cond, "base_signal"] = -1
        
        return data'''

if old_signal in t:
    t = t.replace(old_signal, new_signal)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Upgraded Base Signal to use Trend + Order Flow")
else:
    print("ERROR: Old signal block not found")

import ast
ast.parse(t)
print("AST OK")
