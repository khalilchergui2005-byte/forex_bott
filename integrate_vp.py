f = 'feature_engineer.py'
t = open(f, encoding='utf-8').read()

# 1. إضافة الاستيراد
if 'from volume_profile import VolumeProfile' not in t:
    t = t.replace('import os', 'import os\nfrom volume_profile import VolumeProfile')

# 2. إضافة استدعاء الـ Volume Profile داخل build_features
old_atr_line = '        data["atr_14"] = (data["high"] - data["low"]).rolling(14).mean()'
new_atr_line = '''        data["atr_14"] = (data["high"] - data["low"]).rolling(14).mean()
        
        # ==========================================
        # VOLUME PROFILE FEATURES (New)
        # ==========================================
        vp = VolumeProfile(lookback=24, num_bins=20)
        data = vp.calculate_features(data)'''

if old_atr_line in t:
    t = t.replace(old_atr_line, new_atr_line)
    open(f, 'w', encoding='utf-8').write(t)
    print("SUCCESS: Integrated Volume Profile into Feature Engineer V3")
else:
    print("ERROR: ATR line not found")

import ast
ast.parse(t)
print("AST OK")
