# ============================================================
# الذرة 3.6: PyWavelets – نسخة مصححة
# ============================================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("📥 تحميل بيانات الساعة...")
df = pd.read_csv("data/eurusd_hourly_dukascopy.csv", index_col=0, parse_dates=True)
print(f"✅ {len(df)} شمعة ساعة")

try:
    import pywt
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "PyWavelets"], capture_output=True)
    import pywt

# نسخ البيانات إلى مصفوفة قابلة للكتابة
close = df['close'].values.copy()
wavelet = 'db8'

# تحليل المويجات على كامل السلسلة
print("🧹 تطبيق Wavelet (db8) لتصفية الضوضاء...")
coeffs = pywt.wavedec(close, wavelet, level=5)
coeffs[1] = np.zeros_like(coeffs[1])
coeffs[2] = np.zeros_like(coeffs[2])
denoised = pywt.waverec(coeffs, wavelet)
df['close_wavelet'] = denoised[:len(df)]

# استخراج ميزات Wavelet على نوافذ متحركة
print("📊 استخراج ميزات Wavelet...")
window = 64
wavelet_features = []
for i in range(window, len(df)):
    segment = close[i-window:i].copy()
    c = pywt.wavedec(segment, wavelet, level=3)
    energy = [np.sum(np.abs(x)**2) / len(x) if len(x) > 0 else 0 for x in c]
    total = sum(energy)
    if total > 0:
        energy = [e/total for e in energy]
    wavelet_features.append({
        'w_energy_0': energy[0] if len(energy)>0 else 0,
        'w_energy_1': energy[1] if len(energy)>1 else 0,
        'w_energy_2': energy[2] if len(energy)>2 else 0,
        'w_detail_ratio': sum(energy[1:])/energy[0] if energy[0]>0 else 0,
    })

features_df = pd.DataFrame(wavelet_features, index=df.index[window:])
df = df.merge(features_df, left_index=True, right_index=True, how='left')
df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
df['ATR_ratio'] = df['ATR'] / df['close']
df['SMA_20'] = df['close'].rolling(20).mean()
df['Return'] = df['close'].pct_change()
df['Volatility'] = df['Return'].rolling(24).std()
df.dropna(inplace=True)

Path("data").mkdir(exist_ok=True)
df.to_csv("data/eurusd_wavelet_features.csv")
print(f"✅ تم حفظ {len(df)} صف بميزات Wavelet")
print(f"🎉 الذرة 3.6 اكتملت.")
