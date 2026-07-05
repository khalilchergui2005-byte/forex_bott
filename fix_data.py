# ============================================================
# المشروع: الكيان الفيروسي – إعادة بناء بيانات الساعة
# ============================================================
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
import yfinance as yf

print("Loading hourly data...")
end = pd.Timestamp.now()
start = end - pd.DateOffset(days=730)
data = yf.download("EURUSD=X", start=start, end=end, interval="1h", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
df = data[["Open","High","Low","Close"]].copy()
df.columns = ["open","high","low","close"]
df.dropna(inplace=True)

df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
df["RSI"] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
df["Return"] = df["close"].pct_change()
df["Volatility"] = df["Return"].rolling(24).std()

df_temp = df.dropna(subset=["Return","Volatility"])
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
df.loc[df_temp.index, "Regime"] = gmm.fit_predict(df_temp[["Return","Volatility"]].values)
df["Regime"] = df["Regime"].fillna(0)
df.dropna(inplace=True)

# التأكد من أن الفهرس عمود وليس اسمه Date
df.index.name = "Date"

train_size = int(0.7 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

Path("data").mkdir(exist_ok=True)
train_df.to_csv("data/train_hourly.csv")
test_df.to_csv("data/test_hourly.csv")
print(f"Saved: train={len(train_df)} test={len(test_df)}")
