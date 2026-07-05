# ============================================================
# ???????: ?????? ???????? ??????? ?????? ??????
# ????? 1.2: ??????? ???????? (EDA)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

df = pd.read_csv("data/eurusd_daily.csv", index_col="Date", parse_dates=True)
print(f"? ?? ????? {len(df)} ????")

df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["daily_return"] = df["close"].pct_change()
df["volatility_20"] = df["daily_return"].rolling(20).std()
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df.dropna(inplace=True)

total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
annual_vol = df["log_return"].std() * np.sqrt(252) * 100
sharpe = (df["log_return"].mean() / df["log_return"].std() * np.sqrt(252)) if df["log_return"].std() != 0 else 0
max_dd = (df["close"] / df["close"].cummax() - 1).min() * 100

print(f"\n?? ????????? ????????:")
print(f"?????? ?????: {total_return:.2f}%")
print(f"?????? ??????: {annual_vol:.2f}%")
print(f"???? Sharpe: {sharpe:.2f}")
print(f"???? ??????: {max_dd:.2f}%")

Path("reports").mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
axes[0].plot(df.index, df["close"], linewidth=1.2, color="#1f77b4")
axes[0].plot(df.index, df["SMA_20"], linewidth=1.0, color="#ff7f0e", label="SMA 20")
axes[0].plot(df.index, df["SMA_50"], linewidth=1.0, color="#2ca02c", label="SMA 50")
axes[0].set_title("EUR/USD - ????? ?????????? ????????", fontsize=14)
axes[0].legend()
axes[1].plot(df.index, df["volatility_20"] * 100, color="#d62728", linewidth=1.2)
axes[1].set_title("?????? ???????? (20 ???)", fontsize=14)
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
plt.tight_layout()
fig.savefig("reports/1.2_eda_report.png", dpi=150)
plt.close()
print("? ?? ??? ??????? ?? reports/1.2_eda_report.png")
