# ============================================================
# ???????: ?????? ???????? ??????? ?????? ??????
# ????? 1.3+1.4+1.5: ?????????? SMA + Backtester + ????? ?????
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 1. ????? ????????
df = pd.read_csv("data/eurusd_daily.csv", index_col="Date", parse_dates=True)
df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
df = df.dropna()

# 2. ????? ???????? (????? SMA ?? ???? ATR)
df["Signal"] = 0
df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1)), "Signal"] = 1
df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["SMA_20"].shift(1) >= df["SMA_50"].shift(1)), "Signal"] = -1

# 3. ????? ??????? ?????????
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01
STOP_ATR_MULT = 2.0
TAKE_ATR_MULT = 3.0

capital = INITIAL_CAPITAL
equity = []
trades = []
position = 0
entry_price = 0.0
entry_date = None
stop_loss = 0.0
take_profit = 0.0
units = 0.0

for i in range(len(df)):
    date = df.index[i]
    price = df["close"].iloc[i]
    atr = df["ATR"].iloc[i]
    signal = df["Signal"].iloc[i]

    if position != 0:
        hit_sl = (position == 1 and price <= stop_loss) or (position == -1 and price >= stop_loss)
        hit_tp = (position == 1 and price >= take_profit) or (position == -1 and price <= take_profit)
        reverse = (position == 1 and signal == -1) or (position == -1 and signal == 1) or signal == 0
        
        if hit_sl or hit_tp or reverse:
            pnl = (price - entry_price) * position * units
            cost = (units / 100_000) * 7
            net_pnl = pnl - cost
            capital += net_pnl
            trades.append({"entry": entry_date, "exit": date, "pnl": net_pnl, "type": "Long" if position == 1 else "Short",
                          "reason": "SL" if hit_sl else ("TP" if hit_tp else "Rev")})
            position = 0
            continue

    if position == 0 and signal != 0:
        risk_amount = capital * RISK_PER_TRADE
        stop_distance = STOP_ATR_MULT * atr
        if stop_distance > 0.0001:
            units = min(risk_amount / stop_distance, 50000)
            if signal == 1:
                stop_loss = price - stop_distance
                take_profit = price + (TAKE_ATR_MULT * atr)
            else:
                stop_loss = price + stop_distance
                take_profit = price - (TAKE_ATR_MULT * atr)
            entry_price = price
            entry_date = date
            position = signal

    equity.append({"Date": date, "Equity": capital})

# 4. ???????
eq_df = pd.DataFrame(equity).set_index("Date")
total_return = (capital / INITIAL_CAPITAL - 1) * 100
n = len(trades)

if n > 0:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / n * 100
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    profit_factor = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else (np.inf if wins else 0)
    max_dd = (eq_df["Equity"] / eq_df["Equity"].cummax() - 1).min() * 100

    print(f"\n?? ????? ??????? ?????:")
    print(f"?? ??? ?????: {INITIAL_CAPITAL:,.0f} -> {capital:,.0f} $")
    print(f"?? ??????: {total_return:.2f}% | ?? ???? ??????: {max_dd:.2f}%")
    print(f"?? ???????: {n} | ? ???: {win_rate:.1f}%")
    print(f"?? ????? ?????: {avg_win:,.2f} | ?? ????? ???????: {avg_loss:,.2f}")
    print(f"?? Profit Factor: {profit_factor:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    axes[0].plot(eq_df.index, eq_df["Equity"], color="green", linewidth=2)
    axes[0].axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--")
    axes[0].set_title("????? ??????", fontsize=14)
    dd = (eq_df["Equity"] / eq_df["Equity"].cummax() - 1) * 100
    axes[1].fill_between(eq_df.index, 0, dd, color="red", alpha=0.4)
    axes[1].plot(eq_df.index, dd, color="red", linewidth=1)
    axes[1].set_title(f"Drawdown (??????={max_dd:.2f}%)", fontsize=14)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    Path("reports").mkdir(exist_ok=True)
    fig.savefig("reports/1.5_equity_curve.png", dpi=150)
    plt.close()
    print("? ??????? ????? ?? reports/1.5_equity_curve.png")
else:
    print("?? ?? ??? ????? ?? ?????.")
