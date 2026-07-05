# ============================================================
# ???????: ?????? ???????? ??????? ?????? ??????
# ????? 2.2+2.3+2.4: ?????????? HMM + ????? + ??? ????????
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import joblib

# 1. ????? ??????? ?????????
df = pd.read_csv("data/eurusd_daily_with_regimes.csv", index_col="Date", parse_dates=True)
gmm = joblib.load("models/hmm_regime_model.pkl")
print(f"? ?? ????? {len(df)} ???? ?? ???????")

# 2. ????????
df["SMA_20"] = df["close"].rolling(20).mean()
df["SMA_50"] = df["close"].rolling(50).mean()
df["ATR"] = (df["high"] - df["low"]).rolling(14).mean()
df = df.dropna()

# 3. ?????? ??????? (????? SMA)
df["Raw_Signal"] = 0
df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1)), "Raw_Signal"] = 1
df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["SMA_20"].shift(1) >= df["SMA_50"].shift(1)), "Raw_Signal"] = -1

# 4. ????? HMM (???? ?????)
df["Signal"] = 0
df.loc[(df["Raw_Signal"] == 1) & (df["Regime"].isin([0, 1])), "Signal"] = 1
df.loc[(df["Raw_Signal"] == -1) & (df["Regime"].isin([-1, 0])), "Signal"] = -1

raw_count = (df["Raw_Signal"] != 0).sum()
filtered_count = (df["Signal"] != 0).sum()
print(f"?? ????? HMM: {raw_count} -> {filtered_count} ?????")

# 5. ?????? ?? ????? ????? + ??? ?????
INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE = 0.01
STOP_ATR_MULT = 2.0
TAKE_ATR_MULT = 3.0
TRAIL_ATR_MULT = 1.0

capital = INITIAL_CAPITAL
equity = []
trades = []
position = 0
entry_price = 0.0
entry_date = None
stop_loss = 0.0
take_profit = 0.0
highest_price = 0.0
lowest_price = 0.0
units = 0.0

for i in range(len(df)):
    date = df.index[i]
    price = df["close"].iloc[i]
    atr = df["ATR"].iloc[i]
    signal = df["Signal"].iloc[i]

    # ??? ????? ?????? ????????
    if position == 1:
        highest_price = max(highest_price, price)
        trail_stop = highest_price - (TRAIL_ATR_MULT * atr)
        stop_loss = max(stop_loss, trail_stop)
    elif position == -1:
        lowest_price = min(lowest_price, price)
        trail_stop = lowest_price + (TRAIL_ATR_MULT * atr)
        stop_loss = min(stop_loss, trail_stop)

    # ??????
    if position != 0:
        hit_sl = (position == 1 and price <= stop_loss) or (position == -1 and price >= stop_loss)
        hit_tp = (position == 1 and price >= take_profit) or (position == -1 and price <= take_profit)
        reverse = (position == 1 and signal == -1) or (position == -1 and signal == 1) or signal == 0

        if hit_sl or hit_tp or reverse:
            pnl = (price - entry_price) * position * units
            cost = (units / 100_000) * 7
            net_pnl = pnl - cost
            capital += net_pnl
            reason = "Trail SL" if hit_sl and not (hit_tp or reverse) else ("TP" if hit_tp else "Rev")
            trades.append({"entry": entry_date, "exit": date, "pnl": net_pnl, 
                          "type": "Long" if position == 1 else "Short", "reason": reason})
            position = 0
            continue

    # ??????
    if position == 0 and signal != 0:
        risk_amount = capital * RISK_PER_TRADE
        stop_distance = STOP_ATR_MULT * atr
        if stop_distance > 0.0001:
            units = min(risk_amount / stop_distance, 50000)
            if signal == 1:
                stop_loss = price - stop_distance
                take_profit = price + (TAKE_ATR_MULT * atr)
                highest_price = price
            else:
                stop_loss = price + stop_distance
                take_profit = price - (TAKE_ATR_MULT * atr)
                lowest_price = price
            entry_price = price
            entry_date = date
            position = signal

    equity.append({"Date": date, "Equity": capital})

# 6. ???????
eq_df = pd.DataFrame(equity).set_index("Date")
total_return = (capital / INITIAL_CAPITAL - 1) * 100
n = len(trades)

if n > 0:
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / n * 100
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
    pf = abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else (np.inf if wins else 0)
    max_dd = (eq_df["Equity"] / eq_df["Equity"].cummax() - 1).min() * 100
    sharpe = (eq_df["Equity"].pct_change().mean() / eq_df["Equity"].pct_change().std() * np.sqrt(252)) if eq_df["Equity"].pct_change().std() != 0 else 0

    print(f"\n?? ????? ??????? ?????? (HMM + ??? ?????):")
    print(f"?? ??? ?????: {INITIAL_CAPITAL:,.0f} -> {capital:,.0f} $")
    print(f"?? ??????: {total_return:.2f}% | ?? ???? ??????: {max_dd:.2f}%")
    print(f"?? Sharpe: {sharpe:.2f}")
    print(f"?? ???????: {n} | ? ???: {win_rate:.1f}%")
    print(f"?? ????? ?????: {avg_win:,.2f} | ?? ????? ???????: {avg_loss:,.2f}")
    print(f"?? Profit Factor: {pf:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    axes[0].plot(eq_df.index, eq_df["Equity"], color="green", linewidth=2)
    axes[0].axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--")
    axes[0].set_title("????? ?????? ? ??????? ?????? (HMM)", fontsize=14)
    dd = (eq_df["Equity"] / eq_df["Equity"].cummax() - 1) * 100
    axes[1].fill_between(eq_df.index, 0, dd, color="red", alpha=0.4)
    axes[1].plot(eq_df.index, dd, color="red", linewidth=1)
    axes[1].set_title(f"Drawdown (??????={max_dd:.2f}%)", fontsize=14)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    fig.savefig("reports/2.4_hmm_equity.png", dpi=150)
    plt.close()
    print("? ??????? ????? ?? reports/2.4_hmm_equity.png")
else:
    print("?? ?? ??? ????? ?? ?????.")

print("?? ??????? ?????? ?????.")
