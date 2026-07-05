import time
import threading
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from datetime import datetime, timedelta
from pathlib import Path
import csv
import os

# ============================================================
# 1. قائمة الأحداث الاقتصادية عالية التأثير (EUR/USD)
# التوقيت بتوقيت UTC. سنحول إلى توقيت باريس (UTC+1 صيفاً)
# ============================================================
HIGH_IMPACT_NEWS = [
    # (الشهر, اليوم, الساعة (UTC), الاسم)
    # NFP – أول جمعة من كل شهر
    (1, 10, 13, 30, "NFP"), (2, 7, 13, 30, "NFP"), (3, 7, 13, 30, "NFP"),
    (4, 4, 12, 30, "NFP"), (5, 2, 12, 30, "NFP"), (6, 6, 12, 30, "NFP"),
    (7, 3, 12, 30, "NFP"), (8, 1, 12, 30, "NFP"), (9, 5, 12, 30, "NFP"),
    (10, 3, 12, 30, "NFP"), (11, 7, 13, 30, "NFP"), (12, 5, 13, 30, "NFP"),
    # CPI – منتصف الشهر تقريباً
    (1, 15, 13, 30, "CPI"), (2, 14, 13, 30, "CPI"), (3, 14, 12, 30, "CPI"),
    (4, 15, 12, 30, "CPI"), (5, 14, 12, 30, "CPI"), (6, 13, 12, 30, "CPI"),
    (7, 15, 12, 30, "CPI"), (8, 14, 12, 30, "CPI"), (9, 15, 12, 30, "CPI"),
    (10, 15, 12, 30, "CPI"), (11, 14, 13, 30, "CPI"), (12, 15, 13, 30, "CPI"),
    # FOMC – 8 مرات سنوياً
    (1, 31, 19, 0, "FOMC"), (3, 20, 18, 0, "FOMC"), (5, 1, 18, 0, "FOMC"),
    (6, 12, 18, 0, "FOMC"), (7, 31, 18, 0, "FOMC"), (9, 18, 18, 0, "FOMC"),
    (11, 7, 19, 0, "FOMC"), (12, 18, 19, 0, "FOMC"),
]

def is_news_danger_zone():
    """التحقق إذا كنا في منطقة خطر (15 دقيقة قبل الخبر حتى 30 دقيقة بعده)"""
    now = datetime.utcnow()  # توقيت UTC
    current_month = now.month
    
    for month, day, hour, minute, name in HIGH_IMPACT_NEWS:
        if month != current_month:
            continue
        news_time = datetime(now.year, month, day, hour, minute)
        start_danger = news_time - timedelta(minutes=15)
        end_danger = news_time + timedelta(minutes=30)
        if start_danger <= now <= end_danger:
            return True, name, news_time
    return False, None, None

class BeastBot(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.nextOrderId = -1
        self.bars = []
        self.data_ready = False
        self.log_file = "logs/beast_log.csv"
        Path("logs").mkdir(exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event", "Regime", "Raw_Signal", "Final_Signal", "Price", "Details"])

    def log(self, event, regime="N/A", raw_signal=0, final_signal=0, price=0.0, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        regime_names = {-1: "DOWN", 0: "FLAT", 1: "UP"}
        regime_str = regime_names.get(regime, str(regime))
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event, regime_str, raw_signal, final_signal, f"{price:.5f}", details])
        print(f"📝 [{timestamp}] {event} | Regime={regime_str} | Signal={final_signal} | {details}")

    def error(self, reqId, errorCode, errorString, *args):
        if errorCode not in [2104, 2106, 2158, 2107]:
            self.log("ERROR", details=f"Code={errorCode}: {errorString}")

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.connected = True
        self.log("CONNECTION", details="Connected to IBKR")

    def historicalData(self, reqId, bar):
        self.bars.append({
            'Date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close
        })

    def historicalDataEnd(self, reqId, start, end):
        self.data_ready = True
        self.log("DATA_LOADED", details=f"Loaded {len(self.bars)} bars")

    def get_signal(self):
        df = pd.DataFrame(self.bars)
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)

        df["Return"] = df["close"].pct_change()
        df["Range"] = (df["high"] - df["low"]) / df["close"]
        df["Volatility"] = df["Return"].rolling(20).std()
        df["Momentum"] = df["close"].pct_change(10)
        df.dropna(inplace=True)

        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42, n_init=5)
        df["Regime"] = gmm.fit_predict(df[["Return", "Range", "Volatility", "Momentum"]])
        state_returns = df.groupby("Regime")["Return"].mean().sort_values()
        df["Regime"] = df["Regime"].map({state_returns.index[0]: -1, state_returns.index[1]: 0, state_returns.index[2]: 1})

        df["SMA_20"] = df["close"].rolling(20).mean()
        df["SMA_50"] = df["close"].rolling(50).mean()
        df["Signal"] = 0
        df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["SMA_20"].shift(1) <= df["SMA_50"].shift(1)), "Signal"] = 1
        df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["SMA_20"].shift(1) >= df["SMA_50"].shift(1)), "Signal"] = -1

        df["Final"] = 0
        df.loc[(df["Signal"] == 1) & (df["Regime"].isin([0, 1])), "Final"] = 1
        df.loc[(df["Signal"] == -1) & (df["Regime"].isin([-1, 0])), "Final"] = -1

        regime = df["Regime"].iloc[-1]
        raw_sig = df["Signal"].iloc[-1]
        final_sig = df["Final"].iloc[-1]
        price = df["close"].iloc[-1]
        atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 0.005

        reason = ""
        if final_sig == 0:
            if raw_sig == 0:
                reason = "No SMA crossover signal"
            elif regime == 0:
                reason = "Regime is FLAT (neutral) - no trade in flat market"
            elif raw_sig == 1 and regime == -1:
                reason = "BUY signal rejected: Regime is DOWN"
            elif raw_sig == -1 and regime == 1:
                reason = "SELL signal rejected: Regime is UP"
            else:
                reason = "Signal filtered by HMM"
        elif final_sig == 1:
            reason = "BUY signal accepted: Regime supports LONG"
        elif final_sig == -1:
            reason = "SELL signal accepted: Regime supports SHORT"

        self.log("SIGNAL_ANALYSIS", regime=regime, raw_signal=raw_sig, final_signal=final_sig, price=price, details=reason)

        return final_sig, atr, price, regime, reason

    def run_cycle(self):
        self.log("CYCLE_START", details="New cycle started")
        
        # 🛑 مرشح الأخبار
        danger, news_name, news_time = is_news_danger_zone()
        if danger:
            self.log("NEWS_FILTER", details=f"BLOCKED - {news_name} at {news_time.strftime('%H:%M UTC')} (danger zone: 15min before / 30min after)")
            time.sleep(2)
            self.log("CYCLE_END", details="Cycle skipped due to news filter")
            return
        
        self.bars = []
        self.data_ready = False

        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"

        self.reqHistoricalData(1, contract, "", "1 Y", "1 day", "MIDPOINT", 1, 1, False, [])

        timeout = 0
        while not self.data_ready and timeout < 30:
            time.sleep(1)
            timeout += 1

        if not self.data_ready:
            self.log("TIMEOUT", details="Data loading timeout")
            return

        signal, atr, price, regime, reason = self.get_signal()

        if signal != 0:
            risk_percent = 0.01
            stop_distance = 1.5 * atr if atr > 0.001 else 0.005
            quantity = int((10000 * risk_percent) / stop_distance)
            quantity = max(1000, min(quantity, 50000))
            action = "BUY" if signal == 1 else "SELL"

            self.log("TRADE", regime=regime, final_signal=signal, price=price, details=f"{action} {quantity} units")

            order = Order()
            order.action = action
            order.orderType = "MKT"
            order.totalQuantity = quantity
            order.transmit = True
            self.placeOrder(self.nextOrderId, contract, order)
            self.nextOrderId += 1
            self.log("ORDER_SENT", details=f"{action} {quantity} units at {price:.5f}")
        else:
            self.log("NO_TRADE", regime=regime, details=reason)

        time.sleep(2)
        self.log("CYCLE_END", details="Cycle complete")

beast = BeastBot()
beast.connect("127.0.0.1", 4002, 1)
thread = threading.Thread(target=beast.run, daemon=True)
thread.start()

beast.log("BEAST_STARTED", details="Beast is ALIVE. Running 24/7 with full logging + news filter")

while True:
    try:
        beast.run_cycle()
        beast.log("SLEEP", details="Sleeping for 1 hour")
        time.sleep(3600)
    except KeyboardInterrupt:
        beast.log("BEAST_STOPPED", details="Manual stop")
        beast.disconnect()
        break
    except Exception as e:
        beast.log("CRASH", details=str(e))
        time.sleep(300)
