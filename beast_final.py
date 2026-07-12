# Beast V-Final Strategy
# Main strategy module - multi-indicator entry/exit logic with risk management

# ============================================================
# المشروع: الكيان الفيروسي – Beast Final (حدود ديناميكية)
# ============================================================
import time
import threading
import csv
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

LIVE_TRADING_CONFIRMATION_ENV = 'FOREX_BOT_ALLOW_LIVE_TRADING'
LIVE_TRADING_CONFIRMATION_VALUE = 'I_ACCEPT_LIVE_TRADING_RISK'

HIGH_IMPACT_NEWS = [
    (1, 10, 13, 30, "NFP"), (2, 7, 13, 30, "NFP"), (3, 7, 13, 30, "NFP"),
    (4, 4, 12, 30, "NFP"), (5, 2, 12, 30, "NFP"), (6, 6, 12, 30, "NFP"),
    (7, 3, 12, 30, "NFP"), (8, 1, 12, 30, "NFP"), (9, 5, 12, 30, "NFP"),
    (10, 3, 12, 30, "NFP"), (11, 7, 13, 30, "NFP"), (12, 5, 13, 30, "NFP"),
    (1, 15, 13, 30, "CPI"), (2, 14, 13, 30, "CPI"), (3, 14, 12, 30, "CPI"),
    (4, 15, 12, 30, "CPI"), (5, 14, 12, 30, "CPI"), (6, 13, 12, 30, "CPI"),
    (7, 15, 12, 30, "CPI"), (8, 14, 12, 30, "CPI"), (9, 15, 12, 30, "CPI"),
    (10, 15, 12, 30, "CPI"), (11, 14, 13, 30, "CPI"), (12, 15, 13, 30, "CPI"),
    (1, 31, 19, 0, "FOMC"), (3, 20, 18, 0, "FOMC"), (5, 1, 18, 0, "FOMC"),
    (6, 12, 18, 0, "FOMC"), (7, 31, 18, 0, "FOMC"), (9, 18, 18, 0, "FOMC"),
    (11, 7, 19, 0, "FOMC"), (12, 18, 19, 0, "FOMC"),
]

def is_news_danger_zone():
    now = datetime.utcnow()
    current_month = now.month
    for month, day, hour, minute, name in HIGH_IMPACT_NEWS:
        if month != current_month:
            continue
        news_time = datetime(now.year, month, day, hour, minute)
        if news_time - timedelta(minutes=15) <= now <= news_time + timedelta(minutes=30):
            return True, name
    return False, None

class BeastFinal(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected = False
        self.nextOrderId = -1
        self.bars_h1 = []
        self.bars_m15 = []
        self.data_ready_h1 = False
        self.data_ready_m15 = False
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.daily_trades = 0
        self.max_daily_loss = -300
        self.max_weekly_loss = -500
        self.last_trade_date = None
        self.current_position = 0
        self.position_ready = False
        self.account_balance = 10000.0
        self.balance_ready = False
        # حدود ديناميكية من الوسيط
        self.max_order_size = 20000  # قيمة افتراضية لحين جلبها من الوسيط
        self.contract_details_ready = False
        self.log_file = "logs/beast_log.csv"
        Path("logs").mkdir(exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["Timestamp", "Strategy", "Event", "Details"])

    def log(self, strategy, event, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([timestamp, strategy, event, str(details)])
        print(f"[{timestamp}] {strategy} | {event} | {details}")

    def error(self, reqId, errorCode, errorString, *args):
        IGNORE = {2104, 2106, 2158, 2103, 2105, 2107}
        if errorCode in IGNORE:
            return
        if errorCode in {502, 504, 1100, 1300}:
            self.log("SYSTEM", f"CONN_ERROR", f"Code={errorCode}")
            self.connected = False
        else:
            self.log("SYSTEM", f"ERR_{reqId}", f"Code={errorCode}: {str(errorString)[:80]}")

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.connected = True
        self.log("SYSTEM", "CONNECTED", f"orderId={orderId}")

    # جلب تفاصيل العقد (حدود الحجم)
    def contractDetails(self, reqId, contractDetails):
        if contractDetails.contract.symbol == "EUR" and contractDetails.contract.currency == "USD":
            if contractDetails.minSize:
                self.log("SYSTEM", "CONTRACT", f"MinSize={contractDetails.minSize}")
                self.max_order_size = 20000
            self.contract_details_ready = True

    def contractDetailsEnd(self, reqId):
        if not self.contract_details_ready:
            self.log("SYSTEM", "CONTRACT", "Using default max order size 20000")
            self.contract_details_ready = True

    def accountSummary(self, reqId, account, tag, value, currency):
        if tag == "NetLiquidation":
            self.account_balance = float(value)
            self.balance_ready = True

    def accountSummaryEnd(self, reqId):
        self.log("SYSTEM", "BALANCE", f"${self.account_balance:,.2f}")

    def position(self, account, contract, pos, avgCost):
        if contract.symbol == "EUR" and contract.currency == "USD":
            self.current_position = pos
            self.position_ready = True

    def positionEnd(self):
        self.log("SYSTEM", "POSITION", f"Current={self.current_position}")

    def historicalData(self, reqId, bar):
        if reqId == 2:
            self.bars_h1.append({'Date': bar.date, 'open': bar.open, 'high': bar.high, 'low': bar.low, 'close': bar.close})
        elif reqId == 3:
            self.bars_m15.append({'Date': bar.date, 'open': bar.open, 'high': bar.high, 'low': bar.low, 'close': bar.close})

    def historicalDataEnd(self, reqId, start, end):
        if reqId == 2:
            self.data_ready_h1 = True
            self.log("MATRIX-H1", "DATA_LOADED", f"{len(self.bars_h1)} bars")
        elif reqId == 3:
            self.data_ready_m15 = True
            self.log("MM-M15", "DATA_LOADED", f"{len(self.bars_m15)} bars")

    def clean_date(self, date_str):
        return re.sub(r'\s+US/\w+', '', str(date_str))

    def get_matrix_signal(self):
        df = pd.DataFrame(self.bars_h1).set_index('Date')
        df.index = pd.to_datetime(df.index.map(self.clean_date), errors='coerce')
        df.dropna(inplace=True)
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
        df['ATR_ratio'] = df['ATR'] / df['close']
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['Trend'] = 0
        df.loc[df['close'] > df['SMA_20'] * 1.001, 'Trend'] = 1
        df.loc[df['close'] < df['SMA_20'] * 0.999, 'Trend'] = -1
        vol = df['ATR_ratio'].dropna()
        low_thresh = vol.quantile(0.33)
        high_thresh = vol.quantile(0.66)
        df['VolClass'] = 1
        df.loc[df['ATR_ratio'] <= low_thresh, 'VolClass'] = 0
        df.loc[df['ATR_ratio'] >= high_thresh, 'VolClass'] = 2
        df['Regime9'] = df['VolClass'] * 3 + (df['Trend'] + 1)
        regime_rules = {0: -1, 1: 0, 2: 1, 3: -1, 4: 0, 5: 1, 6: -1, 7: 0, 8: 1}
        df['Allowed'] = df['Regime9'].map(regime_rules)
        df.dropna(inplace=True)
        df['SMA_S'] = df['close'].rolling(10).mean()
        df['SMA_L'] = df['close'].rolling(30).mean()
        df['BB_mid'] = df['close'].rolling(20).mean()
        df['BB_std'] = df['close'].rolling(20).std()
        df['BB_lower'] = df['BB_mid'] - 1.5 * df['BB_std']
        df['BB_upper'] = df['BB_mid'] + 1.5 * df['BB_std']
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        df['RSI'] = 100 - (100 / (1 + gain.rolling(14).mean() / loss.rolling(14).mean()))
        df.dropna(inplace=True)
        buy_sma = ((df['SMA_S'] > df['SMA_L']) & (df['SMA_S'].shift(1) <= df['SMA_L'].shift(1)))
        sell_sma = ((df['SMA_S'] < df['SMA_L']) & (df['SMA_S'].shift(1) >= df['SMA_L'].shift(1)))
        buy_bb = (df['low'] <= df['BB_lower'])
        sell_bb = (df['high'] >= df['BB_upper'])
        buy_rsi = (df['RSI'] < 48)
        sell_rsi = (df['RSI'] > 52)
        buy = (buy_sma | buy_bb) & buy_rsi
        sell = (sell_sma | sell_bb) & sell_rsi
        final = pd.Series(0, index=df.index)
        final[buy & df['Allowed'].isin([1, 0])] = 1
        final[sell & df['Allowed'].isin([-1, 0])] = -1
        return final.iloc[-1], df['close'].iloc[-1], df['ATR'].iloc[-1]

    def get_mm_signal(self):
        if len(self.bars_m15) < 30:
            return False, 0, 0
        df = pd.DataFrame(self.bars_m15).set_index('Date')
        df.index = pd.to_datetime(df.index.map(self.clean_date), errors='coerce')
        df.dropna(inplace=True)
        df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
        df['ATR_ratio'] = df['ATR'] / df['close']
        df['SMA_20'] = df['close'].rolling(20).mean()
        df['Trend'] = 0
        df.loc[df['close'] > df['SMA_20'] * 1.001, 'Trend'] = 1
        df.loc[df['close'] < df['SMA_20'] * 0.999, 'Trend'] = -1
        vol = df['ATR_ratio'].dropna()
        if len(vol) < 10:
            return False, 0, 0
        low_thresh = vol.quantile(0.33)
        last_vol = df['ATR_ratio'].iloc[-1]
        last_trend = df['Trend'].iloc[-1]
        last_price = df['close'].iloc[-1]
        last_atr = df['ATR'].iloc[-1]
        active = (last_trend == 0) and (last_vol <= low_thresh) and (last_atr > 0.0001)
        return active, last_price, last_atr

    # حجم الصفقة الديناميكي مع حد أقصى من الوسيط
    def calculate_position_size(self, atr_value):
        risk_amount = self.account_balance * 0.01
        stop_distance = 2.0 * atr_value
        if stop_distance < 0.0001:
            stop_distance = 0.0001
        units = risk_amount / stop_distance
        units = int(min(units, self.max_order_size))
        units = max(units, 1000)
        return units

    def place_trade(self, strategy, action, quantity):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        order = Order()
        order.action = action
        order.orderType = "MKT"
        order.totalQuantity = quantity
        order.transmit = True
        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1
        self.log(strategy, "TRADE", f"{action} {quantity} units (Max={self.max_order_size})")

    def place_limit_order(self, strategy, action, quantity, limit_price):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        order = Order()
        order.action = action
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = round(limit_price, 5)
        order.transmit = True
        self.placeOrder(self.nextOrderId, contract, order)
        self.nextOrderId += 1
        self.log(strategy, "LIMIT_ORDER", f"{action} {quantity}u @ {limit_price:.5f}")

    def kill_switch_check(self):
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.last_trade_date = today
        if self.daily_pnl <= self.max_daily_loss:
            self.log("KILL_SWITCH", "DAILY_LIMIT", f"PnL={self.daily_pnl:.2f}")
            return True
        if self.weekly_pnl <= self.max_weekly_loss:
            self.log("KILL_SWITCH", "WEEKLY_LIMIT", f"PnL={self.weekly_pnl:.2f}")
            return True
        return False

    def run_cycle(self):
        danger, news_name = is_news_danger_zone()
        if danger:
            self.log("SYSTEM", "NEWS_BLOCKED", news_name)
            return
        if self.kill_switch_check():
            self.log("SYSTEM", "KILLED", "Trading halted")
            return

        # جلب تفاصيل العقد (أول مرة فقط، ثم تُخزّن)
        if not self.contract_details_ready:
            contract = Contract()
            contract.symbol = "EUR"
            contract.secType = "CASH"
            contract.currency = "USD"
            contract.exchange = "IDEALPRO"
            self.reqContractDetails(99, contract)
            timeout_cd = 0
            while not self.contract_details_ready and timeout_cd < 5:
                time.sleep(1)
                timeout_cd += 1

        # جلب رصيد الحساب والمراكز
        self.balance_ready = False
        self.reqAccountSummary(1, "All", "NetLiquidation")
        self.position_ready = False
        self.reqPositions()
        timeout_meta = 0
        while (not self.balance_ready or not self.position_ready) and timeout_meta < 5:
            time.sleep(1)
            timeout_meta += 1

        self.bars_h1 = []
        self.bars_m15 = []
        self.data_ready_h1 = False
        self.data_ready_m15 = False
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.currency = "USD"
        contract.exchange = "IDEALPRO"
        self.reqHistoricalData(2, contract, "", "6 M", "1 hour", "MIDPOINT", 1, 1, False, [])
        self.reqHistoricalData(3, contract, "", "1 W", "15 mins", "MIDPOINT", 1, 1, False, [])
        timeout = 0
        while (not self.data_ready_h1 or not self.data_ready_m15) and timeout < 60:
            time.sleep(1)
            timeout += 1
        if not self.data_ready_h1 or not self.data_ready_m15:
            self.log("SYSTEM", "TIMEOUT", f"H1={self.data_ready_h1} M15={self.data_ready_m15}")
            return

        matrix_signal, matrix_price, matrix_atr = self.get_matrix_signal()
        mm_active, mm_price, mm_atr = self.get_mm_signal()

        if matrix_signal != 0:
            if self.current_position == 0:
                quantity = self.calculate_position_size(matrix_atr)
                self.place_trade("MATRIX-H1", "BUY" if matrix_signal == 1 else "SELL", quantity)
                self.log("MATRIX-H1", "SIGNAL", f"{'BUY' if matrix_signal==1 else 'SELL'} {quantity}u @ {matrix_price:.5f}")
            elif matrix_signal == self.current_position:
                self.log("MATRIX-H1", "SKIP", f"Already have position in this direction")
            else:
                self.log("MATRIX-H1", "REVERSE", "Closing first")
                close_action = "SELL" if self.current_position == 1 else "BUY"
                close_qty = abs(self.current_position)
                self.place_trade("MATRIX-H1", close_action, close_qty)
                time.sleep(2)
                quantity = self.calculate_position_size(matrix_atr)
                self.place_trade("MATRIX-H1", "BUY" if matrix_signal == 1 else "SELL", quantity)
                self.log("MATRIX-H1", "SIGNAL", f"{'BUY' if matrix_signal==1 else 'SELL'} {quantity}u @ {matrix_price:.5f}")
        elif mm_active and self.current_position == 0:
            buy_limit = mm_price - 0.3 * mm_atr
            sell_limit = mm_price + 0.3 * mm_atr
            mm_qty = int(self.account_balance * 0.002 / mm_atr) if mm_atr > 0 else 500
            mm_qty = max(min(mm_qty, self.max_order_size // 10), 100)  # 10% من الحد الأقصى كحد أقصى لصانع السوق
            self.place_limit_order("MM-M15", "BUY", mm_qty, buy_limit)
            self.place_limit_order("MM-M15", "SELL", mm_qty, sell_limit)
            self.log("MM-M15", "GRID_PLACED", f"Buy@{buy_limit:.5f} Sell@{sell_limit:.5f} x{mm_qty}u")
        else:
            self.log("SYSTEM", "NO_SIGNAL", "Waiting")
        self.log("SYSTEM", "CYCLE_COMPLETE", "Matrix + Market Maker")

    def connect_with_retry(self, max_retries=10):
        for attempt in range(1, max_retries + 1):
            try:
                self.log("SYSTEM", "CONNECTING", f"attempt {attempt}/{max_retries}")
                self.connect("127.0.0.1", 4002, 1)
                t = threading.Thread(target=self.run, daemon=True)
                t.start()
                for _ in range(10):
                    if self.connected:
                        return True
                    time.sleep(1)
            except Exception as e:
                self.log("SYSTEM", "CONN_FAIL", str(e)[:80])
            wait = 30 * attempt
            self.log("SYSTEM", "RETRY_WAIT", f"{wait}s")
            time.sleep(wait)
        return False

def watchdog(beast):
    while True:
        time.sleep(60)
        if not beast.connected:
            beast.log("SYSTEM", "WATCHDOG", "Reconnecting...")
            try:
                beast.disconnect()
            except:
                pass
            time.sleep(5)
            beast.connect_with_retry()

def require_live_trading_confirmation():
    confirmation = os.environ.get(LIVE_TRADING_CONFIRMATION_ENV)
    if confirmation != LIVE_TRADING_CONFIRMATION_VALUE:
        raise SystemExit(
            "LIVE TRADING BLOCKED. Set "
            f"{LIVE_TRADING_CONFIRMATION_ENV}={LIVE_TRADING_CONFIRMATION_VALUE} "
            "only after explicit founder approval."
        )


def main():
    require_live_trading_confirmation()
    print("BEAST FINAL – Immune System Active")
    print("=" * 50)
    beast = BeastFinal()
    ok = beast.connect_with_retry()
    if not ok:
        print("Failed to connect. Ensure IB Gateway is open on port 4002.")
        exit(1)
    beast.log("SYSTEM", "BEAST_ONLINE", "Immune System Active – Watchdog Running")
    wd = threading.Thread(target=watchdog, args=(beast,), daemon=True)
    wd.start()
    beast.log("SYSTEM", "WATCHDOG_STARTED", "Monitoring every 60s")
    while True:
        try:
            if beast.connected:
                beast.run_cycle()
                beast.log("SYSTEM", "SLEEP", "15 minutes")
                time.sleep(900)
            else:
                beast.log("SYSTEM", "WAITING", "Watchdog will reconnect")
                time.sleep(60)
        except KeyboardInterrupt:
            beast.log("SYSTEM", "STOPPED", "Manual")
            beast.disconnect()
            break
        except Exception as e:
            beast.log("SYSTEM", "CRASH", str(e)[:100])
            time.sleep(300)


if __name__ == "__main__":
    main()


