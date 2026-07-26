# ============================================================
# الذرة 5.3: قياس تكلفة التداول الفعلية على IBKR Demo
# ============================================================
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import numpy as np

class CostMeasurer(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.bid = 0.0
        self.ask = 0.0
        self.spreads = []
        self.done = False

    def error(self, reqId, errorCode, errorString, *args):
        if errorCode not in [2104, 2106, 2158]:
            print(f"   [INFO] {errorString}")

    def nextValidId(self, orderId):
        print("✅ Connected (client 2)")

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 1:  # Bid
            self.bid = price
        elif tickType == 2:  # Ask
            self.ask = price
            if self.bid > 0:
                spread = self.ask - self.bid
                self.spreads.append(spread)

    def tickSize(self, reqId, tickType, size):
        pass

print("📡 Measuring real IBKR Demo spread for EUR/USD (10 seconds)...")
measurer = CostMeasurer()
measurer.connect("127.0.0.1", 4002, clientId=2)  # clientId=2 different from beast bot
thread = threading.Thread(target=measurer.run, daemon=True)
thread.start()
time.sleep(1)

contract = Contract()
contract.symbol = "EUR"
contract.secType = "CASH"
contract.currency = "USD"
contract.exchange = "IDEALPRO"
measurer.reqMarketDataType(3)  # delayed data is fine for spread measurement
measurer.reqMktData(1, contract, "", False, False, [])

time.sleep(12)  # Collect data for ~10 seconds
measurer.cancelMktData(1)
measurer.disconnect()

if measurer.spreads:
    avg_spread = np.mean(measurer.spreads)
    avg_spread_pips = avg_spread * 10000
    print(f"📊 Average spread: {avg_spread_pips:.2f} pips ({avg_spread:.6f})")
else:
    avg_spread = 0.0001  # fallback
    avg_spread_pips = 1.0
    print(f"⚠️  No spread data. Using default 1 pip.")

# IBKR commission: 0.20 bps of trade value, min $2 per side
# For 100,000 units trade, value ≈ $100,000 * price
# We'll compute for typical trade sizes
print("\n💰 Commission estimation (IBKR standard):")
print("   Commission = max(trade_value * 0.00002, $2) per side")
print("   Round-turn (entry+exit) = 2 * per-side")
print()
# Example: 50,000 units, price 1.10
example_price = 1.10
example_units = 50000
example_value = example_units * example_price
comm_per_side = max(example_value * 0.00002, 2.0)
roundturn_comm = 2 * comm_per_side
print(f"   Example: {example_units} units @ {example_price:.5f} = ${example_value:,.2f} trade value")
print(f"   Commission per side: ${comm_per_side:.2f}")
print(f"   Round-turn commission: ${roundturn_comm:.2f}")

# Total cost for a trade
total_cost_pct = (avg_spread * example_units + roundturn_comm) / example_value * 100
print(f"\n📉 Total estimated round-trip cost: {total_cost_pct:.3f}% of trade value")
print(f"   (spread: {avg_spread*example_units:.2f} + comm: {roundturn_comm:.2f})")
print("✅ الذرة 5.3 اكتملت.")
