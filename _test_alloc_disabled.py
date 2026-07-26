import numpy as np
from portfolio_allocator import PortfolioAllocator, AllocationDecision

class R:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

pa = PortfolioAllocator()

# Test 1: Clean approval
d = pa.allocate("S1", "long", 1.0)
print("=== Test 1: Clean ===")
print(f"  approved={d.approved} size={d.final_size:.2f} reason={d.reason}")
assert d.approved and d.final_size == 1.0
print("  PASS")

# Test 2: News blocks
d = pa.allocate("S1","long",1.0,
    news_report=R(should_avoid=True,reason="NFP",recommended_reduction=1.0))
print("\n=== Test 2: News block ===")
print(f"  approved={d.approved} reason={d.reason}")
assert not d.approved and "News" in d.reason
print("  PASS")

# Test 3: Spread blocks
d = pa.allocate("S1","long",1.0,
    spread_report=R(should_avoid=True,reason="spike"))
print("\n=== Test 3: Spread block ===")
assert not d.approved and "Spread" in d.reason
print("  PASS")

# Test 4: News reduces
d = pa.allocate("S1","long",1.0,
    news_report=R(should_avoid=False,recommended_reduction=0.5,reason=""))
print("\n=== Test 4: News 50% reduction ===")
print(f"  size={d.final_size:.2f} factors={d.factors}")
assert d.approved and abs(d.final_size - 0.5) < 0.01
print("  PASS")

# Test 5: Session reduces
d = pa.allocate("S1","long",1.0,
    session_report=R(should_avoid=False,recommended_reduction=0.3,reason=""))
print("\n=== Test 5: Session 30% reduction ===")
assert d.approved and abs(d.final_size - 0.7) < 0.01
print("  PASS")

# Test 6: Exposure scaling
pos = [{"side":"long","size":8},{"side":"long","size":2}]
d = pa.allocate("S1","long",1.0,positions=pos)
print("\n=== Test 6: Exposure ===")
print(f"  size={d.final_size:.3f} factors={d.factors}")
assert d.final_size < 1.0 and "exposure" in d.factors
print("  PASS")

# Test 7: Risk parity
np.random.seed(42)
eq = {"S1": np.cumsum(np.random.randn(100)*0.01)+100,
      "S2": np.cumsum(np.random.randn(100)*0.05)+100}
d = pa.allocate("S1","long",1.0,equity_curves=eq)
print("\n=== Test 7: Risk parity ===")
print(f"  size={d.final_size:.3f} rp={d.factors.get('risk_parity')}")
assert "risk_parity" in d.factors and d.final_size < 1.0
print("  PASS")

# Test 8: Regime volatile
d = pa.allocate("S1","long",1.0,
    regime_report=R(regime_name="VOLATILE"))
print("\n=== Test 8: Volatile regime ===")
assert d.approved and abs(d.final_size - 0.5) < 0.01
print("  PASS")

# Test 9: Multiple factors
d = pa.allocate("S1","long",1.0,
    regime_report=R(regime_name="VOLATILE"),
    session_report=R(should_avoid=False,recommended_reduction=0.2,reason=""))
print("\n=== Test 9: Combined ===")
print(f"  size={d.final_size:.3f} factors={d.factors}")
assert abs(d.final_size - 0.4) < 0.01
print("  PASS")

# Test 10: Bias block
from bias_score import BiasReport
from bias_trading_compat import BiasTradingCompat
btc = BiasTradingCompat()
bias = BiasReport(-0.8, 0.8, "BEARISH", {})
d = pa.allocate("S1","long",1.0,bias_report=bias,trading_type="trend_following")
print("\n=== Test 10: Bias block ===")
print(f"  approved={d.approved} reason={d.reason}")
assert not d.approved and "Bias" in d.reason
print("  PASS")

print("\n=== ALL 10 TESTS PASSED ===")
