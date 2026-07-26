from datetime import datetime
from session_risk import SessionRiskMonitor, SessionRiskReport

# Test 1: London-NY overlap + EUR + TF
m = SessionRiskMonitor("trend_following")
r = m.check(datetime(2025,1,15,13,0), currency="EUR")
print("=== Test 1: London-NY EUR TF ===")
print(f"  sessions={r.active_sessions} ov={r.overlaps}")
print(f"  quality={r.quality_score:.2f} avoid={r.should_avoid}")
print(f"  scores={r.session_scores}")
assert r.quality_score > 0.7
assert not r.should_avoid
print("  PASS")

# Test 2: Tokyo only + TF
r = m.check(datetime(2025,1,15,6,30))
print("\n=== Test 2: Tokyo TF ===")
print(f"  quality={r.quality_score:.2f} avoid={r.should_avoid}")
assert r.quality_score < 0.5
print("  PASS")

# Test 3: Weekend
r = m.check(datetime(2025,1,18,13,0))
print("\n=== Test 3: Saturday ===")
print(f"  avoid={r.should_avoid} reason={r.reason}")
assert r.should_avoid and "Weekend" in r.reason
print("  PASS")

# Test 4: Friday close
r = m.check(datetime(2025,1,17,21,0))
print("\n=== Test 4: Friday 21:00 ===")
print(f"  avoid={r.should_avoid} reason={r.reason}")
assert r.should_avoid
print("  PASS")

# Test 5: Friday afternoon
r = m.check(datetime(2025,1,17,17,0))
print("\n=== Test 5: Friday 17:00 ===")
print(f"  quality={r.quality_score:.3f} scores={r.session_scores}")
assert r.quality_score < 0.3
print("  PASS")

# Test 6: AUD + Sydney + MR
m2 = SessionRiskMonitor("mean_reversion")
r = m2.check(datetime(2025,1,15,22,0), currency="AUD")
print("\n=== Test 6: AUD Sydney MR ===")
print(f"  sessions={r.active_sessions} quality={r.quality_score:.2f}")
assert "SYDNEY" in r.active_sessions
assert r.quality_score > 0.4
print("  PASS")

# Test 7: London edge (07:10)
r = m.check(datetime(2025,1,15,7,10), currency="EUR")
print("\n=== Test 7: London edge ===")
print(f"  quality={r.quality_score:.3f} scores={r.session_scores}")
london_sc = r.session_scores.get("LONDON", 1.0)
assert london_sc < 0.8
print("  PASS")

# Test 8: JPY + Tokyo + TF
r = m.check(datetime(2025,1,15,2,0), currency="JPY")
print("\n=== Test 8: JPY Tokyo TF ===")
print(f"  quality={r.quality_score:.2f}")
assert "TOKYO" in r.active_sessions
print("  PASS")

# Test 9: Monday discount
r = m.check(datetime(2025,1,13,13,0), currency="EUR")
r2 = m.check(datetime(2025,1,14,13,0), currency="EUR")
print("\n=== Test 9: Mon vs Tue ===")
print(f"  Mon={r.quality_score:.3f} Tue={r2.quality_score:.3f}")
assert r.quality_score < r2.quality_score
print("  PASS")

print("\n=== ALL 9 TESTS PASSED ===")
