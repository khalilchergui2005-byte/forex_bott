from datetime import datetime, timedelta
"""test_risk_guardian.py - 6-level RiskGuardian tests"""
import pytest
from risk_guardian import  (
    LevelConfig,
    RiskGuardian, ProtectionLevel as PL,
    GuardianDecision, VerdictAction, LevelConfig,
)
from risk_guardian import RiskManager

LEVELS = [PL.TRADE, PL.STRATEGY, PL.TYPE, PL.ACCOUNT, PL.FIRM, PL.SYSTEM]
LKEYS = ["trade","strategy","trading_type","account","firm","system"]

@pytest.fixture
def rg():
    return RiskGuardian(initial_equity=100_000.0)

class TestInit:
    def test_6_levels(self, rg):
        s = rg.get_state()
        assert len(s) == 6, f"Expected 6, got {len(s)}"
        for k in LKEYS:
            assert k in s, f"Missing key: {k}"
    def test_l1_config(self, rg):
        c = rg._configs[PL.TRADE]
        assert c.max_positions == 1
    def test_not_halted(self, rg):
        for k in LKEYS:
            assert rg.get_state()[k]["halted"] is False
    def test_equity(self, rg):
        for k in LKEYS:
            assert rg.get_state()[k]["equity"] == 100_000.0
    def test_all_configs_exist(self, rg):
        for pl in LEVELS:
            c = rg._configs[pl]
            assert isinstance(c, LevelConfig)
            assert c.max_positions > 0
            assert c.max_daily_loss_pct > 0
            assert c.max_drawdown_pct > 0

class TestSignal:
    def test_fresh_allowed(self, rg):
        d = rg.check_signal()
        assert d.approved is True
        assert d.action == VerdictAction.ALLOW
    def test_returns_decision(self, rg):
        d = rg.check_signal()
        assert isinstance(d, GuardianDecision)
        assert len(d.verdicts) == 6
        assert d.size_factor == 1.0

class TestPositions:
    def test_l1_block_at_1(self, rg):
        rg.update_positions(PL.TRADE, 1)
        d = rg.check_signal()
        assert d.approved is False
    def test_l1_allow_at_0(self, rg):
        rg.update_positions(PL.TRADE, 0)
        assert rg.check_signal().approved is True
    def test_hierarchy(self, rg):
        rg.update_positions(PL.TRADE, 0)
        assert rg.check_signal().approved is True
        rg.update_positions(PL.TRADE, 1)
        assert rg.check_signal().approved is False
    def test_reduce_restores(self, rg):
        rg.update_positions(PL.TRADE, 1)
        assert rg.check_signal().approved is False
        rg.update_positions(PL.TRADE, 0)
        assert rg.check_signal().approved is True
class TestDailyPnL:
    def test_loss_blocks(self, rg):
        rg.record_trade(-2000, level=PL.TRADE)
        d = rg.check_signal()
        assert d.approved is False, f"L1 1% daily should block, got {d.action}"
    def test_small_profit_ok(self, rg):
        rg.record_trade(500, level=PL.TRADE)
        assert rg.check_signal().approved is True

class TestDrawdown:
    def test_dd_blocks(self, rg):
        rg.update_equity(PL.TRADE, 99_000.0)
        d = rg.check_signal()
        assert d.approved is False, f"1% DD should block L1, got {d.action}"
    def test_peak_tracks_up(self, rg):
        rg.update_equity(PL.TRADE, 105_000.0)
        rg.update_equity(PL.TRADE, 95_000.0)
        s = rg.get_state()
        assert s["trade"]["equity"] == 95_000.0
        assert s["trade"]["peak_equity"] == 105_000.0
    def test_dd_from_peak(self, rg):
        rg.update_equity(PL.TRADE, 110_000.0)
        rg.update_equity(PL.TRADE, 108_900.0)
        d = rg.check_signal()
        assert d.approved is False

class TestCascade:
    def test_l2_cascades_up(self, rg):
        rg.record_trade(-1000, level=PL.STRATEGY)
        s = rg.get_state()
        assert s["trade"]["daily_pnl"] == 0.0, "L1 should NOT be affected"
        for k in LKEYS[1:]:
            assert s[k]["daily_pnl"] == -1000.0, f"{k} should be -1000"
    def test_l1_cascades_all(self, rg):
        rg.record_trade(500, level=PL.TRADE)
        s = rg.get_state()
        for k in LKEYS:
            assert s[k]["daily_pnl"] == 500.0, f"{k} should be 500"
    def test_l6_only_l6(self, rg):
        rg.record_trade(200, level=PL.SYSTEM)
        s = rg.get_state()
        for k in LKEYS[:5]:
            assert s[k]["daily_pnl"] == 0.0, f"{k} should be 0"
        assert s["system"]["daily_pnl"] == 200.0
    def test_accumulate(self, rg):
        rg.record_trade(-500, level=PL.STRATEGY)
        rg.record_trade(-1000, level=PL.STRATEGY)
        rg.record_trade(200, level=PL.STRATEGY)
        s = rg.get_state()
        for k in LKEYS[1:]:
            assert s[k]["daily_pnl"] == -1300.0, f"{k} expected -1300"
class TestHaltResume:
    def test_halt_blocks(self, rg):
        rg.halt_level(PL.TRADE, "test")
        assert rg.check_signal().approved is False
    def test_resume_allows(self, rg):
        rg.halt_level(PL.TRADE, "test")
        rg.resume_level(PL.TRADE)
        assert rg.check_signal().approved is True
    def test_halt_higher_blocks(self, rg):
        rg.halt_level(PL.SYSTEM, "test")
        assert rg.check_signal().approved is False
    def test_flag_set(self, rg):
        rg.halt_level(PL.TYPE, "test")
        assert rg.get_state()["trading_type"]["halted"] is True
        assert rg.get_state()["trade"]["halted"] is False

class TestEmergency:
    def test_blocks_all(self, rg):
        rg.emergency_halt("test")
        assert rg.check_signal().approved is False
    def test_all_halted(self, rg):
        rg.emergency_halt()
        for k in LKEYS:
            assert rg.get_state()[k]["halted"] is True, f"{k}"

class TestCanTrade:
    def test_true_clear(self, rg):
        assert rg.can_trade() is True
    def test_false_halted(self, rg):
        rg.halt_level(PL.TRADE, "t")
        assert rg.can_trade() is False
    def test_false_emergency(self, rg):
        rg.emergency_halt()
        assert rg.can_trade() is False

class TestHealth:
    def test_healthy_fresh(self, rg):
        ok, msg = rg.health_check()
        assert ok is True
    def test_tuple(self, rg):
        h = rg.health_check()
        assert isinstance(h, tuple) and len(h) == 2

class TestNewSession:
    def test_clears_pnl(self, rg):
        rg.record_trade(-2000, level=PL.STRATEGY)
        rg.new_session()
        for k in LKEYS:
            assert rg.get_state()[k]["daily_pnl"] == 0.0
    def test_clears_halts(self, rg):
        rg.emergency_halt()
        rg.new_session()
        for k in LKEYS:
            assert rg.get_state()[k]["halted"] is False
    def test_resets_equity(self, rg):
        rg.update_equity(PL.TRADE, 95_000.0)
        rg.new_session()
        assert rg.get_state()["trade"]["equity"] == 100_000.0
    def test_allows_again(self, rg):
        rg.emergency_halt()
        rg.new_session()
        assert rg.check_signal().approved is True

class TestEdges:
    def test_zero_equity(self):
        rg2 = RiskGuardian(initial_equity=0.0)
        assert isinstance(rg2.check_signal(), GuardianDecision)
    def test_large_loss(self, rg):
        rg.record_trade(-999_999, level=PL.TRADE)
        assert rg.check_signal().approved is False
    def test_profit_no_block(self, rg):
        rg.record_trade(50_000, level=PL.TRADE)
        assert rg.check_signal().approved is True
    def test_idempotent_halt(self, rg):
        for _ in range(5):
            rg.halt_level(PL.TYPE, "t")
            assert rg.check_signal().approved is False
            rg.resume_level(PL.TYPE)
        assert rg.check_signal().approved is True
class TestNotional:
 def test_notional_blocks_over_limit(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_notional_exposure=100000)
  rg.update_notional(PL.TRADE,80000)
  d=rg.check_signal(proposed_notional=30000)
  assert not d.approved
  assert 'Notional' in d.reason

 def test_notional_exact_limit_blocks(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_notional_exposure=100000)
  rg.update_notional(PL.TRADE,50000)
  d=rg.check_signal(proposed_notional=50000)
  assert not d.approved

 def test_notional_allows_below(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_notional_exposure=100000)
  rg.update_notional(PL.TRADE,50000)
  d=rg.check_signal(proposed_notional=40000)
  assert d.approved

 def test_notional_none_skips(self):
  rg=RiskGuardian(100000)
  rg.update_notional(PL.TRADE,999999)
  d=rg.check_signal(proposed_notional=999999)
  assert d.approved

 def test_update_notional_state(self):
  rg=RiskGuardian(100000)
  rg.update_notional(PL.ACCOUNT,50000)
  assert rg.get_state()['account']['notional_exposure']==50000

 def test_notional_resets_on_session(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_notional_exposure=100000)
  rg.update_notional(PL.TRADE,80000)
  rg.new_session()
  assert rg.get_state()['trade']['notional_exposure']==0.0

class TestConcentration:
 def test_concentration_blocks(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_parent_exposure_pct=50.0)
  rg.update_notional(PL.STRATEGY,100000)
  rg.update_notional(PL.TRADE,40000)
  d=rg.check_signal(proposed_notional=20000)
  assert not d.approved
  assert 'Concentration' in d.reason

 def test_concentration_allows(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_parent_exposure_pct=50.0)
  rg.update_notional(PL.STRATEGY,100000)
  rg.update_notional(PL.TRADE,40000)
  d=rg.check_signal(proposed_notional=5000)
  assert d.approved

 def test_concentration_none_skips(self):
  rg=RiskGuardian(100000)
  rg.update_notional(PL.STRATEGY,100000)
  rg.update_notional(PL.TRADE,80000)
  d=rg.check_signal(proposed_notional=30000)
  assert d.approved

 def test_concentration_zero_parent_skips(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,max_parent_exposure_pct=50.0)
  rg.update_notional(PL.TRADE,50000)
  d=rg.check_signal(proposed_notional=50000)
  assert d.approved

 def test_concentration_system_no_parent(self):
  rg=RiskGuardian(100000)
  rg._configs[PL.SYSTEM]=LevelConfig(PL.SYSTEM,max_parent_exposure_pct=50.0)
  rg.update_notional(PL.FIRM,100000)
  rg.update_notional(PL.SYSTEM,80000)
  d=rg.check_signal(proposed_notional=30000)
  assert d.approved
class TestWhipsaw:
    @staticmethod
    def _rg():
        rg=RiskGuardian(100000)
        for lv in [PL.TRADE,PL.STRATEGY,PL.TYPE,PL.ACCOUNT,PL.FIRM,PL.SYSTEM]:
            rg._configs[lv]=LevelConfig(lv,max_daily_loss_pct=100.0,max_consecutive_losses=999,max_drawdown_pct=100.0)
            rg._configs[PL.TRADE]=LevelConfig(PL.TRADE,whipsaw_loss_count=3,whipsaw_time_window_minutes=5.0,whipsaw_cooldown_minutes=10.0,max_daily_loss_pct=100.0,max_consecutive_losses=999,max_drawdown_pct=100.0)
        return rg

    def test_whipsaw_triggers(self):
        rg=self._rg()
        t0=datetime(2024,1,1,10,0,0)
        for i in range(3):rg.record_trade(-100,PL.TRADE,current_time=t0+timedelta(minutes=i))
        d=rg.check_signal(current_time=t0+timedelta(minutes=3))
        assert not d.approved;assert 'Whipsaw' in d.reason

    def test_whipsaw_allows_under_threshold(self):
        rg=self._rg()
        t0=datetime(2024,1,1,10,0,0)
        rg.record_trade(-100,PL.TRADE,current_time=t0)
        rg.record_trade(-100,PL.TRADE,current_time=t0+timedelta(minutes=1))
        d=rg.check_signal(current_time=t0+timedelta(minutes=2))
        assert d.approved

    def test_whipsaw_disabled_by_default(self):
        rg=RiskGuardian(100000)
        for lv in [PL.TRADE,PL.STRATEGY,PL.TYPE,PL.ACCOUNT,PL.FIRM,PL.SYSTEM]:
            rg._configs[lv]=LevelConfig(lv,max_daily_loss_pct=100.0,max_consecutive_losses=999,max_drawdown_pct=100.0)
        t0=datetime(2024,1,1,10,0,0)
        for i in range(5):rg.record_trade(-100,PL.TRADE,current_time=t0+timedelta(minutes=i))
        d=rg.check_signal(current_time=t0+timedelta(minutes=5))
        assert d.approved

    def test_whipsaw_cooldown_expires(self):
        rg=self._rg()
        t0=datetime(2024,1,1,10,0,0)
        for i in range(3):rg.record_trade(-100,PL.TRADE,current_time=t0+timedelta(minutes=i))
        d=rg.check_signal(current_time=t0+timedelta(minutes=15))
        assert d.approved

    def test_whipsaw_resets_on_session(self):
        rg=self._rg()
        t0=datetime(2024,1,1,10,0,0)
        for i in range(3):rg.record_trade(-100,PL.TRADE,current_time=t0+timedelta(minutes=i))
        rg.new_session()
        d=rg.check_signal(current_time=t0+timedelta(minutes=3))
        assert d.approved

class TestDrawdownCascade:
    def test_drawdown_cascade_halt_triggers(self):
        rg = RiskGuardian(100000)
        rg._configs[PL.ACCOUNT] = LevelConfig(
            PL.ACCOUNT, max_drawdown_pct=10.0,
            max_daily_loss_pct=100.0, drawdown_cascade_halt=True)
        rg.record_trade(-15000, PL.ACCOUNT)
        rg.check_signal()
        for lv in [PL.TRADE, PL.STRATEGY, PL.TYPE, PL.ACCOUNT]:
            assert rg._states[lv].halted, f"{lv} should be halted"
        for lv in [PL.FIRM, PL.SYSTEM]:
            assert not rg._states[lv].halted, f"{lv} should not be halted"

    def test_drawdown_no_cascade_by_default(self):
        rg = RiskGuardian(100000)
        rg._configs[PL.ACCOUNT] = LevelConfig(
            PL.ACCOUNT, max_drawdown_pct=10.0, max_daily_loss_pct=100.0)
        rg.record_trade(-15000, PL.ACCOUNT)
        rg.check_signal()
        for lv in [PL.TRADE, PL.STRATEGY, PL.TYPE]:
            assert not rg._states[lv].halted, f"{lv} no halt without cascade"

    def test_drawdown_cascade_from_system(self):
        rg = RiskGuardian(100000)
        rg._configs[PL.SYSTEM] = LevelConfig(
            PL.SYSTEM, max_drawdown_pct=10.0,
            max_daily_loss_pct=100.0, drawdown_cascade_halt=True)
        rg.record_trade(-15000, PL.SYSTEM)
        rg.check_signal()
        for lv in list(PL):
            assert rg._states[lv].halted, f"{lv} should be halted"

    def test_drawdown_cascade_blocks_future(self):
        rg = RiskGuardian(100000)
        rg._configs[PL.ACCOUNT] = LevelConfig(
            PL.ACCOUNT, max_drawdown_pct=10.0,
            max_daily_loss_pct=100.0, drawdown_cascade_halt=True)
        rg.record_trade(-15000, PL.ACCOUNT)
        rg.check_signal()
        d = rg.check_signal()
        assert d.action == VerdictAction.BLOCK

    def test_drawdown_cascade_cleared_by_resume(self):
        rg = RiskGuardian(100000)
        rg._configs[PL.ACCOUNT] = LevelConfig(
            PL.ACCOUNT, max_drawdown_pct=10.0,
            max_daily_loss_pct=100.0, drawdown_cascade_halt=True)
        rg.record_trade(-15000, PL.ACCOUNT)
        rg.check_signal()
        for lv in [PL.TRADE, PL.STRATEGY, PL.TYPE, PL.ACCOUNT]:
            rg.resume_level(lv)
        for lv in [PL.TRADE, PL.STRATEGY, PL.TYPE, PL.ACCOUNT]:
            assert not rg._states[lv].halted

class TestFailClosed:
    def test_fail_closed_on_exception(self):
        rg = RiskGuardian(100000)
        def crash():
            raise RuntimeError("test crash")
        rg._check_trade_level = crash
        d = rg.check_signal()
        assert d.action == VerdictAction.BLOCK
        assert "FAIL-CLOSED" in d.reason
        assert "RuntimeError" in d.reason

    def test_fail_closed_approved_false(self):
        rg = RiskGuardian(100000)
        rg._check_account_level = lambda: (_ for _ in ()).throw(ValueError("boom"))
        d = rg.check_signal()
        assert d.approved is False
        assert d.size_factor == 0.0

    def test_fail_closed_preserves_timestamp(self):
        from datetime import datetime, timezone
        rg = RiskGuardian(100000)
        ct = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rg._check_firm_level = lambda: (_ for _ in ()).throw(RuntimeError("x"))
        d = rg.check_signal(current_time=ct)
        assert d.timestamp == ct

class TestRiskManager:
    def test_propose_trade_allows_increments(self):
        rm = RiskManager(100000)
        d = rm.propose_trade(level=PL.TRADE)
        assert d.action == VerdictAction.ALLOW
        assert rm._rg._states[PL.TRADE].open_positions == 1

    def test_propose_trade_blocked_no_increment(self):
        rm = RiskManager(100000)
        rm._rg._configs[PL.TRADE] = LevelConfig(PL.TRADE, max_positions=0)
        d = rm.propose_trade(level=PL.TRADE)
        assert d.action == VerdictAction.BLOCK
        assert rm._rg._states[PL.TRADE].open_positions == 0

    def test_settle_trade_records_and_decrements(self):
        rm = RiskManager(100000)
        rm.propose_trade(level=PL.TRADE)
        rm.settle_trade(-500, level=PL.TRADE)
        assert rm._rg._states[PL.TRADE].open_positions == 0
        assert rm._rg._states[PL.TRADE].daily_pnl == -500

    def test_audit_trail_records_both(self):
        rm = RiskManager(100000)
        rm.propose_trade(level=PL.TRADE)
        rm.settle_trade(200, level=PL.TRADE)
        trail = rm.get_audit_trail()
        assert len(trail) == 2
        assert trail[0]['action'] == 'propose_trade'
        assert trail[1]['pnl'] == 200

    def test_guardian_property_accessible(self):
        rm = RiskManager(100000)
        assert isinstance(rm.guardian, RiskGuardian)

class TestThreadSafety:
    def test_lock_exists_and_is_rlock(self):
        rg = RiskGuardian(100000)
        assert type(rg._lock).__name__ == 'RLock', f"Expected RLock, got {type(rg._lock).__name__}"
        # Verify reentrant: same thread can acquire twice without deadlock
        rg._lock.acquire()
        rg._lock.acquire()
        rg._lock.release()
        rg._lock.release()

    def test_concurrent_record_trade(self):
        import threading
        rg = RiskGuardian(100000)
        errors = []
        def worker():
            try:
                for _ in range(100):
                    rg.record_trade(100, PL.TRADE)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Errors: {errors}"

    def test_concurrent_check_and_record(self):
        import threading
        rg = RiskGuardian(100000)
        errors = []
        def checker():
            try:
                for _ in range(50):
                    rg.check_signal()
            except Exception as e:
                errors.append(e)
        def recorder():
            try:
                for _ in range(50):
                    rg.record_trade(10, PL.TRADE)
            except Exception as e:
                errors.append(e)
        t1, t2 = threading.Thread(target=checker), threading.Thread(target=recorder)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert not errors

