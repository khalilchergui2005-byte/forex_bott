from __future__ import annotations

"""risk_guardian.py - Risk Guardian: 6-Level Protection System

Protection Levels (bottom-up):
  L1 Trade     - per-trade validation
  L2 Strategy  - per-strategy aggregate limits
  L3 Type      - per-trading-type aggregate limits
  L4 Account   - per-account limits (daily loss, drawdown, positions)
  L5 Firm      - per-prop-firm rules (daily loss, max loss, consistency)
  L6 System    - system-wide limits (total notional, aggregate drawdown)
"""


import threading
import functools

from dataclasses import dataclass, field
from datetime import timedelta, datetime, date, timezone
from enum import Enum
from typing import Optional, List, Dict, Any


class ProtectionLevel(Enum):
    TRADE = "trade"
    STRATEGY = "strategy"
    TYPE = "trading_type"
    ACCOUNT = "account"
    FIRM = "firm"
    SYSTEM = "system"


class VerdictAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDUCE = "reduce"


@dataclass
class RiskVerdict:
    """Decision from a single protection level."""
    level: ProtectionLevel
    action: VerdictAction
    reason: str = ""
    size_factor: float = 1.0


@dataclass
class GuardianDecision:
    """Final aggregated decision from all 6 levels."""
    approved: bool
    action: VerdictAction
    size_factor: float
    verdicts: List[RiskVerdict] = field(default_factory=list)
    reason: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class LevelConfig:
    """Tunable limits for one protection level."""
    level: ProtectionLevel
    max_positions: int = 10
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.10
    max_consecutive_losses: int = 5
    cooldown_minutes: int = 0
    max_notional_exposure: Optional[float] = None
    max_parent_exposure_pct: Optional[float] = None
    whipsaw_loss_count: int = 0
    whipsaw_time_window_minutes: float = 0.0
    whipsaw_cooldown_minutes: float = 0.0
    drawdown_cascade_halt: bool = False


@dataclass
class LevelState:
    """Runtime state tracked for one protection level."""
    level: ProtectionLevel
    equity: float = 0.0
    peak_equity: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    drawdown_pct: float = 0.0
    open_positions: int = 0
    consecutive_losses: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    halted: bool = False
    halt_reason: str = ""
    notional_exposure: float = 0.0
    recent_loss_times: List[datetime] = field(default_factory=list)
    whipsaw_active_until: Optional[datetime] = None
    halt_time: Optional[datetime] = None
    warnings: List[str] = field(default_factory=list)


_LEVEL_ORDER = [
    ProtectionLevel.TRADE,
    ProtectionLevel.STRATEGY,
    ProtectionLevel.TYPE,
    ProtectionLevel.ACCOUNT,
    ProtectionLevel.FIRM,
    ProtectionLevel.SYSTEM,
]


def synchronized(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper

class RiskGuardian:
    """6-Level Risk Protection System.

    Every trade signal passes through all 6 levels.
    The most restrictive verdict wins (BLOCK > REDUCE > ALLOW).
    """

    def __init__(
        self,
        initial_equity: float,
        configs: Optional[Dict[ProtectionLevel, LevelConfig]] = None,
    ):
        self._lock = threading.RLock()
        self.initial_equity = initial_equity
        self._configs = (
            configs if configs is not None else self._default_configs()
        )
        self._states: Dict[ProtectionLevel, LevelState] = {
            level: LevelState(
                level=level,
                equity=initial_equity,
                peak_equity=initial_equity,
            )
            for level in ProtectionLevel
        }
        self._last_date: Optional[date] = None
        self._last_week: Optional[int] = None
        self._last_month: Optional[tuple] = None
        self._healthy = True
        self._health_reason = ""

    @staticmethod
    def _default_configs() -> Dict[ProtectionLevel, LevelConfig]:
        return {
            ProtectionLevel.TRADE: LevelConfig(
                level=ProtectionLevel.TRADE,
                max_positions=1,
                max_daily_loss_pct=1.0,  # FIX-BUG-22: Logical hierarchy
                max_drawdown_pct=1.0,
                max_consecutive_losses=3,
            ),
            ProtectionLevel.STRATEGY: LevelConfig(
                level=ProtectionLevel.STRATEGY,
                max_positions=3,
                max_daily_loss_pct=2.0,
                max_drawdown_pct=2.0,
                max_consecutive_losses=5,
            ),
            ProtectionLevel.TYPE: LevelConfig(
                level=ProtectionLevel.TYPE,
                max_positions=5,
                max_daily_loss_pct=3.0,
                max_drawdown_pct=3.0,
                max_consecutive_losses=7,
            ),
            ProtectionLevel.ACCOUNT: LevelConfig(
                level=ProtectionLevel.ACCOUNT,
                max_positions=10,
                max_daily_loss_pct=4.0,
                max_drawdown_pct=4.0,
                max_consecutive_losses=10,
            ),
            ProtectionLevel.FIRM: LevelConfig(
                level=ProtectionLevel.FIRM,
                max_positions=20,
                max_daily_loss_pct=5.0,
                max_drawdown_pct=5.0,
                max_consecutive_losses=15,
            ),
            ProtectionLevel.SYSTEM: LevelConfig(
                level=ProtectionLevel.SYSTEM,
                max_positions=50,
                max_daily_loss_pct=6.0,
                max_drawdown_pct=6.0,
                max_consecutive_losses=999,
            ),
        }

    # ?? Core Check ?????????????????????????????????????

    @synchronized
    def check_signal(
        self,
        current_time: Optional[datetime] = None,
        strategy_name: Optional[str] = None,
        trading_type: Optional[str] = None,
        account_id: Optional[str] = None,
        firm_id: Optional[str] = None,
        proposed_size: float = 0.0,
        proposed_notional: float = 0.0,
        side: str = "long",
    ) -> GuardianDecision:
        """Main entry: check signal against all 6 levels. Fail-closed."""
        try:
            self._proposed_notional = proposed_notional
            self._current_time = current_time
            self._reset_periodic_pnl(current_time)
            verdicts: List[RiskVerdict] = []
            verdicts.append(self._check_trade_level())
            verdicts.append(self._check_strategy_level())
            verdicts.append(self._check_type_level())
            verdicts.append(self._check_account_level())
            verdicts.append(self._check_firm_level())
            verdicts.append(self._check_system_level())
            return self._aggregate_verdicts(verdicts)
        except Exception as e:
            return GuardianDecision(
                approved=False,
                action=VerdictAction.BLOCK,
                size_factor=0.0,
                reason=f"FAIL-CLOSED: {type(e).__name__}: {e}",
                timestamp=current_time or datetime.now(timezone.utc),
            )

    def _aggregate_verdicts(
        self, verdicts: List[RiskVerdict]
    ) -> GuardianDecision:
        """Pick the most restrictive verdict."""
        worst_action = VerdictAction.ALLOW
        worst_factor = 1.0
        reasons: List[str] = []

        for v in verdicts:
            if v.action == VerdictAction.BLOCK:
                worst_action = VerdictAction.BLOCK
                reasons.append(
                    f"[{v.level.value}] {v.reason}"
                )
            elif v.action == VerdictAction.REDUCE:
                if worst_action != VerdictAction.BLOCK:
                    worst_action = VerdictAction.REDUCE
                    worst_factor = min(worst_factor, v.size_factor)
                    reasons.append(
                        f"[{v.level.value}] {v.reason}"
                    )

        if worst_action == VerdictAction.BLOCK:
            return GuardianDecision(
                approved=False,
                action=VerdictAction.BLOCK,
                size_factor=0.0,
                verdicts=verdicts,
                reason=" | ".join(reasons) if reasons else "Blocked",
            )
        if worst_action == VerdictAction.REDUCE:
            return GuardianDecision(
                approved=True,
                action=VerdictAction.REDUCE,
                size_factor=worst_factor,
                verdicts=verdicts,
                reason=" | ".join(reasons) if reasons else "Reduced",
            )
        return GuardianDecision(
            approved=True,
            action=VerdictAction.ALLOW,
            size_factor=1.0,
            verdicts=verdicts,
            reason="All levels clear",
        )


    # ?? Level Checkers ?????????????????????????????????

    def _reset_periodic_pnl(self, current_time: Optional[datetime] = None):
        """Reset daily/weekly/monthly PnL counters if a new period has started."""
        if current_time is None:
            return
            
        today = current_time.date()
        
        # Daily reset
        if not hasattr(self, '_last_daily_reset') or self._last_daily_reset != today:
            for state in self._states.values():
                state.daily_pnl = 0.0
            self._last_daily_reset = today
            
        # Weekly reset (Monday)
        week_start = today - timedelta(days=today.weekday())
        if not hasattr(self, '_last_weekly_reset') or self._last_weekly_reset != week_start:
            for state in self._states.values():
                state.weekly_pnl = 0.0
            self._last_weekly_reset = week_start

        # Monthly reset
        month_start = today.replace(day=1)
        if not hasattr(self, '_last_monthly_reset') or self._last_monthly_reset != month_start:
            for state in self._states.values():
                state.monthly_pnl = 0.0
            self._last_monthly_reset = month_start

    def _cascade_halt(self, breached_level):
        order = list(ProtectionLevel)
        idx = order.index(breached_level)
        reason = "Drawdown cascade: " + breached_level.value
        for lv in order[:idx + 1]:
            self._states[lv].halted = True
            self._states[lv].halt_reason = reason
            self._states[lv].halt_time = self._current_time  # FIX-BUG-08: Moved inside loop

    def _check_level(self, level: ProtectionLevel) -> RiskVerdict:
        """Generic check for any level (FIX-BUG-27: DRY violation)."""
        state = self._states[level]
        cfg = self._configs[level]
        if state.halted:
            return RiskVerdict(level, VerdictAction.BLOCK, state.halt_reason)
        if state.open_positions >= cfg.max_positions:
            return RiskVerdict(level, VerdictAction.BLOCK, f"{level.value} max positions ({state.open_positions}/{cfg.max_positions})")
        if state.consecutive_losses >= cfg.max_consecutive_losses:  # FIX-BUG-09: Added to TRADE
            return RiskVerdict(level, VerdictAction.BLOCK, f"{level.value} consecutive losses ({state.consecutive_losses}/{cfg.max_consecutive_losses})")
        daily_limit = (cfg.max_daily_loss_pct / 100.0) * self.initial_equity
        if state.daily_pnl < -daily_limit:
            return RiskVerdict(level, VerdictAction.BLOCK, f"{level.value} daily loss ({state.daily_pnl:.2f})")
        if state.drawdown_pct >= (cfg.max_drawdown_pct / 100.0):
            if cfg.drawdown_cascade_halt:
                self._cascade_halt(level)
            msg = f"Drawdown cascade halt ({state.drawdown_pct:.1%})" if cfg.drawdown_cascade_halt else f"Drawdown ({state.drawdown_pct:.1%})"
            return RiskVerdict(level, VerdictAction.BLOCK, msg)
        if cfg.max_notional_exposure is not None:
            if state.notional_exposure + self._proposed_notional >= cfg.max_notional_exposure:
                return RiskVerdict(level, VerdictAction.BLOCK, f"Notional limit ({state.notional_exposure:.0f}+{self._proposed_notional:.0f}>={cfg.max_notional_exposure:.0f})")
        if cfg.max_parent_exposure_pct is not None:
            parent_map = {
                ProtectionLevel.TRADE: ProtectionLevel.STRATEGY,
                ProtectionLevel.STRATEGY: ProtectionLevel.TYPE,
                ProtectionLevel.TYPE: ProtectionLevel.ACCOUNT,
                ProtectionLevel.ACCOUNT: ProtectionLevel.FIRM,
                ProtectionLevel.FIRM: ProtectionLevel.SYSTEM,
            }
            parent_lv = parent_map.get(level)
            if parent_lv:
                ps = self._states[parent_lv]
                if ps.notional_exposure > 0:
                    ma = (cfg.max_parent_exposure_pct / 100.0) * ps.notional_exposure
                    if state.notional_exposure + self._proposed_notional > ma:
                        return RiskVerdict(level, VerdictAction.BLOCK, f"Concentration: {level.value} ({state.notional_exposure + self._proposed_notional:.0f}) > {cfg.max_parent_exposure_pct}% of {parent_lv.value} ({ma:.0f})")
        if state.whipsaw_active_until is not None:
            ct = self._current_time
            if ct is not None and ct < state.whipsaw_active_until:
                rem = (state.whipsaw_active_until - ct).total_seconds() / 60
                return RiskVerdict(level, VerdictAction.BLOCK, f"Whipsaw: cooldown active ({rem:.1f} min remaining)")
            state.whipsaw_active_until = None
        return RiskVerdict(level, VerdictAction.ALLOW)

    def _check_trade_level(self) -> RiskVerdict:
        return self._check_level(ProtectionLevel.TRADE)

    def _check_strategy_level(self) -> RiskVerdict:
        return self._check_level(ProtectionLevel.STRATEGY)

    def _check_type_level(self) -> RiskVerdict:
        return self._check_level(ProtectionLevel.TYPE)

    def _check_account_level(self) -> RiskVerdict:
        return self._check_level(ProtectionLevel.ACCOUNT)

    def _check_firm_level(self) -> RiskVerdict:
        return self._check_level(ProtectionLevel.FIRM)

    def _check_system_level(self) -> RiskVerdict:
        return self._check_level(ProtectionLevel.SYSTEM)

    @synchronized
    def record_trade(
        self,
        pnl: float,
        level: ProtectionLevel = ProtectionLevel.ACCOUNT,
        current_time: Optional[datetime] = None,
    ):
        """Record completed trade PnL; cascades to higher levels."""
        self._reset_periodic_pnl(current_time)
        idx = _LEVEL_ORDER.index(level)
        for lvl in _LEVEL_ORDER[idx:]:
            state = self._states[lvl]
            state.total_trades += 1
            state.daily_pnl += pnl
            state.weekly_pnl += pnl
            state.monthly_pnl += pnl
            if pnl > 0:
                state.winning_trades += 1
                state.consecutive_losses = 0
            else:
                state.consecutive_losses += 1
                cfg=self._configs[lvl]
                if current_time is not None and cfg.whipsaw_time_window_minutes>0:
                    state.recent_loss_times.append(current_time)
                    cutoff=current_time-timedelta(minutes=cfg.whipsaw_time_window_minutes)
                    state.recent_loss_times=[t for t in state.recent_loss_times if t>cutoff]
                    if len(state.recent_loss_times)>=cfg.whipsaw_loss_count and cfg.whipsaw_cooldown_minutes>0:
                        state.whipsaw_active_until=current_time+timedelta(minutes=cfg.whipsaw_cooldown_minutes)
            state.equity += pnl
            if state.equity > state.peak_equity:
                state.peak_equity = state.equity
            if state.peak_equity > 0:
                state.drawdown_pct = (
                    (state.peak_equity - state.equity)
                    / state.peak_equity
                )
            else:
                state.drawdown_pct = 0.0

    def update_positions(self, level: ProtectionLevel, count: int):
        """Set open position count for a level."""
        self._states[level].open_positions = count

    def update_equity(self, level: ProtectionLevel, equity: float):
        """Set equity and recalculate drawdown for a level."""
        state = self._states[level]
        state.equity = equity
        if equity > state.peak_equity:
            state.peak_equity = equity
        if state.peak_equity > 0:
            state.drawdown_pct = (
                (state.peak_equity - equity) / state.peak_equity
            )

    def update_notional(self, level: ProtectionLevel, total_notional: float):
        """Set total notional exposure for a protection level."""
        self._states[level].notional_exposure = total_notional

    @synchronized
    def halt_level(self, level: ProtectionLevel, reason: str):
        """Manually halt a protection level."""
        state = self._states[level]
        state.halted = True
        state.halt_reason = reason
        state.halt_time = datetime.now(timezone.utc)

    @synchronized
    def resume_level(self, level: ProtectionLevel):
        """Resume a halted protection level."""
        state = self._states[level]
        state.halted = False
        state.halt_reason = ""
        state.halt_time = None
        state.consecutive_losses = 0

    @synchronized
    def new_session(self, current_time: Optional[datetime] = None):
        """Reset all counters and states for a new trading session."""
        self._reset_periodic_pnl(current_time)
        for state in self._states.values():
            state.daily_pnl = 0.0
            state.weekly_pnl = 0.0
            state.monthly_pnl = 0.0
            state.consecutive_losses = 0
            state.halted = False
            state.halt_reason = ""
            state.open_positions = 0
            state.notional_exposure = 0.0
            state.recent_loss_times = []
            state.whipsaw_active_until = None
            state.equity = self.initial_equity
            state.peak_equity = self.initial_equity
            state.drawdown_pct = 0.0
            state.warnings.clear()

    def health_check(self) -> tuple:
        """Verify Risk Guardian integrity.
        Returns (healthy: bool, reason: str)."""
        self._healthy = False
        try:
            for level in ProtectionLevel:
                if level not in self._states:
                    return False, f"Missing state: {level.value}"
                if level not in self._configs:
                    return False, f"Missing config: {level.value}"
                state = self._states[level]
                if state.equity < 0:
                    return False, (
                        f"Negative equity at {level.value}: "
                        f"{state.equity}"
                    )
                if not (0.0 <= state.drawdown_pct <= 1.0):
                    return False, (
                        f"Invalid drawdown at {level.value}: "
                        f"{state.drawdown_pct}"
                    )
            self._healthy = True
            self._health_reason = ""
            return True, ""
        except Exception as e:
            self._healthy = False
            self._health_reason = str(e)
            return False, str(e)

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # ?? State Access ????????????????????????????????????

    def get_state(
        self, level: Optional[ProtectionLevel] = None
    ) -> Dict[str, Any]:
        """Get state for one level or all levels."""
        if level is not None:
            s = self._states[level]
            return {
                "level": s.level.value,
                "equity": s.equity,
                "peak_equity": s.peak_equity,
                "daily_pnl": s.daily_pnl,
                "weekly_pnl": s.weekly_pnl,
                "monthly_pnl": s.monthly_pnl,
                "drawdown_pct": s.drawdown_pct,
                "open_positions": s.open_positions,
                "consecutive_losses": s.consecutive_losses,
                "total_trades": s.total_trades,
                "winning_trades": s.winning_trades,
                "halted": s.halted,
                "halt_reason": s.halt_reason,
                "warnings": s.warnings,
                "notional_exposure": s.notional_exposure,
                "whipsaw_active": s.whipsaw_active_until is not None,
                "recent_loss_count": len(s.recent_loss_times),
            }
        return {
            lvl.value: self.get_state(lvl)
            for lvl in ProtectionLevel
        }

    def can_trade(
        self, level: Optional[ProtectionLevel] = None
    ) -> bool:
        """Quick check: is trading allowed?"""
        if not self._healthy:
            return False
        if level is not None:
            return not self._states[level].halted
        return all(
            not s.halted for s in self._states.values()
        )

    @synchronized
    def emergency_halt(self, reason: str = "Emergency halt"):
        """Halt ALL levels immediately."""
        for level in ProtectionLevel:
            self.halt_level(level, reason)
class RiskManager:
    """Production facade: atomic check+execute with audit trail."""
    def __init__(self, initial_equity: float):
        self._rg = RiskGuardian(initial_equity)
        self._audit: List[dict] = []
    def propose_trade(self, level=None, **kw):
        if level is None: level = ProtectionLevel.TRADE
        d = self._rg.check_signal(**kw)
        if d.action == VerdictAction.ALLOW:
            self._rg._states[level].open_positions += 1  # FIX: Facade direct access for state mutation
        self._audit.append({'action':'propose_trade','level':level.value,'allowed':d.action==VerdictAction.ALLOW,'reason':d.reason})
        return d
    def settle_trade(self, pnl: float, level=None, current_time=None):
        if level is None: level = ProtectionLevel.TRADE
        self._rg.record_trade(pnl, level, current_time=current_time)  # FIX-BUG-10: Pass current_time
        st = self._rg._states[level]  # FIX: Facade direct access for state mutation
        if st.open_positions > 0: st.open_positions -= 1
        self._audit.append({'action':'settle_trade','level':level.value,'pnl':pnl})
    def get_audit_trail(self) -> List[dict]:
        return list(self._audit)
    @property
    def guardian(self) -> RiskGuardian:
        return self._rg
