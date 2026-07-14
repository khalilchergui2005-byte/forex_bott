"""Risk Guardian - circuit breakers, P&L tracking, position limits."""
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class CircuitBreakerConfig:
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    caution_drawdown_pct: float = 0.05
    max_consecutive_losses: int = 5
    max_positions: int = 10
    max_position_size_pct: float = 0.05
@dataclass
class RiskState:
    level: str
    equity: float
    peak_equity: float
    daily_pnl: float
    daily_pnl_pct: float
    drawdown_pct: float
    open_positions: int
    consecutive_losses: int
    warnings: list = field(default_factory=list)
    reason: str = ""
class RiskGuardian:
    def __init__(self, initial_equity: float,
                 config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.daily_realized = 0.0
        self.unrealized_pnl = 0.0
        self.consecutive_losses = 0
        self.positions: dict = {}
        self._halted = False
        self._halt_reason = ""
        self._session_start = float(initial_equity)

    @property
    def daily_pnl(self) -> float:
        return self.daily_realized + self.unrealized_pnl

    @property
    def daily_pnl_pct(self) -> float:
        if self._session_start <= 0:
            return 0.0
        return self.daily_pnl / self._session_start

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max((self.peak_equity - self.equity) / self.peak_equity, 0.0)

    def update_positions(self, positions: dict) -> None:
        self.positions = positions
        self.unrealized_pnl = sum(p.get("pnl", 0.0) for p in positions.values())
        self._sync_equity()

    def record_trade(self, pnl: float) -> None:
        self.daily_realized += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self._sync_equity()

    def new_session(self, current_equity: float = None) -> None:
        if current_equity is not None:
            self.equity = float(current_equity)
        self._session_start = self.equity
        self.peak_equity = self.equity
        self.daily_realized = 0.0
        self.unrealized_pnl = 0.0
        self.consecutive_losses = 0
        self._halted = False
        self._halt_reason = ""
        self.positions = {}

    def _sync_equity(self) -> None:
        self.equity = self._session_start + self.daily_pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def get_state(self) -> RiskState:
        warnings = []
        level = "normal"
        reason = "All clear"
        if self._halted:
            return RiskState(level="halted", equity=self.equity,
                peak_equity=self.peak_equity, daily_pnl=self.daily_pnl,
                daily_pnl_pct=self.daily_pnl_pct,
                drawdown_pct=self.drawdown_pct,
                open_positions=len(self.positions),
                consecutive_losses=self.consecutive_losses,
                warnings=["Manual halt"], reason=self._halt_reason)
        if self.drawdown_pct >= self.config.max_drawdown_pct:
            self._halted = True
            r = f"Max drawdown {self.drawdown_pct:.2%} >= {self.config.max_drawdown_pct:.2%}"
            self._halt_reason = r
            return RiskState(level="halted", equity=self.equity,
                peak_equity=self.peak_equity, daily_pnl=self.daily_pnl,
                daily_pnl_pct=self.daily_pnl_pct,
                drawdown_pct=self.drawdown_pct,
                open_positions=len(self.positions),
                consecutive_losses=self.consecutive_losses,
                warnings=[r], reason=r)
        if self.daily_pnl_pct <= -self.config.max_daily_loss_pct:
            self._halted = True
            r = f"Daily loss {abs(self.daily_pnl_pct):.2%} >= limit {self.config.max_daily_loss_pct:.2%}"
            self._halt_reason = r
            return RiskState(level="halted", equity=self.equity,
                peak_equity=self.peak_equity, daily_pnl=self.daily_pnl,
                daily_pnl_pct=self.daily_pnl_pct,
                drawdown_pct=self.drawdown_pct,
                open_positions=len(self.positions),
                consecutive_losses=self.consecutive_losses,
                warnings=[r], reason=r)
        if self.drawdown_pct >= self.config.caution_drawdown_pct:
            level = "caution"
            warnings.append(f"Drawdown {self.drawdown_pct:.2%}")
        if self.daily_pnl_pct <= -self.config.max_daily_loss_pct * 0.6:
            if level == "normal":
                level = "caution"
            warnings.append(f"Daily loss {abs(self.daily_pnl_pct):.2%}")
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            level = "restricted"
            warnings.append(f"{self.consecutive_losses} consecutive losses")
        return RiskState(level=level, equity=self.equity,
            peak_equity=self.peak_equity, daily_pnl=self.daily_pnl,
            daily_pnl_pct=self.daily_pnl_pct,
            drawdown_pct=self.drawdown_pct,
            open_positions=len(self.positions),
            consecutive_losses=self.consecutive_losses,
            warnings=warnings, reason=reason)

    def can_trade(self, pair: str = None) -> tuple:
        state = self.get_state()
        if state.level in ("halted", "restricted"):
            return False, state
        if pair and len(self.positions) >= self.config.max_positions:
            state.reason = f"Max positions ({self.config.max_positions}) reached"
            return False, state
        return True, state

    def emergency_halt(self, reason: str = "Manual emergency halt") -> RiskState:
        self._halted = True
        self._halt_reason = reason
        return self.get_state()
