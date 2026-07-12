"""Stress Test — validates strategy robustness under adverse market conditions."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StressScenario:
    name: str
    description: str
    n_trades: int
    pf: float
    total_pnl: float
    max_dd_pct: float
    pnl_change_pct: float
    passed: bool


@dataclass
class StressReport:
    n_trades_original: int = 0
    real_pf: float = 0.0
    real_pnl: float = 0.0
    real_max_dd_pct: float = 0.0
    scenarios: list = field(default_factory=list)
    n_scenarios: int = 0
    n_passed: int = 0
    worst_scenario: str = ""
    classification: str = "UNKNOWN"
    reasons: list = field(default_factory=list)


class StressTest:
    """Applies 6 stress scenarios to trade sequence."""

    def __init__(self, slippage_pct=0.1, gap_pct=0.02, gap_mult=3.0, seed=None):
        self.slippage_pct = slippage_pct
        self.gap_pct = gap_pct
        self.gap_mult = gap_mult
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _metrics(trades: np.ndarray) -> dict:
        wins = trades[trades > 0].sum() if (trades > 0).any() else 0.0
        losses = abs(trades[trades < 0].sum()) if (trades < 0).any() else 1e-10
        pf = wins / losses
        pnl = float(trades.sum())
        eq = np.concatenate([[0.0], np.cumsum(trades)])
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0
        return {"pf": pf, "pnl": pnl, "max_dd": max_dd}

    def _pnl_pct(self, new_pnl, real_pnl):
        if real_pnl == 0:
            return 0.0
        return (new_pnl - real_pnl) / abs(real_pnl) * 100

    def _scenario(self, name, desc, trades, real_pnl):
        m = self._metrics(np.array(trades, dtype=np.float64))
        passed = m["pf"] > 1.0 and m["pnl"] > 0
        return StressScenario(
            name=name, description=desc, n_trades=len(trades),
            pf=m["pf"], total_pnl=m["pnl"], max_dd_pct=m["max_dd"],
            pnl_change_pct=self._pnl_pct(m["pnl"], real_pnl), passed=passed,
        )

    def run(self, trades: list) -> StressReport:
        arr = np.array(trades, dtype=np.float64)
        n = len(arr)
        if n < 10:
            return StressReport(classification="FAIL",
                                reasons=[f"Insufficient trades: {n} < 10"])
        real = self._metrics(arr)
        avg_size = float(np.mean(np.abs(arr)))
        slip = avg_size * self.slippage_pct
        n_gaps = max(1, int(n * self.gap_pct))
        std = arr.std()
        scenarios = []

        # 1. Spread 2x
        t = arr.copy(); t[t < 0] *= 2.0
        scenarios.append(self._scenario("spread_2x", "Spread widened 2x", t, real["pnl"]))

        # 2. Spread 3x
        t = arr.copy(); t[t < 0] *= 3.0
        scenarios.append(self._scenario("spread_3x", "Spread widened 3x", t, real["pnl"]))

        # 3. Slippage
        t = arr - slip
        scenarios.append(self._scenario("slippage", f"Slippage {slip:.4f}/trade", t, real["pnl"]))

        # 4. Gap shock
        if std > 0:
            gaps = -np.abs(self.rng.normal(0, std * self.gap_mult, size=n_gaps))
            t = np.concatenate([arr, gaps])
        else:
            t = arr
        scenarios.append(self._scenario("gap_shock",
            f"{n_gaps} gap events ({self.gap_mult}x vol)", t, real["pnl"]))

        # 5. Win reduction 20%
        t = arr.copy(); t[t > 0] *= 0.8
        scenarios.append(self._scenario("win_reduce_20",
            "Winning trades reduced 20%", t, real["pnl"]))

        # 6. Combined: spread 2x + slippage + win reduce
        t = arr.copy(); t[t < 0] *= 2.0
        t = t - slip
        t[t > 0] *= 0.8
        scenarios.append(self._scenario("combined",
            "Spread 2x + Slippage + Win -20%", t, real["pnl"]))

        n_passed = sum(1 for s in scenarios if s.passed)
        failed = [f"{s.name}: PF={s.pf:.3f} PnL={s.total_pnl:.1f}"
                  for s in scenarios if not s.passed]
        worst = min(scenarios, key=lambda s: s.pf).name
        cls = "PASS" if n_passed == len(scenarios) else "FAIL"
        return StressReport(
            n_trades_original=n, real_pf=real["pf"], real_pnl=real["pnl"],
            real_max_dd_pct=real["max_dd"], scenarios=scenarios,
            n_scenarios=len(scenarios), n_passed=n_passed,
            worst_scenario=worst, classification=cls, reasons=failed,
        )
