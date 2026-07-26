"""Multi-Period Validation — checks consistency across time segments."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PeriodResult:
    period_id: int
    start_idx: int
    end_idx: int
    n_trades: int
    pf: float
    pnl: float
    max_dd_pct: float
    sharpe: float
    profitable: bool


@dataclass
class MultiPeriodReport:
    n_periods: int = 0
    n_profitable: int = 0
    profit_rate: float = 0.0
    worst_period_pf: float = 0.0
    best_period_pf: float = 0.0
    avg_period_pf: float = 0.0
    std_period_pf: float = 0.0
    periods: list = field(default_factory=list)
    classification: str = "UNKNOWN"
    reasons: list = field(default_factory=list)


class MultiPeriodValidator:
    """Splits trades into N periods, checks each is profitable."""

    def __init__(self, n_periods=4, min_profit_rate=0.8, min_worst_pf=0.8):
        self.n_periods = n_periods
        self.min_profit_rate = min_profit_rate
        self.min_worst_pf = min_worst_pf

    @staticmethod
    def _period_metrics(trades: np.ndarray) -> dict:
        wins = trades[trades > 0].sum() if (trades > 0).any() else 0.0
        losses = abs(trades[trades < 0].sum()) if (trades < 0).any() else 1e-10
        pf = wins / losses
        pnl = float(trades.sum())
        eq = np.concatenate([[0.0], np.cumsum(trades)])
        peak = np.maximum.accumulate(eq)
        dd = np.where(peak > 0, (peak - eq) / peak, 0.0)
        max_dd = float(dd.max()) if len(dd) > 0 else 0.0
        sharpe = float(trades.mean() / trades.std()) if trades.std() > 0 else 0.0
        return {"pf": pf, "pnl": pnl, "max_dd": max_dd, "sharpe": sharpe}

    def run(self, trades: list) -> MultiPeriodReport:
        arr = np.array(trades, dtype=np.float64)
        n = len(arr)
        if n < 20:
            return MultiPeriodReport(
                classification="FAIL",
                reasons=[f"Insufficient trades: {n} < 20"])

        chunk = n // self.n_periods
        periods = []
        pfs = []

        for i in range(self.n_periods):
            start = i * chunk
            end = start + chunk if i < self.n_periods - 1 else n
            segment = arr[start:end]
            if len(segment) < 5:
                continue
            m = self._period_metrics(segment)
            profitable = m["pf"] > 1.0 and m["pnl"] > 0
            periods.append(PeriodResult(
                period_id=i, start_idx=start, end_idx=end,
                n_trades=len(segment), pf=m["pf"], pnl=m["pnl"],
                max_dd_pct=m["max_dd"], sharpe=m["sharpe"],
                profitable=profitable,
            ))
            pfs.append(m["pf"])

        if not periods:
            return MultiPeriodReport(
                classification="FAIL", reasons=["No valid periods"])

        pfs_arr = np.array(pfs)
        n_profit = sum(1 for p in periods if p.profitable)
        profit_rate = n_profit / len(periods)
        worst_pf = float(pfs_arr.min())
        best_pf = float(pfs_arr.max())

        reasons = []
        if profit_rate < self.min_profit_rate:
            reasons.append(
                f"Profit rate {profit_rate:.0%} < {self.min_profit_rate:.0%}")
        if worst_pf < self.min_worst_pf:
            reasons.append(
                f"Worst period PF {worst_pf:.3f} < {self.min_worst_pf}")

        cls = "PASS" if not reasons else "FAIL"
        return MultiPeriodReport(
            n_periods=len(periods), n_profitable=n_profit,
            profit_rate=profit_rate, worst_period_pf=worst_pf,
            best_period_pf=best_pf,
            avg_period_pf=float(pfs_arr.mean()),
            std_period_pf=float(pfs_arr.std()),
            periods=periods, classification=cls, reasons=reasons,
        )
