"""
CPCV Validator - Validate strategies via CPCV.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from cpv_temporal import TemporalCPCV, CPCVSplit

@dataclass
class SplitResult:
    """Metrics for one CPCV split."""
    split_id: int
    train_groups: tuple
    test_groups: tuple
    n_trades: int
    profit_factor: float
    win_rate: float
    expectancy: float
    total_pnl: float
    max_drawdown_pct: float
    sharpe: float

@dataclass
class CPCVReport:
    """Comprehensive CPCV validation report."""
    strategy_name: str
    n_splits: int
    n_passed: int
    defect_rate: float
    pf_mean: float
    pf_std: float
    pf_min: float
    pf_max: float
    pf_p5: float
    pf_p50: float
    pf_p95: float
    avg_win_rate: float
    avg_expectancy: float
    avg_max_dd: float
    avg_sharpe: float
    avg_n_trades: float
    pbo: float
    splits: List[SplitResult] = field(
        default_factory=list)
    classification: str = "PENDING"
    reasons: List[str] = field(
        default_factory=list)

    def summary(self):
        """Formatted summary of validation results."""
        sep = "=" * 50
        lines = [
            sep,
            f"CPCV Report: {self.strategy_name}",
            sep,
            f"Splits: {self.n_splits}, "
            f"Passed: {self.n_passed}",
            f"Defect Rate: {self.defect_rate:.1%}",
            f"PBO: {self.pbo:.1%}",
            "",
            "PF Distribution:",
            f"  Mean: {self.pf_mean:.3f} "
            f"+/- {self.pf_std:.3f}",
            f"  P5:   {self.pf_p5:.3f}",
            f"  P50:  {self.pf_p50:.3f}",
            f"  P95:  {self.pf_p95:.3f}",
            f"  Min:  {self.pf_min:.3f}",
            f"  Max:  {self.pf_max:.3f}",
            "",
            "Avg Metrics:",
            f"  WR:     {self.avg_win_rate:.1%}",
            f"  Expect: {self.avg_expectancy:.4f}",
            f"  DD:     {self.avg_max_dd:.1%}",
            f"  Sharpe: {self.avg_sharpe:.3f}",
            f"  Trades: {self.avg_n_trades:.0f}",
            "",
            f"Result: {self.classification}",
        ]
        if self.reasons:
            for r in self.reasons:
                lines.append(f"  - {r}")
        lines.append(sep)
        return "\n".join(lines)

class CPCVValidator:
    """
    Validate strategies using CPCV.
    backtest_fn(data, train_idx, test_idx) -> dict
    Must return: {trades: list, equity: list}
    """

    def __init__(self, backtest_fn, n_groups=6,
                 n_test_groups=2, purge_gap=24,
                 embargo_pct=0.01):
        self.cpcv = TemporalCPCV(
            n_groups=n_groups,
            n_test_groups=n_test_groups,
            purge_gap=purge_gap,
            embargo_pct=embargo_pct)
        self.backtest_fn = backtest_fn

    def validate(self, data, strategy_name="unnamed",
                 min_trades=30):
        """Run full CPCV validation."""
        n_bars = len(data)
        splits = self.cpcv.generate_splits(n_bars)
        results = []
        for sp in splits:
            r = self._run_split(data, sp, min_trades)
            if r is not None:
                results.append(r)
        return self._build_report(strategy_name, results)

    def _run_split(self, data, split, min_trades):
        try:
            bt = self.backtest_fn(
                data, split.train_indices,
                split.test_indices)
            trades = bt.get("trades", [])
            equity = bt.get("equity", [])
            if len(trades) < min_trades:
                return None
            return self._metrics(
                trades, equity, split)
        except Exception:
            return None

    def _metrics(self, trades, equity, split):
        t = np.array(trades, dtype=float)
        wins = t[t > 0]
        losses = t[t < 0]
        gp = wins.sum() if len(wins) else 0.0
        gl = abs(losses.sum()) if len(losses) else 1e-10
        pf = gp / gl
        wr = len(wins) / len(t) if len(t) else 0.0
        exp = t.mean() if len(t) else 0.0
        eq = np.array(equity) if equity else np.cumsum(t)
        rm = np.maximum.accumulate(eq)
        dd = (rm - eq) / (rm + 1e-10)
        mdd = dd.max() if len(dd) else 0.0
        sh = ((t.mean() / (t.std() + 1e-10))
              * np.sqrt(252)) if len(t) > 1 else 0.0
        return SplitResult(
            split_id=split.split_id,
            train_groups=split.train_groups,
            test_groups=split.test_groups,
            n_trades=len(t), profit_factor=pf,
            win_rate=wr, expectancy=exp,
            total_pnl=t.sum(),
            max_drawdown_pct=mdd, sharpe=sh)

    def _build_report(self, name, results):
        if not results:
            return CPCVReport(
                strategy_name=name, n_splits=0,
                n_passed=0, defect_rate=1.0,
                pf_mean=0, pf_std=0,
                pf_min=0, pf_max=0,
                pf_p5=0, pf_p50=0, pf_p95=0,
                avg_win_rate=0, avg_expectancy=0,
                avg_max_dd=0, avg_sharpe=0,
                avg_n_trades=0, pbo=1.0,
                classification="FAIL",
                reasons=["No valid splits"])
        pfs = np.array([r.profit_factor
                     for r in results])
        np_ = int(np.sum(pfs >= 1.0))
        pbo = self._compute_pbo(results)
        rpt = CPCVReport(
            strategy_name=name,
            n_splits=len(results),
            n_passed=np_,
            defect_rate=1.0 - np_ / len(results),
            pf_mean=float(np.mean(pfs)),
            pf_std=float(np.std(pfs)),
            pf_min=float(np.min(pfs)),
            pf_max=float(np.max(pfs)),
            pf_p5=float(np.percentile(pfs, 5)),
            pf_p50=float(np.percentile(pfs, 50)),
            pf_p95=float(np.percentile(pfs, 95)),
            avg_win_rate=float(np.mean(
                [r.win_rate for r in results])),
            avg_expectancy=float(np.mean(
                [r.expectancy for r in results])),
            avg_max_dd=float(np.mean(
                [r.max_drawdown_pct for r in results])),
            avg_sharpe=float(np.mean(
                [r.sharpe for r in results])),
            avg_n_trades=float(np.mean(
                [r.n_trades for r in results])),
            pbo=pbo, splits=results)
        self._classify(rpt)
        return rpt

    def _compute_pbo(self, results):
        if len(results) < 4:
            return 0.5
        pfs = np.array([r.profit_factor
                     for r in results])
        med = np.median(pfs)
        best = np.argmax(pfs)
        if med > 0:
            gap = (pfs[best] - med) / (med + 1e-10)
            return round(
                max(0, min(1, 1 - gap)), 4)
        return 1.0

    def _classify(self, rpt):
        reasons = []
        if rpt.defect_rate > 0.5:
            reasons.append(
                f"Defect {rpt.defect_rate:.0%} > 50%")
        if rpt.pf_mean < 1.1:
            reasons.append(
                f"Mean PF {rpt.pf_mean:.3f} < 1.1")
        if rpt.pf_p5 < 0.8:
            reasons.append(
                f"P5 PF {rpt.pf_p5:.3f} < 0.8")
        if rpt.pbo > 0.5:
            reasons.append(
                f"PBO {rpt.pbo:.1%} > 50%")
        if rpt.avg_max_dd > 0.25:
            reasons.append(
                f"Avg DD {rpt.avg_max_dd:.1%} > 25%")
        if not reasons:
            rpt.classification = "PASS"
        else:
            rpt.classification = "FAIL"
            rpt.reasons = reasons
