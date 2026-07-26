"""
Walk-Forward Validation Engine.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Callable

@dataclass
class WFWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    n_trades: int
    profit_factor: float
    win_rate: float
    expectancy: float
    total_pnl: float
    max_drawdown_pct: float
    sharpe: float
    is_pf: float = 0.0
    is_n_trades: int = 0

    @property
    def degradation(self) -> float:
        """IS to OOS PF degradation ratio."""
        if self.is_pf <= 0:
            return 1.0
        return max(0, 1.0 - self.profit_factor / self.is_pf)

@dataclass
class WFReport:
    """Walk-forward validation report."""
    strategy_name: str
    n_windows: int
    n_profitable: int
    avg_oos_pf: float
    oos_pf_std: float
    oos_pf_min: float
    oos_pf_max: float
    avg_oos_wr: float
    avg_oos_expectancy: float
    avg_oos_dd: float
    avg_oos_sharpe: float
    avg_oos_trades: float
    avg_is_pf: float
    avg_degradation: float
    classification: str = "PENDING"
    reasons: List[str] = field(
        default_factory=list)
    windows: List[WFWindow] = field(
        default_factory=list)

    def summary(self):
        """Formatted walk-forward report."""
        sep = "=" * 50
        lines = [
            sep,
            f"Walk-Forward: {self.strategy_name}",
            sep,
            f"Windows: {self.n_windows}",
            f"Profitable: {self.n_profitable}/{self.n_windows}",
            "",
            "OOS Performance:",
            f"  PF:     {self.avg_oos_pf:.3f} +/- {self.oos_pf_std:.3f}",
            f"  Range:  [{self.oos_pf_min:.3f}, {self.oos_pf_max:.3f}]",
            f"  WR:     {self.avg_oos_wr:.1%}",
            f"  Expect: {self.avg_oos_expectancy:.4f}",
            f"  DD:     {self.avg_oos_dd:.1%}",
            f"  Sharpe: {self.avg_oos_sharpe:.3f}",
            f"  Trades: {self.avg_oos_trades:.0f}",
            "",
            "IS vs OOS:",
            f"  Avg IS PF:       {self.avg_is_pf:.3f}",
            f"  Avg OOS PF:      {self.avg_oos_pf:.3f}",
            f"  Avg Degradation: {self.avg_degradation:.1%}",
            "",
            f"Result: {self.classification}",
        ]
        if self.reasons:
            for r in self.reasons:
                lines.append(f"  - {r}")
        lines.append(sep)
        return "\n".join(lines)

class WalkForwardValidator:
    """
    Walk-Forward validation for financial strategies.
    backtest_fn(data, train_idx, test_idx) -> dict
    Must return: {trades: list, equity: list}
    """

    def __init__(self, backtest_fn, train_size=4000,
                 test_size=1000, step=1000,
                 purge_bars=24):
        self.backtest_fn = backtest_fn
        self.train_size = train_size
        self.test_size = test_size
        self.step = step
        self.purge_bars = purge_bars

    def validate(self, data, strategy_name="unnamed",
                 min_trades=20, run_is=True):
        """Run walk-forward validation."""
        n_bars = len(data)
        windows = self._generate_windows(n_bars)
        results = []
        for w in windows:
            r = self._run_window(
                data, w, min_trades, run_is)
            if r is not None:
                results.append(r)
        return self._build_report(
            strategy_name, results)

    def _generate_windows(self, n_bars):
        """Generate rolling walk-forward windows."""
        windows = []
        w_id = 0
        start = 0
        while True:
            tr_s = start
            tr_e = start + self.train_size
            te_s = tr_e + self.purge_bars
            te_e = te_s + self.test_size
            if te_e > n_bars:
                break
            windows.append({
                "id": w_id,
                "train_start": tr_s,
                "train_end": tr_e,
                "test_start": te_s,
                "test_end": te_e})
            w_id += 1
            start += self.step
        return windows

    def _run_window(self, data, window, min_trades,
                       run_is):
        """Run backtest on one window."""
        try:
            tr_idx = np.arange(
                window["train_start"],
                window["train_end"])
            te_idx = np.arange(
                window["test_start"],
                window["test_end"])
            bt = self.backtest_fn(data, tr_idx, te_idx)
            trades = bt.get("trades", [])
            equity = bt.get("equity", [])
            if len(trades) < min_trades:
                return None
            oos = self._compute_metrics(trades, equity)
            is_pf = 0.0
            is_n = 0
            if run_is:
                try:
                    is_bt = self.backtest_fn(
                        data, tr_idx, tr_idx)
                    is_tr = is_bt.get("trades", [])
                    is_n = len(is_tr)
                    if is_n > 0:
                        iw = [t for t in is_tr if t > 0]
                        il = [t for t in is_tr if t < 0]
                        gp = sum(iw) if iw else 0.0
                        gl = abs(sum(il)) if il else 1e-10
                        is_pf = gp / gl
                except Exception:
                    pass
            return WFWindow(
                window_id=window["id"],
                train_start=window["train_start"],
                train_end=window["train_end"],
                test_start=window["test_start"],
                test_end=window["test_end"],
                n_train=window["train_end"] - window["train_start"],
                n_test=window["test_end"] - window["test_start"],
                n_trades=len(trades),
                profit_factor=oos["pf"],
                win_rate=oos["wr"],
                expectancy=oos["exp"],
                total_pnl=oos["pnl"],
                max_drawdown_pct=oos["mdd"],
                sharpe=oos["sh"],
                is_pf=is_pf,
                is_n_trades=is_n)
        except Exception:
            return None

    def _compute_metrics(self, trades, equity):
        """Compute PF, WR, DD, Sharpe from trades."""
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
        return {
            "pf": pf, "wr": wr, "exp": exp,
            "pnl": t.sum(), "mdd": mdd, "sh": sh}

    def _build_report(self, name, results):
        """Build final walk-forward report."""
        if not results:
            return WFReport(
                strategy_name=name, n_windows=0,
                n_profitable=0,
                avg_oos_pf=0, oos_pf_std=0,
                oos_pf_min=0, oos_pf_max=0,
                avg_oos_wr=0, avg_oos_expectancy=0,
                avg_oos_dd=0, avg_oos_sharpe=0,
                avg_oos_trades=0,
                avg_is_pf=0, avg_degradation=0,
                classification="FAIL",
                reasons=["No valid windows"])
        pfs = [r.profit_factor for r in results]
        np_ = sum(1 for p in pfs if p >= 1.0)
        degrads = [r.degradation for r in results]
        rpt = WFReport(
            strategy_name=name,
            n_windows=len(results),
            n_profitable=np_,
            avg_oos_pf=float(np.mean(pfs)),
            oos_pf_std=float(np.std(pfs)),
            oos_pf_min=float(np.min(pfs)),
            oos_pf_max=float(np.max(pfs)),
            avg_oos_wr=float(np.mean(
                [r.win_rate for r in results])),
            avg_oos_expectancy=float(np.mean(
                [r.expectancy for r in results])),
            avg_oos_dd=float(np.mean(
                [r.max_drawdown_pct for r in results])),
            avg_oos_sharpe=float(np.mean(
                [r.sharpe for r in results])),
            avg_oos_trades=float(np.mean(
                [r.n_trades for r in results])),
            avg_is_pf=float(np.mean(
                [r.is_pf for r in results])),
            avg_degradation=float(np.mean(degrads)),
            windows=results)
        self._classify(rpt)
        return rpt

    def _classify(self, rpt):
        """Classify walk-forward results."""
        reasons = []
        if rpt.n_profitable == 0:
            reasons.append(
                "No profitable OOS windows")
        if rpt.avg_oos_pf < 1.0:
            reasons.append(
                f"OOS PF {rpt.avg_oos_pf:.3f} < 1.0")
        if rpt.oos_pf_std > 0.5:
            reasons.append(
                f"OOS PF std {rpt.oos_pf_std:.3f} > 0.5")
        if rpt.avg_degradation > 0.6:
            reasons.append(
                f"Degradation {rpt.avg_degradation:.1%} > 60%")
        if rpt.avg_oos_dd > 0.20:
            reasons.append(
                f"Avg DD {rpt.avg_oos_dd:.1%} > 20%")
        if rpt.avg_oos_trades < 20:
            reasons.append(
                f"Avg trades {rpt.avg_oos_trades:.0f} < 20")
        if not reasons:
            rpt.classification = "PASS"
        else:
            rpt.classification = "FAIL"
            rpt.reasons = reasons
