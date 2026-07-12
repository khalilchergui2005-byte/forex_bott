"""Monte Carlo Bootstrap Test — resamples trades to test robustness."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class MCReport:
    n_trades: int = 0
    n_sims: int = 0
    confidence: float = 0.95
    real_pf: float = 0.0
    real_total_pnl: float = 0.0
    real_max_dd_pct: float = 0.0
    real_sharpe: float = 0.0
    sim_pf_mean: float = 0.0
    sim_pf_std: float = 0.0
    sim_pf_p5: float = 0.0
    sim_pf_p95: float = 0.0
    sim_pnl_mean: float = 0.0
    sim_pnl_std: float = 0.0
    sim_pnl_p5: float = 0.0
    sim_pnl_p95: float = 0.0
    sim_dd_p95: float = 0.0
    p_value_pf: float = 1.0
    p_value_pnl: float = 1.0
    classification: str = "UNKNOWN"
    reasons: list = field(default_factory=list)


class MonteCarloTest:
    """Bootstrap resample trades, test if profitability survives resampling."""

    def __init__(self, n_sims=1000, confidence=0.95, seed=None):
        self.n_sims = n_sims
        self.confidence = confidence
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _metrics(trades: np.ndarray) -> dict:
        wins = trades[trades > 0].sum() if (trades > 0).any() else 0.0
        losses = abs(trades[trades < 0].sum()) if (trades < 0).any() else 1e-10
        pf = wins / losses
        pnl = trades.sum()
        eq = np.concatenate([[0.0], np.cumsum(trades)])
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, 1e-10)
        max_dd = dd.max() if len(dd) > 0 else 0.0
        sharpe = (trades.mean() / trades.std() if trades.std() > 0 else 0.0)
        return {"pf": pf, "pnl": pnl, "max_dd": max_dd, "sharpe": sharpe}

    def run(self, trades: list) -> MCReport:
        arr = np.array(trades, dtype=np.float64)
        n = len(arr)
        r = MCReport(n_trades=n, n_sims=self.n_sims, confidence=self.confidence)
        if n < 10:
            r.classification = "FAIL"
            r.reasons = [f"Insufficient trades: {n} < 10"]
            return r

        real = self._metrics(arr)
        sim_pf = np.empty(self.n_sims)
        sim_pnl = np.empty(self.n_sims)
        sim_dd = np.empty(self.n_sims)

        for i in range(self.n_sims):
            boot = self.rng.choice(arr, size=n, replace=True)
            m = self._metrics(boot)
            sim_pf[i] = m["pf"]
            sim_pnl[i] = m["pnl"]
            sim_dd[i] = m["max_dd"]

        alpha = 1.0 - self.confidence
        p_pf = float(np.mean(sim_pf < 1.0))
        p_pnl = float(np.mean(sim_pnl < 0.0))

        r.real_pf = real["pf"]
        r.real_total_pnl = real["pnl"]
        r.real_max_dd_pct = real["max_dd"]
        r.real_sharpe = real["sharpe"]
        r.sim_pf_mean = float(sim_pf.mean())
        r.sim_pf_std = float(sim_pf.std())
        r.sim_pf_p5 = float(np.percentile(sim_pf, 5))
        r.sim_pf_p95 = float(np.percentile(sim_pf, 95))
        r.sim_pnl_mean = float(sim_pnl.mean())
        r.sim_pnl_std = float(sim_pnl.std())
        r.sim_pnl_p5 = float(np.percentile(sim_pnl, 5))
        r.sim_pnl_p95 = float(np.percentile(sim_pnl, 95))
        r.sim_dd_p95 = float(np.percentile(sim_dd, 95))
        r.p_value_pf = p_pf
        r.p_value_pnl = p_pnl

        reasons = []
        if p_pf > alpha:
            reasons.append(f"PF fail-rate {p_pf:.2%} > {alpha:.0%}")
        if p_pnl > alpha:
            reasons.append(f"PnL fail-rate {p_pnl:.2%} > {alpha:.0%}")
        if not reasons:
            r.classification = "PASS"
        else:
            r.classification = "FAIL"
            r.reasons = reasons
        return r
