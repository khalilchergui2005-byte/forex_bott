"""Portfolio Allocator — correlation, exposure control, risk parity."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


def compute_correlation_matrix(
    data: dict[str, list[float]],
    window: int = 60,
    input_type: str = "equity",
) -> pd.DataFrame:
    """Rolling Pearson correlation between strategy returns.

    Args:
        data: {strategy_name: values}
            If input_type="equity": cumulative equity curve.
            If input_type="returns": per-trade P&L.
        window: number of recent values to use.
        input_type: "equity" or "returns".
    Returns:
        NxN DataFrame with correlation values (-1 to 1).
        Diagonal = 1.0, insufficient data = NaN.
    """
    names = list(data.keys())
    n = len(names)
    if n < 2:
        return pd.DataFrame(1.0, index=names, columns=names)
    rets = {}
    for name in names:
        arr = np.array(data[name], dtype=float)
        if input_type == "equity" and len(arr) > 1:
            arr = np.diff(arr)
        arr = arr[-window:]
        rets[name] = arr if len(arr) >= 3 else None
    mat = pd.DataFrame(np.nan, index=names, columns=names)
    for i in range(n):
        mat.iloc[i, i] = 1.0
        for j in range(i + 1, n):
            ri, rj = rets[names[i]], rets[names[j]]
            if ri is None or rj is None:
                continue
            ml = min(len(ri), len(rj))
            c = float(np.corrcoef(ri[-ml:], rj[-ml:])[0, 1])
            mat.iloc[i, j] = c
            mat.iloc[j, i] = c
    return mat


def avg_correlation(corr_matrix: pd.DataFrame,
                    strategy_name: str) -> float:
    """Average correlation of one strategy with all others."""
    if strategy_name not in corr_matrix.columns:
        return 0.0
    vals = corr_matrix[strategy_name].drop(strategy_name, errors="ignore")
    vals = vals.dropna()
    return float(vals.mean()) if len(vals) > 0 else 0.0

@dataclass
class ExposureReport:
    aggregate_exposure: float
    long_exposure: float
    short_exposure: float
    scaling_factor: float
    should_reduce: bool
    reason: str


class ExposureController:

    def __init__(self, max_direction_exposure=0.7):
        self.max_exp = max_direction_exposure

    def check(self, positions):
        if not positions:
            return ExposureReport(0.0, 0.0, 0.0, 1.0, False, "No positions")
        long_sum = sum(p["size"] for p in positions
                       if p.get("side") == "long")
        short_sum = sum(p["size"] for p in positions
                        if p.get("side") == "short")
        total = long_sum + short_sum
        if total < 1e-10:
            return ExposureReport(0.0, 0.0, 0.0, 1.0, False, "Zero total")
        agg = (long_sum - short_sum) / total
        aa = abs(agg)
        if aa > self.max_exp:
            sc = self.max_exp / aa
            return ExposureReport(
                aggregate_exposure=agg, long_exposure=long_sum,
                short_exposure=short_sum, scaling_factor=sc,
                should_reduce=True,
                reason=f"|exp| {aa:.2f} > {self.max_exp}")
        return ExposureReport(
            aggregate_exposure=agg, long_exposure=long_sum,
            short_exposure=short_sum, scaling_factor=1.0,
            should_reduce=False, reason="Normal")
