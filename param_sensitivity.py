"""Parameter Sensitivity — detects overfitting by perturbing strategy parameters."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ParamResult:
    param_name: str
    original_value: float
    values_tested: list = field(default_factory=list)
    pfs: list = field(default_factory=list)
    min_pf: float = 0.0
    max_pf: float = 0.0
    std_pf: float = 0.0
    degradation_ratio: float = 0.0
    sensitivity_range: float = 0.0


@dataclass
class ParamSensitivityReport:
    original_pf: float = 0.0
    param_results: list = field(default_factory=list)
    n_params: int = 0
    n_robust: int = 0
    worst_param: str = ""
    worst_degradation: float = 1.0
    classification: str = "UNKNOWN"
    reasons: list = field(default_factory=list)


class ParamSensitivity:
    """One-at-a-time parameter perturbation to detect overfitting."""

    def __init__(self, make_backtest_fn, param_specs,
                 n_perturbations=7, min_degradation=0.7):
        self.make_backtest_fn = make_backtest_fn
        self.param_specs = param_specs
        self.n_perturbations = n_perturbations
        self.min_degradation = min_degradation

    @staticmethod
    def _calc_pf(trades: list) -> float:
        arr = np.array(trades, dtype=np.float64)
        wins = arr[arr > 0].sum() if (arr > 0).any() else 0.0
        losses = abs(arr[arr < 0].sum()) if (arr < 0).any() else 1e-10
        return float(wins / losses)

    def run(self, data, train_idx=None, test_idx=None) -> ParamSensitivityReport:
        if train_idx is None:
            train_idx = list(range(int(len(data) * 0.7)))
        if test_idx is None:
            test_idx = list(range(train_idx[-1] + 1, len(data)))
        if not train_idx or not test_idx:
            return ParamSensitivityReport(
                classification="FAIL",
                reasons=["Empty train or test indices"])

        base_params = {k: v[0] for k, v in self.param_specs.items()}
        base_bt = self.make_backtest_fn(base_params)
        base_result = base_bt(data, train_idx, test_idx)
        base_pf = self._calc_pf(base_result.get("trades", []))
        if base_pf <= 0:
            return ParamSensitivityReport(
                classification="FAIL",
                reasons=[f"Base PF {base_pf:.3f} is not positive"])

        param_results = []
        n_robust = 0
        worst_name = ""
        worst_deg = 1.0

        for pname, (center, delta) in self.param_specs.items():
            values = np.linspace(center - delta, center + delta,
                                 self.n_perturbations).tolist()
            pfs = []
            for val in values:
                params = base_params.copy()
                params[pname] = val
                bt = self.make_backtest_fn(params)
                result = bt(data, train_idx, test_idx)
                pfs.append(self._calc_pf(result.get("trades", [])))

            pfs_arr = np.array(pfs)
            min_pf = float(pfs_arr.min())
            max_pf = float(pfs_arr.max())
            std_pf = float(pfs_arr.std())
            deg = min_pf / base_pf if base_pf > 0 else 0.0
            sens = (max_pf - min_pf) / base_pf if base_pf > 0 else 0.0

            if deg < worst_deg:
                worst_deg = deg
                worst_name = pname
            if deg >= self.min_degradation:
                n_robust += 1

            param_results.append(ParamResult(
                param_name=pname, original_value=center,
                values_tested=values, pfs=[round(p, 4) for p in pfs],
                min_pf=min_pf, max_pf=max_pf, std_pf=std_pf,
                degradation_ratio=deg, sensitivity_range=sens,
            ))

        reasons = []
        for pr in param_results:
            if pr.degradation_ratio < self.min_degradation:
                reasons.append(
                    f"{pr.param_name}: degradation {pr.degradation_ratio:.2%} "
                    f"< {self.min_degradation:.0%} (PF {pr.min_pf:.3f})")

        cls = "PASS" if not reasons else "FAIL"
        return ParamSensitivityReport(
            original_pf=base_pf, param_results=param_results,
            n_params=len(self.param_specs), n_robust=n_robust,
            worst_param=worst_name, worst_degradation=worst_deg,
            classification=cls, reasons=reasons,
        )

