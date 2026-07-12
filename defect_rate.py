"""Defect Rate Calculator — aggregates validation test results."""

from dataclasses import dataclass, field


@dataclass
class DefectReport:
    total_tests: int = 0
    n_passed: int = 0
    n_failed: int = 0
    defect_rate: float = 0.0
    details: list = field(default_factory=list)
    classification: str = "UNKNOWN"
    reasons: list = field(default_factory=list)


class DefectRateCalculator:
    """Counts PASS/FAIL across all validation tests."""

    def __init__(self, max_defect_rate=0.5):
        self.max_defect_rate = max_defect_rate

    def run(self, test_results: list) -> DefectReport:
        """
        test_results: list of (name, classification, reasons)
        Example: [("CPCV", "PASS", []), ("MC", "FAIL", ["p > 0.05"])]
        """
        total = len(test_results)
        if total == 0:
            return DefectReport(
                classification="FAIL",
                reasons=["No test results provided"])

        n_pass = sum(1 for _, cls, _ in test_results if cls == "PASS")
        n_fail = total - n_pass
        rate = n_fail / total

        failed_names = [
            f"{name}: {'; '.join(reasons)}" if reasons else name
            for name, cls, reasons in test_results
            if cls == "FAIL"
        ]

        reasons = []
        if rate > self.max_defect_rate:
            reasons.append(
                f"Defect rate {rate:.0%} > {self.max_defect_rate:.0%}")

        cls = "PASS" if not reasons else "FAIL"
        return DefectReport(
            total_tests=total, n_passed=n_pass, n_failed=n_fail,
            defect_rate=rate, details=test_results,
            classification=cls, reasons=reasons,
        )
