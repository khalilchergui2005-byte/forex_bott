"""Composite Score — weighted aggregation of all validation tests."""

from dataclasses import dataclass, field

WEIGHTS = {
    "CPCV": 25,
    "WalkForward": 20,
    "MonteCarlo": 15,
    "StressTest": 15,
    "MultiPeriod": 10,
    "ParamSensitivity": 10,
    "DefectRate": 5,
}

GRADES = [(90, "A"), (75, "B"), (60, "C"), (40, "D"), (0, "F")]


@dataclass
class TestScore:
    name: str
    classification: str
    weight: int
    earned: int


@dataclass
class CompositeReport:
    total_tests: int = 0
    n_passed: int = 0
    n_failed: int = 0
    composite_score: float = 0.0
    grade: str = "F"
    test_scores: list = field(default_factory=list)
    classification: str = "UNKNOWN"
    reasons: list = field(default_factory=list)


class CompositeScorer:
    """Weighted scoring across all validation tests."""

    def __init__(self, weights=None, min_score=60.0):
        self.weights = weights or WEIGHTS.copy()
        self.min_score = min_score

    @staticmethod
    def _grade(score: float) -> str:
        for threshold, letter in GRADES:
            if score >= threshold:
                return letter
        return "F"

    def run(self, test_results: list) -> CompositeReport:
        """
        test_results: list of (name, classification, reasons)
        """
        total = len(test_results)
        if total == 0:
            return CompositeReport(
                classification="FAIL", reasons=["No results"])

        scores = []
        n_pass = 0
        total_w = 0
        earned_w = 0

        for name, cls, _ in test_results:
            w = self.weights.get(name, 0)
            e = w if cls == "PASS" else 0
            total_w += w
            earned_w += e
            if cls == "PASS":
                n_pass += 1
            scores.append(TestScore(
                name=name, classification=cls, weight=w, earned=e))

        score = (earned_w / total_w * 100) if total_w > 0 else 0.0
        grade = self._grade(score)
        reasons = []
        if score < self.min_score:
            reasons.append(f"Score {score:.0f} < {self.min_score:.0f}")

        return CompositeReport(
            total_tests=total, n_passed=n_pass,
            n_failed=total - n_pass,
            composite_score=score, grade=grade,
            test_scores=scores,
            classification="PASS" if not reasons else "FAIL",
            reasons=reasons,
        )
