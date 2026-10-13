from __future__ import annotations

from typing import TYPE_CHECKING

from skore._checks.base import Check
from skore._checks.utils import (
    baseline_estimator_report,
    cast_report,
    check_score_better_than_baseline,
    collect_scores,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckUnderfitting(Check):
    """Check for underfitting (SKD002).

    Detects train and test scores close to a dummy baseline.
    Raises :class:`CheckNotApplicable` when train+test data is
    unavailable.
    """

    code = "SKD002"
    title = "Potential underfitting"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd002-underfitting"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect train and test scores close to a dummy baseline."""
        report = cast_report(report)
        baseline = baseline_estimator_report(report, kind="dummy")

        report_train = collect_scores(report, data_source="train")
        report_test = collect_scores(report, data_source="test")
        baseline_train = collect_scores(baseline, data_source="train")
        baseline_test = collect_scores(baseline, data_source="test")

        votes = [
            not check_score_better_than_baseline(
                score=report_train[key]["score"],
                baseline=baseline_train[key]["score"],
                greater_is_better=baseline_train[key]["greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            and not check_score_better_than_baseline(
                score=report_test[key]["score"],
                baseline=baseline_test[key]["score"],
                greater_is_better=baseline_test[key]["greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            for key in (
                report_train.keys()
                & report_test.keys()
                & baseline_train.keys()
                & baseline_test.keys()
            )
        ]

        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                "Train/test scores are on par and not significantly better "
                f"than the dummy baseline for {n_positive}/{total} "
                "comparable metrics."
            )
        return None
