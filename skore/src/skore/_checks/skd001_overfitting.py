from __future__ import annotations

from typing import TYPE_CHECKING

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    cast_report,
    check_score_better_than_baseline,
    collect_scores,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckOverfitting(Check):
    """Check for overfitting (SKD001).

    Detects significant gaps between train and test scores.
    Raises :class:`CheckNotApplicable` when train+test data is
    unavailable.
    """

    code = "SKD001"
    title = "Potential overfitting"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd001-overfitting"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect significant gaps between train and test scores."""
        report = cast_report(report)
        if report._report_type == "estimator" and (
            report.X_train is None or report.y_train is None
        ):
            raise CheckNotApplicable("Train data is unavailable.")

        report_train = collect_scores(report, data_source="train")
        report_test = collect_scores(report, data_source="test")

        votes = [
            check_score_better_than_baseline(
                score=report_train[key]["score"],
                baseline=report_test[key]["score"],
                greater_is_better=report_train[key]["greater_is_better"],
                floor=0.03,
                fraction=0.10,
            )
            for key in report_train.keys() & report_test.keys()
        ]

        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                "Significant train/test gaps were found for "
                f"{n_positive}/{total} default predictive metrics."
            )
        return None
