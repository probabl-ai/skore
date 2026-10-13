"""Check whether the model is slower than a fast baseline (SKD010)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    baseline_estimator_report,
    cast_report,
    check_score_better_than_baseline,
    collect_scores,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.cross_validation.report import CrossValidationReport
    from skore._reports.estimator.report import EstimatorReport


def get_fit_time(report: EstimatorReport | CrossValidationReport) -> float:
    if report._report_type == "cross-validation":
        return float(report.metrics.timings(aggregate="mean").loc["Fit time (s)"])
    if report._fit_time is None:
        raise CheckNotApplicable("Fit time is unavailable.")
    return report._fit_time


def get_predict_time(report: EstimatorReport | CrossValidationReport) -> float:
    if report._report_type == "cross-validation":
        return float(
            report.metrics.predict_time(aggregate="mean").loc["Predict time (s)"]
        )
    return cast(float, report.metrics.predict_time(data_source="test"))


class CheckSlowerThanBaseline(Check):
    """Check whether the model is slower than a fast baseline (SKD010).

    Compares fit and predict time, and test-set scores, against a
    :func:`skrub.tabular_pipeline`-wrapped fast linear baseline
    (:class:`~sklearn.linear_model.RidgeCV` for regression,
    :class:`~sklearn.linear_model.LogisticRegression` for classification).

    The slowness gate uses whichever of the fit-time or predict-time ratios is
    larger, and triggers when that ratio is at least ``2x`` the baseline's,
    with an absolute gap of at least ``1s`` on the winning dimension: below
    that, the difference is negligible in practice regardless of the ratio.
    """

    code = "SKD010"
    title = "Model slower than baseline"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd010-slower-than-baseline"
    severity = "issue"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        baseline = baseline_estimator_report(report, kind="fast")

        report_test = collect_scores(report, data_source="test")
        baseline_test = collect_scores(baseline, data_source="test")

        report_fit_time = get_fit_time(report)
        baseline_fit_time = get_fit_time(baseline)
        report_predict_time = get_predict_time(report)
        baseline_predict_time = get_predict_time(baseline)

        fit_ratio = report_fit_time / baseline_fit_time
        predict_ratio = report_predict_time / baseline_predict_time

        if fit_ratio >= predict_ratio:
            slowness_ratio = fit_ratio
            dimension = "Fit time"
            gap = report_fit_time - baseline_fit_time
        else:
            slowness_ratio = predict_ratio
            dimension = "Predict time"
            gap = report_predict_time - baseline_predict_time

        if slowness_ratio < 2.0 or gap < 1.0:
            return None

        votes = [
            not check_score_better_than_baseline(
                score=report_test[key]["score"],
                baseline=baseline_test[key]["score"],
                greater_is_better=baseline_test[key]["greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            for key in report_test.keys() & baseline_test.keys()
        ]
        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                f"{dimension} is ~{slowness_ratio:.1f}x slower than a fast linear"
                " baseline without significantly better test scores"
                f" ({n_positive}/{total} default predictive metrics)."
            )
        return None
