from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._sklearn._diagnostic.base import Check
from skore._sklearn._diagnostic.utils import (
    _TIMING_METRICS,
    DiagnosticNotApplicable,
    check_score_gap_to_baseline,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport


def _get_metrics_data(report: _BaseReport) -> tuple:
    """Compute report/baseline metrics data for SKD001 and SKD002.

    Raises :class:`DiagnosticNotApplicable` when train+test data is
    unavailable.
    """
    # Avoid circular import
    from skore._sklearn._estimator.report import EstimatorReport

    if (
        not isinstance(report, EstimatorReport)
        or report.X_train is None
        or report.y_train is None
        or report.X_test is None
        or report.y_test is None
    ):
        raise DiagnosticNotApplicable()

    baseline_report = EstimatorReport(
        DummyClassifier(strategy="prior")
        if "classification" in report.ml_task
        else DummyRegressor(strategy="mean"),
        X_train=report.X_train,
        y_train=report.y_train,
        X_test=cast(ArrayLike, report.X_test),
        y_test=report.y_test,
        pos_label=report.pos_label,
    )
    # baseline_report._metric_registry = report._metric_registry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        report_data = report.metrics.summarize(data_source="train").data.rename(
            columns={"score": "score_train"}
        )
        report_data["score_test"] = report.metrics.summarize(data_source="test").data[
            "score"
        ]

        baseline_data = baseline_report.metrics.summarize(
            data_source="train"
        ).data.rename(columns={"score": "score_train"})
        baseline_data["score_test"] = baseline_report.metrics.summarize(
            data_source="test"
        ).data["score"]

    return report_data, baseline_data


class CheckOverfitting(Check):
    code = "SKD001"
    title = "Potential overfitting"
    report_type = "estimator"
    docs_url = "skd001-overfitting"

    def check_function(self, report: _BaseReport) -> str | None:
        """Check for overfitting (SKD001).

        Detects significant gaps between train and test scores.
        Raises :class:`DiagnosticNotApplicable` when train+test data is
        unavailable.
        """
        report_data, _baseline_data = _get_metrics_data(report)

        votes = [
            check_score_gap_to_baseline(
                score=report_data.loc[idx, "score_train"],
                baseline=report_data.loc[idx, "score_test"],
                favorability=report_data.loc[idx, "favorability"],
                floor=0.03,
                fraction=0.10,
            )
            for idx in range(len(report_data))
            if report_data.loc[idx, "metric"] not in _TIMING_METRICS
        ]

        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                "Significant train/test gaps were found for "
                f"{n_positive}/{total} default predictive metrics."
            )
        return None


class CheckUnderfitting(Check):
    code = "SKD002"
    title = "Potential underfitting"
    report_type = "estimator"
    docs_url = "skd002-underfitting"

    def check_function(self, report: _BaseReport) -> str | None:
        """Check for underfitting (SKD002).

        Detects train and test scores close to a dummy baseline.
        Raises :class:`DiagnosticNotApplicable` when train+test data is
        unavailable.
        """
        report_data, baseline_data = _get_metrics_data(report)

        votes = [
            not check_score_gap_to_baseline(
                score=report_data.loc[idx, "score_train"],
                baseline=baseline_data.loc[idx, "score_train"],
                favorability=baseline_data.loc[idx, "favorability"],
                floor=0.01,
                fraction=0.05,
            )
            and not check_score_gap_to_baseline(
                score=report_data.loc[idx, "score_test"],
                baseline=baseline_data.loc[idx, "score_test"],
                favorability=baseline_data.loc[idx, "favorability"],
                floor=0.01,
                fraction=0.05,
            )
            for idx in range(len(report_data))
            if report_data.loc[idx, "metric"] not in _TIMING_METRICS
        ]
        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                "Train/test scores are on par and not significantly better "
                f"than the dummy baseline for {n_positive}/{total} "
                "comparable metrics."
            )
        return None


_BUILTIN_CHECKS = [CheckOverfitting(), CheckUnderfitting()]
