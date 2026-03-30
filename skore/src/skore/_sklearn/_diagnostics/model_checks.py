from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

from numpy.typing import ArrayLike
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._sklearn._diagnostics.base import DiagnosticResult
from skore._sklearn._diagnostics.utils import (
    _TIMING_METRICS,
    DiagnosticNotApplicable,
    check_score_gap_to_baseline,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport


def check_overfitting_underfitting(
    report: EstimatorReport,
) -> list[DiagnosticResult]:
    """Check for overfitting (SKD001) and underfitting (SKD002).

    Both checks share the same pre-conditions and metric data, so they are
    grouped in a single function.  Raises :class:`DiagnosticNotApplicable`
    when train+test data is unavailable.
    """
    if (
        report.X_train is None
        or report.y_train is None
        or report.X_test is None
        or report.y_test is None
    ):
        raise DiagnosticNotApplicable()
    # Avoid circular import
    from skore._sklearn._estimator.report import EstimatorReport

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

    results: list[DiagnosticResult] = []
    # SKD001 - Overfitting
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
        results.append(
            DiagnosticResult(
                code="SKD001",
                title="Potential overfitting",
                docs_anchor="skd001-overfitting",
                explanation=(
                    "Significant train/test gaps were found for "
                    f"{n_positive}/{total} default predictive metrics."
                ),
            )
        )

    # SKD002 - Underfitting
    # train and test scores are close to a dummy baseline.
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
        results.append(
            DiagnosticResult(
                code="SKD002",
                title="Potential underfitting",
                docs_anchor="skd002-underfitting",
                explanation=(
                    "Train/test scores are on par and not significantly better "
                    f"than the dummy baseline for {n_positive}/{total} "
                    "comparable metrics."
                ),
            )
        )

    return results
