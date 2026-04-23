from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._sklearn._diagnostic.base import Check
from skore._sklearn._diagnostic.utils import (
    _TIMING_METRICS,
    CheckNotApplicable,
    check_score_gap_to_baseline,
    detect_outliers_modified_zscore,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport


def _get_metrics_data(report: _BaseReport) -> tuple:
    """Compute report/baseline metrics data for SKD001 and SKD002.

    Raises :class:`CheckNotApplicable` when train+test data is
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
        raise CheckNotApplicable()

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
    baseline_report._metric_registry = report._metric_registry
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
    """Check for overfitting (SKD001).

    Detects significant gaps between train and test scores.
    Raises :class:`CheckNotApplicable` when train+test data is
    unavailable.
    """

    code = "SKD001"
    title = "Potential overfitting"
    report_type = "estimator"
    docs_url = "skd001-overfitting"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        report_data, _baseline_data = _get_metrics_data(report)

        votes = [
            check_score_gap_to_baseline(
                score=report_data.loc[idx, "score_train"],
                baseline=report_data.loc[idx, "score_test"],
                greater_is_better=report_data.loc[idx, "greater_is_better"],
                floor=0.03,
                fraction=0.10,
            )
            for idx in range(len(report_data))
            if report_data.loc[idx, "metric_verbose_name"] not in _TIMING_METRICS
        ]

        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                "Significant train/test gaps were found for "
                f"{n_positive}/{total} default predictive metrics."
            )
        return None


class CheckUnderfitting(Check):
    """Check for underfitting (SKD002).

    Detects train and test scores close to a dummy baseline.
    Raises :class:`CheckNotApplicable` when train+test data is
    unavailable.
    """

    code = "SKD002"
    title = "Potential underfitting"
    report_type = "estimator"
    docs_url = "skd002-underfitting"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        report_data, baseline_data = _get_metrics_data(report)

        votes = [
            not check_score_gap_to_baseline(
                score=report_data.loc[idx, "score_train"],
                baseline=baseline_data.loc[idx, "score_train"],
                greater_is_better=baseline_data.loc[idx, "greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            and not check_score_gap_to_baseline(
                score=report_data.loc[idx, "score_test"],
                baseline=baseline_data.loc[idx, "score_test"],
                greater_is_better=baseline_data.loc[idx, "greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            for idx in range(len(report_data))
            if report_data.loc[idx, "metric_verbose_name"] not in _TIMING_METRICS
        ]
        majority, n_positive, total = majority_vote(votes)
        if majority:
            return (
                "Train/test scores are on par and not significantly better "
                f"than the dummy baseline for {n_positive}/{total} "
                "comparable metrics."
            )
        return None


class CheckMetricsConsistencyAcrossSplits(Check):
    """Check the consistency of metrics across splits (SKD003).

    Outlier splits are identified with a modified Z-score based on the
    Median Absolute Deviation (MAD) to be robust to extreme values.
    """

    code = "SKD003"
    title = "Inconsistent performance across splits"
    report_type = "cross-validation"
    docs_url = "skd003-inconsistent_performance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        from skore._sklearn._cross_validation.report import CrossValidationReport

        if not isinstance(report, CrossValidationReport):
            raise CheckNotApplicable()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            report_data = report.metrics.summarize(data_source="test").frame(
                aggregate=None, flat_index=True
            )
        votes = np.array(
            [
                detect_outliers_modified_zscore(report_data.loc[idx])
                for idx in report_data.index
                if idx not in _TIMING_METRICS
            ]
        )
        explanation = []
        for cv in range(report_data.shape[1]):
            majority, n_positive, total = majority_vote(votes[:, cv].tolist())
            if majority:
                explanation.append(f"in split #{cv} for {n_positive}/{total} metrics")
        if explanation:
            return "Performance is abnormal " + " and ".join(explanation) + "."
        return None


class CheckHighClassImbalance(Check):
    """Check for high class imbalance (SKD004) in binary classification.

    Detects an issue when the most frequent class represents more than 80% of the
    dataset.
    """

    code = "SKD004"
    title = "High class imbalance"
    report_type = "estimator"
    docs_url = "skd004-high_class_imbalance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        from skore._sklearn._estimator.report import EstimatorReport

        if (
            not isinstance(report, EstimatorReport)
            or report.ml_task != "binary-classification"
            or report.y_train is None
            or report.y_test is None
        ):
            raise CheckNotApplicable()

        values, counts = np.unique_counts(
            np.concatenate([report.y_train, report.y_test])
        )

        overrepresented_class = values[counts >= 0.8 * counts.sum()]

        if overrepresented_class.size > 0:
            return (
                f"Class {overrepresented_class} represents more than 80% of the "
                "dataset samples. Accuracy should not be used alone to assess model "
                "performance as it may be misleading by ignoring poor performance on "
                "the underrepresented class."
            )
        return None


class CheckUnderrepresentedClasses(Check):
    """Check for underrepresented classes (SKD005) in multiclass classification.

    Detects an issue when some classes represent less than 10% of the dataset.
    """

    code = "SKD005"
    title = "Underrepresented classes"
    report_type = "estimator"
    docs_url = "skd005-underrepresented_classes"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        from skore._sklearn._estimator.report import EstimatorReport

        if (
            not isinstance(report, EstimatorReport)
            or report.ml_task != "multiclass-classification"
            or report.y_train is None
            or report.y_test is None
        ):
            raise CheckNotApplicable()

        values, counts = np.unique_counts(
            np.concatenate([report.y_train, report.y_test])
        )

        underrepresented_classes = values[counts <= 0.1 * counts.sum()]
        if underrepresented_classes.size > 0:
            return (
                f"Classes {underrepresented_classes} each represent less than 10% of "
                "the dataset samples. Accuracy should not be used alone to assess "
                "model performance as it may be misleading by ignoring poor "
                "performance on underrepresented classes."
            )
        return None


class CheckUnscaledCoefficients(Check):
    """Check for unscaled coefficients (SKD006).

    Warns that when using a linear model, the input features should be scaled
    so that their coefficient values can be used as a measure of feature importance.
    """

    code = "SKD006"
    title = "Unscaled coefficients"
    report_type = "estimator"
    docs_url = "skd006-unscaled_coefficients"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        from skore._sklearn._estimator.report import EstimatorReport

        if not (
            isinstance(report, EstimatorReport)
            and report.X_train is not None
            and report.X_test is not None
            and hasattr(report.estimator, "coef_")
        ):
            raise CheckNotApplicable()

        X = np.concatenate([report.X_train, report.X_test])
        if not (np.allclose(X.mean(axis=0), 0) and np.allclose(X.std(axis=0), 1)):
            return (
                "The input features do not appear to be standardized. Be careful "
                "when interpreting the magnitudes of a linear model's coefficients "
                "as feature importance: they depend on the scale of each feature "
                "and are not directly comparable across features."
            )
        return None


_BUILTIN_CHECKS = [
    CheckOverfitting(),
    CheckUnderfitting(),
    CheckMetricsConsistencyAcrossSplits(),
    CheckHighClassImbalance(),
    CheckUnderrepresentedClasses(),
    CheckUnscaledCoefficients(),
]
