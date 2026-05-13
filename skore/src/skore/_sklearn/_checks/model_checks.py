from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression, RidgeCV

from skore._externals._skrub_compat import tabular_pipeline
from skore._sklearn._checks.base import Check
from skore._sklearn._checks.utils import (
    _TIMING_METRICS,
    CheckNotApplicable,
    _metric_key,
    check_score_gap_to_baseline,
    collect_scores,
    detect_outliers_modified_zscore,
    majority_vote,
)

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport
    from skore._sklearn._estimator.report import EstimatorReport


def _require_estimator_with_train_test(report: _BaseReport) -> EstimatorReport:
    """Return ``report`` cast as an EstimatorReport with full train/test data.

    Raises :class:`CheckNotApplicable` otherwise.
    """
    from skore._sklearn._estimator.report import EstimatorReport

    if (
        not isinstance(report, EstimatorReport)
        or report.X_train is None
        or report.y_train is None
        or report.X_test is None
        or report.y_test is None
    ):
        raise CheckNotApplicable()
    return report


def _baseline_estimator_report(
    report: EstimatorReport,
    kind: Literal["dummy", "performance", "fast"],
) -> EstimatorReport:
    """Build a baseline EstimatorReport mirroring ``report``.

    For ``kind="dummy"``, returns a plain ``DummyClassifier`` / ``DummyRegressor``
    baseline. For ``kind="performance"`` and ``kind="fast"``, the estimator is
    wrapped in :func:`skrub.tabular_pipeline`.

    Raises :class:`CheckNotApplicable` for unsupported ml tasks.
    """
    from skore._sklearn._estimator.report import EstimatorReport

    is_classification = report.ml_task in (
        "binary-classification",
        "multiclass-classification",
    )
    if not (is_classification or report.ml_task == "regression"):
        raise CheckNotApplicable()
    if kind == "dummy":
        estimator = (
            DummyClassifier(strategy="prior")
            if is_classification
            else DummyRegressor(strategy="mean")
        )
    elif kind == "performance":
        estimator = tabular_pipeline(
            HistGradientBoostingClassifier()
            if is_classification
            else HistGradientBoostingRegressor()
        )
    else:  # kind == "fast"
        estimator = tabular_pipeline(
            LogisticRegression(max_iter=1000) if is_classification else RidgeCV()
        )

    baseline_report = EstimatorReport(
        estimator,
        X_train=report.X_train,
        y_train=report.y_train,
        X_test=report.X_test,
        y_test=report.y_test,
        pos_label=report.pos_label,
    )
    baseline_report._metric_registry = report._metric_registry
    return baseline_report


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
        report = _require_estimator_with_train_test(report)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            report_train = collect_scores(report, data_source="train")
            report_test = collect_scores(report, data_source="test")

        votes = [
            check_score_gap_to_baseline(
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
        report = _require_estimator_with_train_test(report)
        baseline = _baseline_estimator_report(report, kind="dummy")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            report_train = collect_scores(report, data_source="train")
            report_test = collect_scores(report, data_source="test")
            baseline_train = collect_scores(baseline, data_source="train")
            baseline_test = collect_scores(baseline, data_source="test")

        votes = [
            not check_score_gap_to_baseline(
                score=report_train[key]["score"],
                baseline=baseline_train[key]["score"],
                greater_is_better=baseline_train[key]["greater_is_better"],
                floor=0.01,
                fraction=0.05,
            )
            and not check_score_gap_to_baseline(
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


class CheckMetricsConsistencyAcrossSplits(Check):
    """Check the consistency of metrics across splits (SKD003).

    Outlier splits are identified with a modified Z-score based on the
    Median Absolute Deviation (MAD) to be robust to extreme values.
    """

    code = "SKD003"
    title = "Inconsistent performance across splits"
    report_type = "cross-validation"
    docs_url = "skd003-inconsistent-performance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        from skore._sklearn._cross_validation.report import CrossValidationReport

        if not isinstance(report, CrossValidationReport):
            raise CheckNotApplicable()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            rows = report.metrics.summarize(data_source="test").rows

        n_splits = len(report.estimator_reports_)
        scores_by_metric: dict[tuple, list[float]] = defaultdict(
            lambda: [np.nan] * n_splits
        )
        for row in rows:
            if row["metric_verbose_name"] in _TIMING_METRICS:
                continue
            scores_by_metric[_metric_key(row)][row["split"]] = row["score"]

        if not scores_by_metric:
            return None

        votes = np.array(
            [
                detect_outliers_modified_zscore(np.asarray(scores))
                for scores in scores_by_metric.values()
            ]
        )
        explanation = []
        for cv in range(n_splits):
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
    docs_url = "skd004-high-class-imbalance"
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
    docs_url = "skd005-underrepresented-classes"
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


class CheckCoefficientsInterpretation(Check):
    """Check coefficient interpretability for linear models (SKD006).

    Tips about whether coefficients can be compared across features and
    whether they retain their original-unit interpretation.
    """

    code = "SKD006"
    title = "Coefficient interpretation"
    report_type = "estimator"
    docs_url = "skd006-unscaled-coefficients"
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
        stds = X.std(axis=0)
        if not np.all(np.isclose(stds, stds[0])):
            return (
                "Features are not on the same scale: coefficient magnitudes "
                "are not directly comparable as feature importance."
            )
        return (
            "Features appear to be standardized: coefficients are comparable "
            "but no longer interpretable in the original feature units."
        )


class CheckWorseThanBaseline(Check):
    """Check whether the model is worse than a strong baseline (SKD009).

    Compares test-set scores against a
    :func:`skrub.tabular_pipeline`-wrapped HistGradientBoosting baseline.
    """

    code = "SKD009"
    title = "Model worse than baseline"
    report_type = "estimator"
    docs_url = "skd009-worse-than-baseline"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        report = _require_estimator_with_train_test(report)
        baseline = _baseline_estimator_report(report, kind="performance")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            report_test = collect_scores(report, data_source="test")
            baseline_test = collect_scores(baseline, data_source="test")

        votes = [
            not check_score_gap_to_baseline(
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
                "Test scores are not significantly better than a "
                "HistGradientBoosting baseline for "
                f"{n_positive}/{total} default predictive metrics."
            )
        return None


class CheckSlowerThanBaseline(Check):
    """Check whether the model is slower than a fast baseline (SKD010).

    Compares fit time and test-set scores against a
    :func:`skrub.tabular_pipeline`-wrapped fast linear baseline
    (:class:`~sklearn.linear_model.RidgeCV` for regression,
    :class:`~sklearn.linear_model.LogisticRegression` for classification).
    The slowness gate triggers when the report's fit time is at least ``2x``
    the baseline's, with an absolute gap of at least ``0.05s`` to avoid noise
    on very fast fits.
    """

    code = "SKD010"
    title = "Model slower than baseline"
    report_type = "estimator"
    docs_url = "skd010-slower-than-baseline"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        report = _require_estimator_with_train_test(report)
        if report.fit_time_ is None:
            raise CheckNotApplicable()
        baseline = _baseline_estimator_report(report, kind="fast")
        if baseline.fit_time_ is None:
            raise CheckNotApplicable()

        slowness_ratio = report.fit_time_ / baseline.fit_time_
        if slowness_ratio < 2.0 or report.fit_time_ - baseline.fit_time_ < 0.05:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            report_test = collect_scores(report, data_source="test")
            baseline_test = collect_scores(baseline, data_source="test")

        votes = [
            not check_score_gap_to_baseline(
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
                f"Fit is ~{slowness_ratio:.1f}x slower than a fast linear baseline "
                f"without significantly better test scores "
                f"({n_positive}/{total} default predictive metrics)."
            )
        return None


_BUILTIN_CHECKS = [
    CheckOverfitting(),
    CheckUnderfitting(),
    CheckMetricsConsistencyAcrossSplits(),
    CheckHighClassImbalance(),
    CheckUnderrepresentedClasses(),
    CheckCoefficientsInterpretation(),
    CheckWorseThanBaseline(),
    CheckSlowerThanBaseline(),
]
