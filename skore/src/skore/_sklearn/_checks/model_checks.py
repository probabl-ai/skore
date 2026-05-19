from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._sklearn._checks._utils import (
    _TIMING_METRICS,
    CheckNotApplicable,
    check_score_gap_to_baseline,
    detect_outliers_modified_zscore,
    get_preprocessed_data,
    majority_vote,
    select_feature,
    split_preprocessor_estimator,
)
from skore._sklearn._checks.base import Check
from skore._sklearn.feature_names import _get_feature_names

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport


def _get_metrics_data(report: EstimatorReport) -> tuple:
    """Compute report/baseline metrics data for SKD001 and SKD002.

    Raises :class:`CheckNotApplicable` when train+test data is
    unavailable.
    """
    # Avoid circular import
    from skore._sklearn._estimator.report import EstimatorReport

    if (
        report.X_train is None
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
        X_test=report.X_test,
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
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
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
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
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
    docs_url = "skd003-inconsistent-performance"
    severity = "issue"
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("CrossValidationReport", report)

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
    docs_url = "skd004-high-class-imbalance"
    severity = "issue"
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
        y = get_preprocessed_data(report, target="y", concatenate=True)
        if report.ml_task != "binary-classification" or y is None:
            raise CheckNotApplicable()

        values, counts = np.unique_counts(y)
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
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)

        y = get_preprocessed_data(report, target="y", concatenate=True)
        if report.ml_task != "multiclass-classification" or y is None:
            raise CheckNotApplicable()

        values, counts = np.unique_counts(y)
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
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
        _, predictor = split_preprocessor_estimator(report.estimator)
        X = get_preprocessed_data(report, target="X", concatenate=True)

        if X is None or not hasattr(predictor, "coef_"):
            raise CheckNotApplicable()

        stds = np.asarray(X.std(axis=0))
        if not np.all(np.isclose(stds, stds[0])):
            return (
                "Features are not on the same scale: coefficient magnitudes "
                "are not directly comparable as feature importance."
            )
        return (
            "Features appear to be standardized: coefficients are comparable "
            "but no longer interpretable in the original feature units."
        )


class CheckMDIHighCardinalityBias(Check):
    """Check for MDI bias with high-cardinality features (SKD007).

    Tips that mean-decrease-in-impurity importances may be inflated for
    continuous or high-cardinality features.

    We consider a feature to be high-cardinality when its number of unique values
    exceeds 50% of the number of samples.
    """

    code = "SKD007"
    title = "MDI biased for high-cardinality features"
    report_type = "estimator"
    docs_url = "skd007-mdi-cardinality-bias"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
        _, predictor = split_preprocessor_estimator(report.estimator)
        X = get_preprocessed_data(report, target="X")

        if X is None or not hasattr(predictor, "feature_importances_"):
            raise CheckNotApplicable()

        if isinstance(X, pd.DataFrame):
            high_cardinality_features = [
                c for c in X.columns if X[c].nunique() > 0.5 * len(X)
            ]
        elif isinstance(X, np.ndarray):
            high_cardinality_features = [
                i for i in range(X.shape[1]) if np.unique(X[:, i]).size > 0.5 * len(X)
            ]
        else:
            raise CheckNotApplicable()

        if high_cardinality_features:
            names = ", ".join(str(s) for s in high_cardinality_features[:3])
            suffix = (
                f" (and {len(high_cardinality_features) - 3} more)"
                if len(high_cardinality_features) > 3
                else ""
            )
            return (
                f"High-cardinality features detected: {names}{suffix}. "
                "Mean Decrease in Impurity (MDI) importance is biased toward "
                "such features. Consider using permutation importance for "
                "a more robust alternative."
            )
        return None


class CheckCorrelatedFeatures(Check):
    """Check for highly correlated input features (SKD008).

    Flags when one or more pairs of numeric features have a Spearman rank
    correlation above 0.9.
    """

    code = "SKD008"
    title = "Highly correlated input features"
    report_type = "estimator"
    docs_url = "skd008-correlated-features"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
        X = get_preprocessed_data(report, target="X")

        if X is None:
            raise CheckNotApplicable()
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include="number")
        if X.shape[1] < 2:
            raise CheckNotApplicable()

        corr = np.abs(spearmanr(X).statistic)
        if corr.ndim < 2:
            return None
        np.fill_diagonal(corr, 0)
        n_pairs = int(np.count_nonzero(corr >= 0.9) // 2)

        if n_pairs:
            return (
                f"{n_pairs} pair(s) of features have a Spearman correlation "
                "above 0.9. Highly correlated features can destabilize "
                "linear model coefficients and feature-importance estimates, "
                "and may cause collinearity-induced numerical issues."
            )
        return None


class CheckGoldenFeature(Check):
    """Check for a golden feature (SKD011).

    Detects a single feature that, used alone to refit the estimator, reaches
    scores close to the full model on the report's default predictive metrics.
    This often signals data leakage or excessive reliance on one feature.
    """

    code = "SKD011"
    title = "Golden feature"
    report_type = "estimator"
    docs_url = "skd011-golden-feature"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        from skore._sklearn._estimator.report import EstimatorReport

        report = cast("EstimatorReport", report)
        if report._initialized_with_data_op:
            raise CheckNotApplicable()

        X_train = report.X_train
        if X_train is None or not isinstance(X_train, (np.ndarray, pd.DataFrame)):
            raise CheckNotApplicable()

        n_features = X_train.shape[1]
        if n_features < 2:
            raise CheckNotApplicable()

        if report.X_test is None or report.y_train is None or report.y_test is None:
            raise CheckNotApplicable()

        feature_names = _get_feature_names(
            report.estimator_, X=X_train, n_features=n_features
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            full_data = report.metrics.summarize(data_source="test").data

        predictive_rows = [
            idx
            for idx in range(len(full_data))
            if full_data.loc[idx, "metric_verbose_name"] not in _TIMING_METRICS
            and not pd.isna(full_data.loc[idx, "greater_is_better"])
        ]
        if not predictive_rows:
            raise CheckNotApplicable()

        golden_features: list[str] = []
        for i in range(n_features):
            try:
                single_estimator = clone(report._raw_estimator)
                single_report = EstimatorReport(
                    single_estimator,
                    X_train=select_feature(X_train, i),
                    y_train=report.y_train,
                    X_test=select_feature(report.X_test, i),
                    y_test=report.y_test,
                    pos_label=report.pos_label,
                )
            except Exception as exc:
                raise CheckNotApplicable() from exc
            single_report._metric_registry = report._metric_registry
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                single_data = single_report.metrics.summarize(data_source="test").data

            votes = [
                not check_score_gap_to_baseline(
                    score=full_data.loc[idx, "score"],
                    baseline=single_data.loc[idx, "score"],
                    greater_is_better=full_data.loc[idx, "greater_is_better"],
                    floor=0.03,
                    fraction=0.10,
                )
                for idx in predictive_rows
            ]
            majority, _, _ = majority_vote(votes)
            if majority:
                golden_features.append(str(feature_names[i]))

        if golden_features:
            return (
                f"Feature(s) {golden_features} alone reach scores close to the "
                "full model's on the default predictive metrics. This may "
                "signal data leakage or excessive reliance on a single feature."
            )
        return None


class CheckUselessFeatures(Check):
    """Check for useless features (SKD012).

    Flags features whose permutation importance is negligible: either the mean
    importance is negative, below 1e-6, or its interval ``[mean - std,
    mean + std]`` contains zero.

    Permutation importance is computed via the inspection accessor with a
    fixed seed so the result is cached and shared with explicit calls to
    :meth:`~skore.EstimatorReport.inspection.permutation_importance`.
    """

    code = "SKD012"
    title = "Useless features"
    report_type = "estimator"
    docs_url = "skd012-useless-features"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
        if (
            report._initialized_with_data_op
            or report.X_train is None
            or report.X_test is None
            or report.y_test is None
        ):
            raise CheckNotApplicable()

        display = report.inspection.permutation_importance(
            data_source="test", seed=0, n_repeats=5
        )
        frame = display.frame()

        per_feature = (
            frame.groupby("feature")[["value_mean", "value_std"]].mean().reset_index()
        )
        mean = per_feature["value_mean"]
        std = per_feature["value_std"]
        useless_mask = (mean <= 1e-6) | ((mean - std <= 0) & (mean + std >= 0))
        useless = per_feature.loc[useless_mask, "feature"].tolist()
        if useless:
            return (
                f"Feature(s) {useless} have permutation importance overlapping "
                "with zero and could likely be dropped without degrading "
                "performance."
            )
        return None


class CheckTrainTestTimeOverlap(Check):
    """Check for train/test temporal overlap (SKD013).

    Flags datetime columns where the latest train timestamp is at or after
    the earliest test timestamp, indicating that future points leak into
    the training set (e.g. data was shuffled before splitting a time series).

    Raises :class:`CheckNotApplicable` when train and test inputs are not
    pandas DataFrames or contain no datetime column.
    """

    code = "SKD013"
    title = "Train-test overlap in time series"
    report_type = "estimator"
    docs_url = "skd013-train-test-time-overlap"
    severity = "issue"
    slow = False

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast("EstimatorReport", report)
        if not isinstance(report.X_train, pd.DataFrame) or not isinstance(
            report.X_test, pd.DataFrame
        ):
            raise CheckNotApplicable()

        datetime_columns = [
            col
            for col in report.X_train.columns
            if col in report.X_test.columns
            and pd.api.types.is_datetime64_any_dtype(report.X_train[col])
            and pd.api.types.is_datetime64_any_dtype(report.X_test[col])
        ]
        if not datetime_columns:
            raise CheckNotApplicable()

        overlapping = [
            col
            for col in datetime_columns
            if report.X_train[col].max() >= report.X_test[col].min()
        ]
        if overlapping:
            return (
                f"Datetime column(s) {overlapping} contain training timestamps "
                "that are at or after the earliest test timestamp. Future "
                "points may be leaking into the training set; consider a "
                "time-based split."
            )
        return None


_BUILTIN_CHECKS = [
    CheckOverfitting(),
    CheckUnderfitting(),
    CheckMetricsConsistencyAcrossSplits(),
    CheckHighClassImbalance(),
    CheckUnderrepresentedClasses(),
    CheckCoefficientsInterpretation(),
    CheckMDIHighCardinalityBias(),
    CheckCorrelatedFeatures(),
    CheckGoldenFeature(),
    CheckUselessFeatures(),
    CheckTrainTestTimeOverlap(),
]
