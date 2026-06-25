from __future__ import annotations

import numbers
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, cast

import narwhals as nw
import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import Interval
from sklearn.utils._pprint import _changed_params
from skrub import tabular_pipeline

from skore._sklearn._checks._utils import (
    CheckNotApplicable,
    ClassName,
    ParameterName,
    StepName,
    cast_report,
    check_score_gap_to_baseline,
    collect_scores,
    detect_outliers_modified_zscore,
    get_fit_time,
    get_fitted_estimator,
    get_preprocessed_X,
    get_report_y,
    majority_vote,
    split_preprocessor_estimator,
)
from skore._sklearn._checks.base import Check
from skore._sklearn._checks.tunable_hyperparameters import (
    EQUIVALENT_PARAM_GROUPS,
    HYPERPARAMETERS_TO_TUNE,
    INFRASTRUCTURE_PARAMS,
)
from skore._sklearn.feature_names import _get_feature_names
from skore._utils._dataframe import UserSeries, _normalize_X_as_dataframe
from skore._utils._skrub import iter_fitted_estimator_steps

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport

_TIMING_METRICS_FLAT = {"fit_time_s", "predict_time_s"}


def _baseline_estimator_report(
    report: EstimatorReport | CrossValidationReport,
    kind: Literal["dummy", "performance", "fast"],
) -> EstimatorReport | CrossValidationReport:
    """Build a baseline report mirroring ``report``.

    For ``kind="dummy"``, returns a plain ``DummyClassifier`` / ``DummyRegressor``
    baseline. For ``kind="performance"`` and ``kind="fast"``, the estimator is
    wrapped in :func:`skrub.tabular_pipeline`.

    Raises :class:`CheckNotApplicable` for unsupported ml tasks.
    """
    is_classification = report.ml_task in (
        "binary-classification",
        "multiclass-classification",
    )
    if not (is_classification or report.ml_task == "regression"):
        raise CheckNotApplicable(
            "Unsupported ML task. Supported tasks are: binary-classification, "
            f"multiclass-classification, regression. Got {report.ml_task}."
        )
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

    if report._report_type == "cross-validation":
        from skore._sklearn._cross_validation.report import CrossValidationReport

        try:
            baseline = CrossValidationReport(
                estimator,
                X=report.X,
                y=report.y,
                splitter=report.splitter,
                pos_label=report.pos_label,
                n_jobs=report.n_jobs,
            )
        except Exception as exc:
            raise CheckNotApplicable("Failed to create baseline report.") from exc
        registry = report.reports_[0]._metric_registry.copy()
        for baseline_split in baseline.reports_:
            baseline_split._metric_registry = registry
        return baseline

    if report.X_train is None:
        raise CheckNotApplicable("Train data is unavailable.")
    try:
        X_train = _normalize_X_as_dataframe(report.X_train)
        X_test = _normalize_X_as_dataframe(report.X_test)
    except NotImplementedError:
        raise CheckNotApplicable("Data is sparse.") from None

    y_train = get_report_y(report, data_source="train")
    y_test = get_report_y(report, data_source="test")
    from skore._sklearn._estimator.report import EstimatorReport

    try:
        baseline_report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pos_label=report.pos_label,
        )
    except Exception as exc:
        raise CheckNotApplicable("Failed to create baseline report.") from exc
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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd002-underfitting"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect train and test scores close to a dummy baseline."""
        report = cast_report(report)
        baseline = _baseline_estimator_report(report, kind="dummy")

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
    report_types = ["cross-validation"]
    docs_url = "skd003-inconsistent-performance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect outlier performance across cross-validation splits."""
        report = cast("CrossValidationReport", report)

        report_data = report.metrics.summarize(data_source="test").frame(
            aggregate=None, flat_index=True
        )
        votes = np.array(
            [
                detect_outliers_modified_zscore(report_data.loc[idx])
                for idx in report_data.index
                if idx not in _TIMING_METRICS_FLAT
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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd004-high-class-imbalance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect when the majority class exceeds 80% of samples."""
        report = cast_report(report)
        if report.ml_task != "binary-classification":
            raise CheckNotApplicable(
                f"ML task is not binary classification. Got {report.ml_task}."
            )
        y = get_report_y(report, data_source="both")

        y = nw.from_native(cast(UserSeries, y), series_only=True)
        counts = y.value_counts()
        value_col = counts.columns[0]
        total = counts["count"].sum()
        overrepresented_class = counts.filter(nw.col("count") >= 0.8 * total)[
            value_col
        ].to_list()

        if len(overrepresented_class) > 0:
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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd005-underrepresented-classes"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect classes that each represent less than 10% of samples."""
        report = cast_report(report)
        if report.ml_task != "multiclass-classification":
            raise CheckNotApplicable(
                f"ML task is not multiclass classification. Got {report.ml_task}."
            )

        y = get_report_y(report, data_source="both")

        y = nw.from_native(cast(UserSeries, y), series_only=True)
        counts = y.value_counts()
        value_col = counts.columns[0]
        total = counts["count"].sum()
        underrepresented_classes = counts.filter(nw.col("count") <= 0.1 * total)[
            value_col
        ].to_list()
        if len(underrepresented_classes) > 0:
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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd006-unscaled-coefficients"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        """Assess whether linear-model coefficients are comparable and interpretable."""
        report = cast_report(report)
        _, predictor = split_preprocessor_estimator(get_fitted_estimator(report))

        if not hasattr(predictor, "coef_"):
            raise CheckNotApplicable(
                "Estimator is not a linear model: it does not have a `coef_` attribute."
            )

        X = get_preprocessed_X(report, data_source="both")

        std_values = nw.from_native(X).select(nw.all().std()).to_numpy().ravel()
        if not np.allclose(std_values, std_values[0], atol=0.05):
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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd007-mdi-cardinality-bias"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect high-cardinality features that may bias MDI importances."""
        report = cast_report(report)
        _, predictor = split_preprocessor_estimator(get_fitted_estimator(report))

        if not hasattr(predictor, "feature_importances_"):
            raise CheckNotApplicable(
                "Estimator is not a tree-based model: it does not have a "
                "`feature_importances_` attribute."
            )

        X = get_preprocessed_X(report, data_source="train")

        X = nw.from_native(X)
        n_samples = X.shape[0]
        high_cardinality_features = [
            column
            for column in X.columns
            if X.select(nw.col(column).n_unique()).item(0, 0) > 0.5 * n_samples
        ]

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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd008-correlated-features"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect pairs of numeric features with Spearman correlation above 0.9.

        Returns
        -------
        str or None
            Check result ``explanation`` when highly correlated features are
            detected; ``None`` when the check passes. Raises
            :class:`CheckNotApplicable` when feature data is unavailable or
            fewer than two numeric features are present.
        """
        report = cast_report(report)
        X = get_preprocessed_X(report, data_source="train")

        X = nw.from_native(X).select(nw.selectors.numeric())
        if X.shape[1] < 2 or X.shape[1] > 1000:
            raise CheckNotApplicable(
                "Expected train data to have between 2 and 1000 features; "
                f"got {X.shape[1]}."
            )

        corr = np.abs(spearmanr(X.to_numpy()).statistic)
        if corr.ndim < 2:
            raise CheckNotApplicable("Less than 2 numeric features are present.")
        np.fill_diagonal(corr, 0)
        n_pairs = int(np.count_nonzero(corr >= 0.9) // 2)

        if n_pairs:
            return (
                f"{n_pairs} pair(s) of features have a Spearman correlation "
                "above 0.9. Highly correlated features can destabilize "
                "linear model coefficients and feature-importance estimates, "
                "and may cause collinearity-induced numerical issues."
                "Dropping redundant features may also improve model performance."
            )
        return None


class CheckWorseThanBaseline(Check):
    """Check whether the model is worse than a strong baseline (SKD009).

    Compares test-set scores against a
    :func:`skrub.tabular_pipeline`-wrapped HistGradientBoosting baseline.
    """

    code = "SKD009"
    title = "Model worse than baseline"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd009-worse-than-baseline"
    severity = "issue"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        baseline = _baseline_estimator_report(report, kind="performance")

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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd010-slower-than-baseline"
    severity = "issue"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        baseline = _baseline_estimator_report(report, kind="fast")

        report_fit_time = get_fit_time(report)
        baseline_fit_time = get_fit_time(baseline)

        slowness_ratio = report_fit_time / baseline_fit_time
        if slowness_ratio < 2.0 or report_fit_time - baseline_fit_time < 0.05:
            return None

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


class CheckGoldenFeature(Check):
    """Check for a golden feature (SKD011).

    Detects a single feature that, used alone to refit the estimator, reaches
    scores close to the full model on the report's default predictive metrics.
    This often signals data leakage or excessive reliance on one feature.
    """

    code = "SKD011"
    title = "Golden feature"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd011-golden-feature"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        if report._report_type == "cross-validation":
            report = cast("CrossValidationReport", report)
            X = nw.from_native(get_preprocessed_X(report))
            y = get_report_y(report)
            if X.shape[1] < 2:
                raise CheckNotApplicable("Train data has only one feature.")
            n_features = X.shape[1]
            metric_registry = report.reports_[0]._metric_registry.copy()
            from skore._sklearn._cross_validation.report import CrossValidationReport
        else:
            report = cast("EstimatorReport", report)
            X_train = nw.from_native(get_preprocessed_X(report, data_source="train"))
            X_test = nw.from_native(get_preprocessed_X(report, data_source="test"))
            y_train = get_report_y(report, data_source="train")
            y_test = get_report_y(report, data_source="test")
            if X_train.shape[1] < 2:
                raise CheckNotApplicable("Train data has only one feature.")
            n_features = X_train.shape[1]
            metric_registry = report._metric_registry
            from skore._sklearn._estimator.report import EstimatorReport

        preprocessor_, predictor_ = split_preprocessor_estimator(
            get_fitted_estimator(report)
        )
        feature_names = _get_feature_names(
            predictor_,
            transformer=preprocessor_,
            X=X if report._report_type == "cross-validation" else X_train,
            n_features=n_features,
        )
        full_feature_scores = collect_scores(report, data_source="test")

        golden_features: list[str] = []
        single_feature_report: EstimatorReport | CrossValidationReport
        for i in range(n_features):
            try:
                if report._report_type == "cross-validation":
                    single_feature_report = CrossValidationReport(
                        clone(predictor_),
                        X=X.select(nw.col(feature_names[i])).to_native(),
                        y=y,
                        splitter=report.splitter,
                        pos_label=report.pos_label,
                        n_jobs=report.n_jobs,
                    )
                    for sub_report in single_feature_report.reports_:
                        sub_report._metric_registry = metric_registry
                else:
                    single_feature_report = EstimatorReport(
                        clone(predictor_),
                        X_train=X_train.select(nw.col(feature_names[i])).to_native(),
                        y_train=y_train,
                        X_test=X_test.select(nw.col(feature_names[i])).to_native(),
                        y_test=y_test,
                        pos_label=report.pos_label,
                    )
                    single_feature_report._metric_registry = metric_registry
                single_feature_scores = collect_scores(
                    single_feature_report, data_source="test"
                )
            except Exception as exc:
                raise CheckNotApplicable(
                    "Failed to create report from single feature."
                ) from exc
            votes = [
                not check_score_gap_to_baseline(
                    score=full_feature_scores[key]["score"],
                    baseline=single_feature_scores[key]["score"],
                    greater_is_better=full_feature_scores[key]["greater_is_better"],
                    floor=0.03,
                    fraction=0.10,
                )
                for key in full_feature_scores.keys() & single_feature_scores.keys()
            ]
            if majority_vote(votes)[0]:
                golden_features.append(str(feature_names[i]))

        if golden_features:
            return (
                f"A model trained on feature(s) {golden_features} alone has similar "
                "performance to a model trained on all the features, on the default "
                "predictive metrics. This may signal data leakage or excessive "
                "reliance on a single feature."
            )
        return None


class CheckUselessFeatures(Check):
    """Check for useless features (SKD012).

    Flags features whose permutation importance is negligible: either the mean
    importance is negative, below 1e-3, or its interval ``[mean - std,
    mean + std]`` contains zero.

    Permutation importance is computed via the inspection accessor with a
    fixed seed so the result is cached and shared with explicit calls to
    :meth:`~skore.EstimatorReport.inspection.permutation_importance`.
    """

    code = "SKD012"
    title = "Useless features"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd012-useless-features"
    severity = "tip"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)

        try:
            importance_frame = report.inspection.permutation_importance(
                data_source="test", seed=0, n_repeats=5
            ).frame()
        except (ValueError, TypeError) as err:
            raise CheckNotApplicable(
                "Failed to compute permutation importance."
            ) from err

        # group by feature and take the mean over metric/label/output
        per_feature = (
            importance_frame.groupby("feature")[["value_mean", "value_std"]]
            .mean()
            .reset_index()
        )
        mean = per_feature["value_mean"]
        std = per_feature["value_std"]
        useless = per_feature.loc[
            (mean <= 1e-3) | ((mean - std <= 0) & (mean + std >= 0)), "feature"
        ].tolist()
        if useless:
            return (
                f"Feature(s) {useless} have permutation importance overlapping "
                "with zero and could likely be dropped without degrading "
                "performance. Dropping redundant features may also improve model "
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
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd013-train-test-time-overlap"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        input_report = cast_report(report)

        overlapping: set[str] = set()
        found_datetime = False
        for report in (
            input_report.reports_
            if input_report._report_type == "cross-validation"
            else [input_report]
        ):
            if report.X_train is None:
                raise CheckNotApplicable("Train data is unavailable.")
            if not nw.dependencies.is_into_dataframe(report.X_train):
                raise CheckNotApplicable(
                    "Input data is not a narwhals compatible DataFrame. "
                    f"Got {type(report.X_train)}."
                )
            if not nw.dependencies.is_into_dataframe(report.X_test):
                raise CheckNotApplicable(
                    "Input data is not a narwhals compatible DataFrame. "
                    f"Got {type(report.X_test)}."
                )
            X_train_nw = nw.from_native(report.X_train)
            X_test_nw = nw.from_native(report.X_test)

            datetime_columns = sorted(
                set(X_train_nw.select(nw.selectors.datetime()).columns)
                & set(X_test_nw.select(nw.selectors.datetime()).columns)
            )
            if datetime_columns:
                found_datetime = True
                overlapping.update(
                    col
                    for col in datetime_columns
                    if X_train_nw[col].max() >= X_test_nw[col].min()
                )

        if not found_datetime:
            raise CheckNotApplicable("No datetime column found.")
        if overlapping:
            return (
                f"Datetime column(s) {sorted(overlapping)} contain training "
                "timestamps that are after the earliest test timestamp. Future "
                "points may be leaking into the training set; consider a "
                "time-based split."
            )
        return None


class CheckHyperparamsAtSearchEdge(Check):
    """Check whether tuned hyperparameters sit at the search boundary (SKD014).

    For :class:`~sklearn.model_selection.BaseSearchCV` estimators, flags when any
    numeric ``best_params_`` value equals the minimum or maximum distinct value
    tried for that parameter. Non-numeric hyperparameters (``bool``, strings, ``None``,
    and similar) are skipped because extending the search range is not meaningful.
    """

    code = "SKD014"
    title = "Hyperparameters at search edge"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd014-hyperparams-at-search-edge"
    severity = "issue"

    @staticmethod
    def _get_space_bound(
        estimator, *, param_name: str, side: Literal["left", "right"]
    ) -> float | None:
        """Fetch the closed parameter-space boundary for `side` if it exists."""
        *step_names, leaf_param = param_name.split("__")
        owner = estimator
        for step_name in step_names:
            owner = owner.get_params(deep=True)[step_name]
        if not hasattr(owner, "_parameter_constraints"):
            return None
        for constraint in owner._parameter_constraints[leaf_param]:
            if isinstance(constraint, Interval) and constraint.closed in [side, "both"]:
                return float(getattr(constraint, side))
        return None

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        estimator = get_fitted_estimator(report)
        if not isinstance(estimator, BaseSearchCV):
            raise CheckNotApplicable(
                f"Estimator is not a BaseSearchCV instance. Got {type(estimator)}."
            )

        param_combinations = estimator.cv_results_["params"]

        edge_params = []
        for param_name, best_value in estimator.best_params_.items():
            tried = [
                param_combination[param_name]
                for param_combination in param_combinations
                if param_name in param_combination
            ]
            if len(set(tried)) < 2 or not all(
                isinstance(value, numbers.Real)
                and not isinstance(value, bool | np.bool_)
                for value in tried
            ):
                continue
            search_low, search_high = min(tried), max(tried)
            if not isinstance(best_value, numbers.Real) or isinstance(
                best_value, bool | np.bool_
            ):
                continue
            if float(best_value) == float(search_low):
                space_low = self._get_space_bound(
                    estimator.estimator, param_name=param_name, side="left"
                )
                if space_low is not None and float(search_low) == space_low:
                    continue
                edge_params.append((param_name, "minimum"))
            elif float(best_value) == float(search_high):
                space_high = self._get_space_bound(
                    estimator.estimator, param_name=param_name, side="right"
                )
                if space_high is not None and float(search_high) == space_high:
                    continue
                edge_params.append((param_name, "maximum"))

        if not edge_params:
            return None
        details = ", ".join(f"{name} ({bound})" for name, bound in edge_params)
        return (
            f"{len(edge_params)} hyperparameter(s) are on the edge of the explored "
            f"search space: {details}. Consider extending the search range or "
            "increasing the number of iterations for randomized search."
        )


def _collapse_equivalents(
    recommended: set[ParameterName], searched: set[ParameterName]
) -> set[ParameterName]:
    """Return ``recommended - searched``, collapsing equivalence groups.

    Some parameters serve the same purpose (e.g. ``max_depth``, ``min_samples_leaf``,
    ``min_samples_split``, ``max_leaf_nodes`` all limit tree depth). If any group
    member is already searched, drop the others; otherwise keep only the first
    missing member of the group.
    """
    missing = recommended - searched
    for group in EQUIVALENT_PARAM_GROUPS:
        group_set = set(group)
        if searched & group_set:
            missing -= group_set
        else:
            in_group = [param for param in group if param in missing]
            if len(in_group) > 1:
                missing -= group_set
                missing.add(in_group[0])
    return missing


class CheckSearchParamsToTune(Check):
    """Check for hyperparameters worth tuning in a search (SKD015).

    For :class:`~sklearn.model_selection.BaseSearchCV` estimators, compares the
    parameters being searched against a set of important hyperparameters and suggests
    any that are missing.

    When the search wraps a :class:`~sklearn.pipeline.Pipeline`, each step
    whose class appears in the recommendation table is checked independently,
    regardless of whether the search currently tunes any of its parameters.
    """

    code = "SKD015"
    title = "Hyperparameters worth tuning"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd015-hyperparameters-worth-tuning"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        estimator = get_fitted_estimator(report)
        if not isinstance(estimator, BaseSearchCV):
            raise CheckNotApplicable(
                f"Estimator is not a BaseSearchCV instance. Got {type(estimator)}."
            )

        searched_keys = {
            key for params in estimator.cv_results_["params"] for key in params
        }
        inner_estimator = estimator.estimator
        if isinstance(inner_estimator, Pipeline):
            searched_params_by_step: dict[StepName, set[ParameterName]] = defaultdict(
                set
            )
            for key in searched_keys:
                if "__" in key:
                    step_name, suffix = key.split("__", 1)
                    searched_params_by_step[step_name].add(suffix)
            searched_by_estimator: list[tuple[ClassName, set[ParameterName]]] = [
                (type(step).__name__, searched_params_by_step.get(name, set()))
                for name, step in inner_estimator.steps
                if type(step).__name__ in HYPERPARAMETERS_TO_TUNE
            ]
            if not searched_by_estimator:
                raise CheckNotApplicable(
                    "No parameter to recommend for any of the steps."
                )
        else:
            class_name = type(inner_estimator).__name__
            if class_name not in HYPERPARAMETERS_TO_TUNE:
                raise CheckNotApplicable("No parameter to recommend for the estimator.")
            searched_by_estimator = [(class_name, searched_keys)]

        messages: list[str] = []
        for class_name, searched in searched_by_estimator:
            missing = _collapse_equivalents(
                HYPERPARAMETERS_TO_TUNE[class_name], searched
            )
            if missing:
                messages.append(f"{sorted(missing)} for {class_name}")
        if not messages:
            return None
        messages.sort()
        return (
            "These hyperparameters are not in the grid and may be worth tuning: "
            f"{'; '.join(messages)}."
        )


class CheckEstimatorNotTuned(Check):
    """Check that the estimator has at least some non-default hyperparameters (SKD016).

    Fires when every parameter of the estimator (or, for pipelines, of every
    step whose class is in the recommendation table) is at scikit-learn's
    default value, ignoring infrastructure params (random_state, n_jobs, ...).
    Suggests the recommended tuning axes from ``HYPERPARAMETERS_TO_TUNE``.

    Skipped (:class:`CheckNotApplicable`) when the estimator is a
    :class:`~sklearn.model_selection.BaseSearchCV` instance, since SKD015
    covers that case.
    """

    code = "SKD016"
    title = "Estimator not tuned"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd016-estimator-not-tuned"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        estimator = get_fitted_estimator(report)
        if isinstance(estimator, BaseSearchCV):
            raise CheckNotApplicable("Estimator is a BaseSearchCV instance.")

        candidates = [
            (class_name, step)
            for class_name, step in iter_fitted_estimator_steps(estimator)
            if class_name in HYPERPARAMETERS_TO_TUNE
        ]
        if not candidates:
            raise CheckNotApplicable("No parameter to recommend.")

        messages: list[str] = []
        for class_name, step in candidates:
            if set(_changed_params(step)) - INFRASTRUCTURE_PARAMS:
                continue
            recommended = _collapse_equivalents(
                HYPERPARAMETERS_TO_TUNE[class_name], set()
            )
            messages.append(f"{sorted(recommended)} for {class_name}")

        if not messages:
            return None
        messages.sort()
        return (
            "Estimator(s) left at default settings; consider tuning: "
            f"{'; '.join(sorted(messages))}."
        )


_BUILTIN_CHECKS = [
    CheckOverfitting(),
    CheckUnderfitting(),
    CheckMetricsConsistencyAcrossSplits(),
    CheckHighClassImbalance(),
    CheckUnderrepresentedClasses(),
    CheckCoefficientsInterpretation(),
    CheckMDIHighCardinalityBias(),
    CheckCorrelatedFeatures(),
    CheckWorseThanBaseline(),
    CheckSlowerThanBaseline(),
    CheckGoldenFeature(),
    CheckUselessFeatures(),
    CheckTrainTestTimeOverlap(),
    CheckHyperparamsAtSearchEdge(),
    CheckSearchParamsToTune(),
    CheckEstimatorNotTuned(),
]
