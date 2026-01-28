from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseReport
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn.types import (
    _DEFAULT,
    Metric,
    PositiveLabel,
)
from skore._utils._cache import Cache
from skore._utils._progress_bar import progress_decorator

if TYPE_CHECKING:
    from skore._sklearn._comparison.feature_importance_accessor import (
        _FeatureImportanceAccessor,
    )
    from skore._sklearn._comparison.metrics_accessor import _MetricsAccessor

    ReportType = Literal["EstimatorReport", "CrossValidationReport"]


class ComparisonReport(_BaseReport, DirNamesMixin):
    """Report for comparing reports.

    This object can be used to compare several :class:`skore.EstimatorReport` instances,
    or several :class:`~skore.CrossValidationReport` instances.

    Refer to the :ref:`comparison_report` section of the user guide for more details.

    .. caution::
       Reports passed to `ComparisonReport` are not copied. If you pass
       a report to `ComparisonReport`, and then modify the report outside later, it
       will affect the report stored inside the `ComparisonReport` as well, which
       can lead to inconsistent results. For this reason, modifying reports after
       creation is strongly discouraged.

    Parameters
    ----------
    reports : list of reports or dict
        Reports to compare. If a dict, keys will be used to label the estimators;
        if a list, the labels are computed from the estimator class names.
        Expects at least two reports to compare, with the same test target.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimators and computing
        the scores are parallelized.
        When accessing some methods of the `ComparisonReport`, the `n_jobs`
        parameter is used to parallelize the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    reports_ : dict mapping names to reports
        The compared reports.

    See Also
    --------
    skore.EstimatorReport
        Report for a fitted estimator.

    skore.CrossValidationReport
        Report for the cross-validation of an estimator.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from skore import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import ComparisonReport, EstimatorReport
    >>> X, y = make_classification(random_state=42)
    >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
    >>> estimator_1 = LogisticRegression()
    >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
    >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
    >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
    >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
    >>> report.reports_
    {'LogisticRegression_1': ..., 'LogisticRegression_2': ...}
    >>> report = ComparisonReport(
    ...     {"model1": estimator_report_1, "model2": estimator_report_2}
    ... )
    >>> report.reports_
    {'model1': ..., 'model2': ...}

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import ComparisonReport, CrossValidationReport
    >>> X, y = make_classification(random_state=42)
    >>> estimator_1 = LogisticRegression()
    >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
    >>> report_1 = CrossValidationReport(estimator_1, X, y)
    >>> report_2 = CrossValidationReport(estimator_2, X, y)
    >>> report = ComparisonReport([report_1, report_2])
    >>> report = ComparisonReport({"model1": report_1, "model2": report_2})
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]] = {
        "metrics": {"name": "metrics"},
        "feature_importance": {"name": "feature_importance"},
    }
    metrics: _MetricsAccessor
    feature_importance: _FeatureImportanceAccessor

    _reports_type: ReportType

    @staticmethod
    def _validate_reports(
        reports: list[EstimatorReport]
        | dict[str, EstimatorReport]
        | list[CrossValidationReport]
        | dict[str, CrossValidationReport],
    ) -> tuple[
        dict[str, EstimatorReport] | dict[str, CrossValidationReport],
        ReportType,
        PositiveLabel,
    ]:
        """Validate that reports are in the right format for comparison.

        Parameters
        ----------
        reports : list of reports or dict
            The reports to be validated.

        Returns
        -------
        dict
            The validated reports.
        {"EstimatorReport", "CrossValidationReport"}
            The inferred type of the reports that will be compared.
        int, float, bool, str or None
            The positive label used in the different reports.
        """
        if not isinstance(reports, Iterable):
            raise TypeError(
                f"Expected reports to be a list or dict; got {type(reports)}"
            )

        if len(reports) < 2:
            raise ValueError(
                f"Expected at least 2 reports to compare; got {len(reports)}"
            )

        if isinstance(reports, list):
            report_names = None
            reports_list = reports
        else:  # dict
            report_names = list(reports.keys())
            for key in report_names:
                if not isinstance(key, str):
                    raise TypeError(
                        f"Expected all report names to be strings; got {type(key)}"
                    )
            reports_list = cast(
                list[EstimatorReport] | list[CrossValidationReport],
                list(reports.values()),
            )

        reports_type: ReportType
        if all(isinstance(report, EstimatorReport) for report in reports_list):
            reports_list = cast(list[EstimatorReport], reports_list)
            reports_type = "EstimatorReport"

            test_dataset_hashes = {
                joblib.hash(report.y_test)
                for report in reports_list
                if report.y_test is not None
            }
            if len(test_dataset_hashes) > 1:
                raise ValueError(
                    "Expected all estimators to share the same test targets."
                )

        elif all(isinstance(report, CrossValidationReport) for report in reports_list):
            reports_list = cast(list[CrossValidationReport], reports_list)
            reports_type = "CrossValidationReport"
        else:
            raise TypeError(
                f"Expected list or dict of {EstimatorReport.__name__} "
                f"or list of dict of {CrossValidationReport.__name__}"
            )

        if len({id(report) for report in reports_list}) < len(reports_list):
            raise ValueError("Expected reports to be distinct objects")

        ml_tasks = {report._ml_task for report in reports_list}
        if len(ml_tasks) > 1:
            raise ValueError(
                f"Expected all estimators to have the same ML usecase; got {ml_tasks}"
            )

        if ml_tasks == {"binary-classification"}:
            pos_labels = {report.pos_label for report in reports_list}
            if len(pos_labels) > 1:
                raise ValueError(
                    "Expected all estimators to have the same positive label. "
                    f"Got {pos_labels}."
                )
            pos_label = pos_labels.pop()
        else:
            pos_label = None

        if report_names is None:
            deduped_report_names = _deduplicate_report_names(
                [report.estimator_name_ for report in reports_list]
            )
        else:
            deduped_report_names = report_names

        reports_dict = cast(
            dict[str, EstimatorReport] | dict[str, CrossValidationReport],
            dict(zip(deduped_report_names, reports_list, strict=True)),
        )

        return reports_dict, reports_type, pos_label

    def __init__(
        self,
        reports: list[EstimatorReport]
        | dict[str, EstimatorReport]
        | list[CrossValidationReport]
        | dict[str, CrossValidationReport],
        *,
        n_jobs: int | None = None,
    ) -> None:
        """
        ComparisonReport instance initializer.

        Notes
        -----
        We check that the estimator reports can be compared:
        - all reports are estimator reports,
        - all estimators are in the same ML use case,
        - all estimators have non-empty X_test and y_test,
        - all estimators have the same X_test and y_test.
        """
        self.reports_, self._reports_type, self._pos_label = (
            ComparisonReport._validate_reports(reports)
        )

        self._progress_info: dict[str, Any] | None = None

        self.n_jobs = n_jobs
        self._rng = np.random.default_rng(time.time_ns())
        self._hash = self._rng.integers(
            low=np.iinfo(np.int64).min, high=np.iinfo(np.int64).max
        )
        self._cache = Cache()
        self._ml_task = next(iter(self.reports_.values()))._ml_task  # type: ignore

    def clear_cache(self) -> None:
        """Clear the cache.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
        >>> report.cache_predictions()
        >>> report.clear_cache()
        >>> report._cache
        {}
        """
        for report in self.reports_.values():
            report.clear_cache()

        self._cache = Cache()

    @progress_decorator(description="Estimator predictions")
    def cache_predictions(
        self,
        response_methods: Literal[
            "auto", "predict", "predict_proba", "decision_function"
        ] = "auto",
        n_jobs: int | None = None,
    ) -> None:
        """Cache the predictions for sub-estimators reports.

        Parameters
        ----------
        response_methods : {"auto", "predict", "predict_proba", "decision_function"},\
                default="auto"
            The methods to use to compute the predictions.

        n_jobs : int, default=None
            The number of jobs to run in parallel. If `None`, we use the `n_jobs`
            parameter when initializing the report.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
        >>> report.cache_predictions()
        >>> report._cache
        {...}
        """
        if n_jobs is None:
            n_jobs = self.n_jobs

        assert self._progress_info is not None, (
            "The rich Progress class was not initialized."
        )
        progress = self._progress_info["current_progress"]
        main_task = self._progress_info["current_task"]

        total_estimators = len(self.reports_)
        progress.update(main_task, total=total_estimators)

        for report in self.reports_.values():
            # Share the parent's progress bar with child report
            report._progress_info = {"current_progress": progress}
            report.cache_predictions(response_methods=response_methods, n_jobs=n_jobs)
            progress.update(main_task, advance=1, refresh=True)

    def get_predictions(
        self,
        *,
        data_source: Literal["train", "test", "X_y"],
        response_method: Literal[
            "predict", "predict_proba", "decision_function"
        ] = "predict",
        X: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> list[ArrayLike] | list[list[ArrayLike]]:
        """Get predictions from the underlying reports.

        This method has the advantage to reload from the cache if the predictions
        were already computed in a previous call.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        response_method : {"predict", "predict_proba", "decision_function"}, \
                default="predict"
            The response method to use to get the predictions.

        X : array-like of shape (n_samples, n_features), default=None
            When `data_source` is "X_y", the input features on which to compute the
            response method.

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing predictions in
            binary classification cases. By default, the positive class is set to the
            one provided when creating the report. If `None`, `estimator_.classes_[1]`
            is used as positive label.

            When `pos_label` is equal to `estimator_.classes_[0]`, it will be equivalent
            to `estimator_.predict_proba(X)[:, 0]` for `response_method="predict_proba"`
            and `-estimator_.decision_function(X)` for
            `response_method="decision_function"`.

        Returns
        -------
        list of np.ndarray of shape (n_samples,) or (n_samples, n_classes) or list of \
                such lists
            The predictions for each :class:`~skore.EstimatorReport` or
            :class:`~skore.CrossValidationReport`.

        Raises
        ------
        ValueError
            If the data source is invalid.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from skore import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> report = ComparisonReport([estimator_report_1, estimator_report_2])
        >>> report.cache_predictions()
        >>> predictions = report.get_predictions(data_source="test")
        >>> print([split_predictions.shape for split_predictions in predictions])
        [(25,), (25,)]
        """
        return [  # type: ignore
            report.get_predictions(
                data_source=data_source,
                response_method=response_method,
                X=X,
                pos_label=pos_label,
            )
            for report in self.reports_.values()
        ]

    @property
    def pos_label(self) -> PositiveLabel | None:
        return self._pos_label

    def get_best_model(
        self,
        *,
        data_source: Literal["train", "test", "X_y"] = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        metric: Metric | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        response_method: str | list[str] | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
        aggregate: str | None = "mean",
    ) -> EstimatorReport | CrossValidationReport:
        """Get the best model from the comparison based on a metric.

        The best model is determined by computing the specified metric for all
        models and selecting the one with the best value (highest or lowest
        depending on the metric).

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use for computing the metric.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            Data on which to compute the metric when `data_source="X_y"`.

        y : array-like of shape (n_samples,), default=None
            Target on which to compute the metric when `data_source="X_y"`.

        metric : str or callable, default=None
            The metric to use for comparison. If None, a default metric is
            automatically selected based on the machine learning task:

            - For classification tasks: "accuracy"
            - For regression tasks: "r2"

            Valid metrics include: "accuracy", "precision", "recall", "roc_auc",
            "log_loss", "brier_score", "r2", "rmse", and any custom metric
            accessible via the metrics accessor.

            Can also be a callable with signature ``metric(y_true, y_pred)`` that
            returns a scalar score; in this case it is assumed that higher values are
            better, as per the scikit-learn convention.

        metric_kwargs : dict
            The keyword arguments to pass to the metric functions.

        response_method : {"predict", "predict_proba", "predict_log_proba", \
            "decision_function"} or list of such str, default=None
            The estimator's method to be invoked to get the predictions. Only necessary
            for custom metrics.

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        aggregate : str or None, default="mean"
            The aggregation function to use across cross-validation splits.
            Only valid for `CrossValidationReport` comparisons, ignored when comparison
            is between `EstimatorReport` instances.

        Returns
        -------
        EstimatorReport or CrossValidationReport
            The report object corresponding to the best model according to the
            specified metric.

        Raises
        ------
        ValueError
            If the metric computation results in invalid data or if the metric
            is not available for the given task.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from skore import ComparisonReport, EstimatorReport, train_test_split
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> report_1 = EstimatorReport(LogisticRegression(), **split_data)
        >>> report_2 = EstimatorReport(DecisionTreeClassifier(), **split_data)
        >>> report = ComparisonReport([report_1, report_2])
        >>> report.get_best_model()
        EstimatorReport(...)

        Using a custom metric:

        >>> report.get_best_model(metric="roc_auc")
        EstimatorReport(...)
        """
        # TODO: We can probably reuse the method in EstimatorReport.metrics.summarize()
        if metric is None:
            if "classification" in self._ml_task:
                metric = "accuracy"
            elif "regression" in self._ml_task:
                metric = "r2"
            else:
                raise ValueError(
                    f"Cannot infer default metric for ML task '{self._ml_task}'. "
                    "Please specify a metric explicitly."
                )

        metrics_display = self.metrics.summarize(
            metric=metric,
            data_source=data_source,
            X=X,
            y=y,
            response_method=response_method,
            pos_label=pos_label,
            favorability=True,
            aggregate=aggregate,
            metric_kwargs=metric_kwargs,
        )

        results = metrics_display.frame()

        favorability = results["Favorability"].iloc[0]
        results = results.drop(columns=["Favorability"])

        # The columns can be multi-level e.g. ("mean", model_name)
        if isinstance(results.columns, pd.MultiIndex):
            results.columns = results.columns.droplevel(0)

        if isinstance(results.index, pd.MultiIndex):
            # For metrics like precision or recall with multiple labels,
            # we average across labels
            comparison_values = results.mean(axis=0)
        else:
            comparison_values = results.iloc[0]

        best_model_name = (
            comparison_values.idxmin()
            if favorability == "(↘︎)"
            else comparison_values.idxmax()
        )

        # Return the corresponding report
        return self.reports_[best_model_name]

    ####################################################################################
    # Methods related to the help and repr
    ####################################################################################

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Tools to compare estimators[/bold cyan]"

    def _get_help_legend(self) -> str:
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"{self.__class__.__name__}(...)"


def _deduplicate_report_names(report_names: list[str]) -> list[str]:
    """De-duplicate report names that appear several times.

    Leave the other report names alone.

    Parameters
    ----------
    report_names : list of str
        The list of report names to be checked.

    Returns
    -------
    list of str
        The de-duplicated list of report names.

    Examples
    --------
    >>> _deduplicate_report_names(['a', 'b'])
    ['a', 'b']
    >>> _deduplicate_report_names(['a', 'a'])
    ['a_1', 'a_2']
    >>> _deduplicate_report_names(['a', 'b', 'a'])
    ['a_1', 'b', 'a_2']
    >>> _deduplicate_report_names(['a', 'b', 'a', 'b'])
    ['a_1', 'b_1', 'a_2', 'b_2']
    >>> _deduplicate_report_names([])
    []
    >>> _deduplicate_report_names(['a'])
    ['a']
    """
    counts = Counter(report_names)
    if len(report_names) == len(counts):
        return report_names

    names = report_names.copy()
    seen: Counter = Counter()
    for i in range(len(names)):
        name = names[i]
        seen[name] += 1
        if counts[name] > 1:
            names[i] = f"{name}_{seen[name]}"
    return names
