from collections.abc import Callable
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import make_scorer
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import (
    _BaseAccessor,
    _BaseMetricsAccessor,
    _get_cached_response_values,
)
from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._plot.metrics import (
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.types import (
    _DEFAULT,
    Aggregate,
    Metric,
    PositiveLabel,
    YPlotData,
)
from skore._utils._accessor import (
    _check_any_sub_report_has_metric,
    _check_supported_ml_task,
)
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._index import flatten_multi_index
from skore._utils._progress_bar import progress_decorator

from .utils import _combine_cross_validation_results, _combine_estimator_results

DataSource = Literal["test", "train", "X_y"]


class _MetricsAccessor(_BaseMetricsAccessor, _BaseAccessor, DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def __init__(self, parent: ComparisonReport) -> None:
        super().__init__(parent)

    def summarize(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        metric: Metric | list[Metric] | dict[str, Metric] | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
        indicator_favorability: bool = False,
        flat_index: bool = False,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for the estimators.

        Parameters
        ----------
        data_source : {"test", "train", "X_y", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        metric : str, callable, scorer, or list of such instances or dict of such \
            instances, default=None
            The metrics to report. The possible values (whether or not in a list) are:

            - if a string, either one of the built-in metrics or a scikit-learn scorer
              name. You can get the possible list of string using
              `report.metrics.help()` or :func:`sklearn.metrics.get_scorer_names` for
              the built-in metrics or the scikit-learn scorers, respectively.
            - if a callable, it should take as arguments `y_true`, `y_pred` as the two
              first arguments. Additional arguments can be passed as keyword arguments
              and will be forwarded with `metric_kwargs`. No favorability indicator can
              be displayed in this case.
            - if the callable API is too restrictive (e.g. need to pass
              same parameter name with different values), you can use scikit-learn
              scorers as provided by :func:`sklearn.metrics.make_scorer`. In this case,
              the metric favorability will only be displayed if it is given explicitly
              via `make_scorer`'s `greater_is_better` parameter.

        metric_kwargs : dict, default=None
            The keyword arguments to pass to the metric functions.

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        indicator_favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=False
            Whether to flatten the `MultiIndex` columns. Flat index will always be lower
            case, do not include spaces and remove the hash symbol to ease indexing.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        MetricsSummaryDisplay
            A display containing the statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.summarize(
        ...     metric=["precision", "recall"],
        ...     pos_label=1,
        ... ).frame()
        Estimator       LogisticRegression_1  LogisticRegression_2
        Metric
        Precision                    0.96...               0.96...
        Recall                       0.97...               0.97...
        """
        results = self._compute_metric_scores(
            report_metric_name="summarize",
            data_source=data_source,
            X=X,
            y=y,
            metric=metric,
            pos_label=pos_label,
            metric_kwargs=metric_kwargs,
            indicator_favorability=indicator_favorability,
            aggregate=aggregate,
        )
        if flat_index:
            if isinstance(results.columns, pd.MultiIndex):
                results.columns = flatten_multi_index(results.columns)
            if isinstance(results.index, pd.MultiIndex):
                results.index = flatten_multi_index(results.index)
            if isinstance(results.index, pd.Index):
                results.index = results.index.str.replace(
                    r"\((.*)\)$", r"\1", regex=True
                )
        return MetricsSummaryDisplay(results)

    @progress_decorator(description="Compute metric for each estimator")
    def _compute_metric_scores(
        self,
        report_metric_name: str,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
        **metric_kwargs: Any,
    ):
        # build the cache key components to finally create a tuple that will be used
        # to check if the metric has already been computed
        cache_key_parts: list[Any] = [
            self._parent._hash,
            report_metric_name,
            data_source,
        ]

        if self._parent._reports_type == "CrossValidationReport":
            if aggregate is None:
                cache_key_parts.append(aggregate)
            else:
                cache_key_parts.extend(tuple(aggregate))

        # we need to enforce the order of the parameter for a specific metric
        # to make sure that we hit the cache in a consistent way
        ordered_metric_kwargs = sorted(metric_kwargs.keys())
        for key in ordered_metric_kwargs:
            if isinstance(metric_kwargs[key], np.ndarray | list | dict):
                cache_key_parts.append(joblib.hash(metric_kwargs[key]))
            else:
                cache_key_parts.append(metric_kwargs[key])

        cache_key = tuple(cache_key_parts)

        assert self._parent._progress_info is not None, "Progress info not set"
        progress = self._parent._progress_info["current_progress"]
        main_task = self._parent._progress_info["current_task"]

        total_estimators = len(self._parent.reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            results = self._parent._cache[cache_key]
        else:
            parallel = joblib.Parallel(
                **_validate_joblib_parallel_params(
                    n_jobs=self._parent.n_jobs, return_as="generator"
                )
            )

            kwargs = dict(
                data_source=data_source,
                X=X,
                y=y,
                **metric_kwargs,
            )
            if self._parent._reports_type == "CrossValidationReport":
                kwargs["aggregate"] = None

            generator = parallel(
                joblib.delayed(getattr(report.metrics, report_metric_name))(**kwargs)
                for report in self._parent.reports_.values()
            )
            individual_results = []
            for result in generator:
                if report_metric_name == "summarize":
                    # for summarize, the output is a display
                    individual_results.append(result.frame())
                else:
                    individual_results.append(result)
                progress.update(main_task, advance=1, refresh=True)

            if self._parent._reports_type == "EstimatorReport":
                results = _combine_estimator_results(
                    individual_results,
                    estimator_names=self._parent.reports_.keys(),
                    indicator_favorability=metric_kwargs.get(
                        "indicator_favorability", False
                    ),
                    data_source=data_source,
                )
            else:  # "CrossValidationReport"
                results = _combine_cross_validation_results(
                    individual_results,
                    estimator_names=self._parent.reports_.keys(),
                    indicator_favorability=metric_kwargs.get(
                        "indicator_favorability", False
                    ),
                    aggregate=aggregate,
                )

            self._parent._cache[cache_key] = results
        return results

    def timings(
        self,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Get all measured processing times related to the different estimators.

        The index of the returned dataframe is the name of the processing time. When
        the estimators were not used to predict, no timings regarding the prediction
        will be present.

        Parameters
        ----------
        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            A dataframe with the processing times.

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
        >>> report = ComparisonReport(
        ...     {"model1": estimator_report_1, "model2": estimator_report_2}
        ... )
        >>> report.metrics.timings()
                        model1    model2
        Fit time (s)       ...       ...
        >>> report.cache_predictions(response_methods=["predict"])
        >>> report.metrics.timings()
                                model1    model2
        Fit time (s)               ...       ...
        Predict time test (s)      ...       ...
        Predict time train (s)     ...       ...
        """
        if self._parent._reports_type == "EstimatorReport":
            timings: pd.DataFrame = pd.concat(
                [
                    pd.Series(report.metrics.timings())
                    for report in self._parent.reports_.values()
                ],
                axis=1,
                keys=self._parent.reports_.keys(),
            )
            timings.index = timings.index.str.replace("_", " ").str.capitalize()

            # Add (s) to time measurements
            new_index = [f"{idx} (s)" for idx in timings.index]

            timings.index = pd.Index(new_index)

            return timings

        else:  # "CrossValidationReport"
            results = [
                report.metrics.timings(aggregate=None)
                for report in self._parent.reports_.values()
            ]

            # Put dataframes in the right shape
            for i, result in enumerate(results):
                result.index.name = "Metric"
                result.columns = pd.MultiIndex.from_product(
                    [[list(self._parent.reports_.keys())[i]], result.columns]
                )

            timings = _combine_cross_validation_results(
                results,
                self._parent.reports_.keys(),
                indicator_favorability=False,
                aggregate=aggregate,
            )
            return timings

    @available_if(_check_any_sub_report_has_metric("accuracy"))
    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.accuracy()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        Accuracy                    0.96...               0.96...
        """
        return self.summarize(
            metric=["accuracy"],
            data_source=data_source,
            X=X,
            y=y,
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("precision"))
    def precision(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        pos_label: PositiveLabel | None = _DEFAULT,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        average : {"binary", "macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by `pos_label`.
              This is applicable only if targets (`y_{true,pred}`) are binary.
            - "micro": Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an F-score
              that is not between precision and recall.
            - "samples": Calculate metrics for each instance, and find their average
              (only meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).

            .. note::
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.precision()
        Estimator                    LogisticRegression_1  LogisticRegression_2
        Metric      Label / Average
        Precision                 0               0.96...               0.96...
                                  1               0.96...               0.96...
        """
        return self.summarize(
            metric=["precision"],
            data_source=data_source,
            X=X,
            y=y,
            pos_label=pos_label,
            metric_kwargs={"average": average},
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("recall"))
    def recall(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        pos_label: PositiveLabel | None = _DEFAULT,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the recall score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by `pos_label`.
              This is applicable only if targets (`y_{true,pred}`) are binary.
            - "micro": Calculate metrics globally by counting the total true positives,
              false negatives and false positives.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean.  This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average
              weighted by support (the number of true instances for each label). This
              alters 'macro' to account for label imbalance; it can result in an F-score
              that is not between precision and recall. Weighted recall is equal to
              accuracy.
            - "samples": Calculate metrics for each instance, and find their average
              (only meaningful for multilabel classification where this differs from
              :func:`accuracy_score`).

            .. note::
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.recall()
        Estimator                    LogisticRegression_1  LogisticRegression_2
        Metric      Label / Average
        Recall                    0              0.944...              0.944...
                                  1              0.977...              0.977...
        """
        return self.summarize(
            metric=["recall"],
            data_source=data_source,
            X=X,
            y=y,
            pos_label=pos_label,
            metric_kwargs={"average": average},
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("brier_score"))
    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.brier_score()
        Estimator         LogisticRegression_1  LogisticRegression_2
        Metric
        Brier score                   0.025...              0.025...
        """
        return self.summarize(
            metric=["brier_score"],
            data_source=data_source,
            X=X,
            y=y,
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("roc_auc"))
    def roc_auc(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: Literal["auto", "macro", "micro", "weighted", "samples"] | None = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the ROC AUC score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        average : {"auto", "macro", "micro", "weighted", "samples"}, \
                default=None
            Average to compute the ROC AUC score in a multiclass setting. By default,
            no average is computed. Otherwise, this determines the type of averaging
            performed on the data.

            - "micro": Calculate metrics globally by considering each element of
              the label indicator matrix as a label.
            - "macro": Calculate metrics for each label, and find their unweighted
              mean. This does not take label imbalance into account.
            - "weighted": Calculate metrics for each label, and find their average,
              weighted by support (the number of true instances for each label).
            - "samples": Calculate metrics for each instance, and find their
              average.

            .. note::
                Multiclass ROC AUC currently only handles the "macro" and
                "weighted" averages. For multiclass targets, `average=None` is only
                implemented for `multi_class="ovr"` and `average="micro"` is only
                implemented for `multi_class="ovr"`.

        multi_class : {"raise", "ovr", "ovo"}, default="ovr"
            The multi-class strategy to use.

            - "raise": Raise an error if the data is multiclass.
            - "ovr": Stands for One-vs-rest. Computes the AUC of each class against the
              rest. This treats the multiclass case in the same way as the multilabel
              case. Sensitive to class imbalance even when `average == "macro"`,
              because class imbalance affects the composition of each of the "rest"
              groupings.
            - "ovo": Stands for One-vs-one. Computes the average AUC of all possible
              pairwise combinations of classes. Insensitive to class imbalance when
              `average == "macro"`.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.roc_auc()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        ROC AUC                     0.99...               0.99...
        """
        return self.summarize(
            metric=["roc_auc"],
            data_source=data_source,
            X=X,
            y=y,
            metric_kwargs={"average": average, "multi_class": multi_class},
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("log_loss"))
    def log_loss(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the log loss.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.log_loss()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        Log loss                   0.082...              0.082...
        """
        return self.summarize(
            metric=["log_loss"],
            data_source=data_source,
            X=X,
            y=y,
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("r2"))
    def r2(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the R² score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        multioutput : {"raw_values", "uniform_average"} or array-like of shape \
                (n_outputs,), default="raw_values"
            Defines aggregating of multiple output values. Array-like value defines
            weights used to average errors. The other possible values are:

            - "raw_values": Returns a full set of errors in case of multioutput input.
            - "uniform_average": Errors of all outputs are averaged with uniform weight.

            By default, no averaging is done.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = Ridge(random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.r2()
        Estimator     Ridge_1    Ridge_2
        Metric
        R²            0.43...    0.43...
        """
        return self.summarize(
            metric=["r2"],
            data_source=data_source,
            X=X,
            y=y,
            metric_kwargs={"multioutput": multioutput},
            aggregate=aggregate,
        ).frame()

    @available_if(_check_any_sub_report_has_metric("rmse"))
    def rmse(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the root mean squared error.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        multioutput : {"raw_values", "uniform_average"} or array-like of shape \
                (n_outputs,), default="raw_values"
            Defines aggregating of multiple output values. Array-like value defines
            weights used to average errors. The other possible values are:

            - "raw_values": Returns a full set of errors in case of multioutput input.
            - "uniform_average": Errors of all outputs are averaged with uniform weight.

            By default, no averaging is done.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = Ridge(random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.rmse()
        Estimator       Ridge_1       Ridge_2
        Metric
        RMSE          55.726...     55.726...
        """
        return self.summarize(
            metric=["rmse"],
            data_source=data_source,
            X=X,
            y=y,
            metric_kwargs={"multioutput": multioutput},
            aggregate=aggregate,
        ).frame()

    def custom_metric(
        self,
        metric_function: Callable,
        response_method: str | list[str],
        *,
        metric_name: str | None = None,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute a custom metric.

        It brings some flexibility to compute any desired metric. However, we need to
        follow some rules:

        - `metric_function` should take `y_true` and `y_pred` as the first two
          positional arguments.
        - `response_method` corresponds to the estimator's method to be invoked to get
          the predictions. It can be a string or a list of strings to defined in which
          order the methods should be invoked.

        Parameters
        ----------
        metric_function : callable
            The metric function to be computed. The expected signature is
            `metric_function(y_true, y_pred, **kwargs)`.

        response_method : str or list of str
            The estimator's method to be invoked to get the predictions. The possible
            values are: `predict`, `predict_proba`, `predict_log_proba`, and
            `decision_function`.

        metric_name : str, default=None
            The name of the metric. If not provided, it will be inferred from the
            metric function.

        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        **kwargs : dict
            Any additional keyword arguments to be passed to the metric function.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The custom metric.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_absolute_error
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = Ridge(random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> comparison_report.metrics.custom_metric(
        ...     metric_function=mean_absolute_error,
        ...     response_method="predict",
        ...     metric_name="MAE",
        ... )
        Estimator      Ridge_1      Ridge_2
        Metric
        MAE           45.91...     45.91...
        """
        # create a scorer with `greater_is_better=True` to not alter the output of
        # `metric_function`
        scorer = make_scorer(
            metric_function,
            greater_is_better=True,
            response_method=response_method,
            **kwargs,
        )
        scoring = {metric_name: scorer} if metric_name is not None else [scorer]
        return self.summarize(
            metric=scoring,
            data_source=data_source,
            X=X,
            y=y,
            aggregate=aggregate,
        ).frame()

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _get_methods_for_help(self) -> list[tuple[str, Callable]]:
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.ComparisonReport.metrics")

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    @progress_decorator(description="Computing predictions for display")
    def _get_display(
        self,
        *,
        X: ArrayLike | None,
        y: ArrayLike | None,
        data_source: DataSource,
        response_method: str | list[str],
        display_class: type[
            RocCurveDisplay | PrecisionRecallCurveDisplay | PredictionErrorDisplay
        ],
        display_kwargs: dict[str, Any],
    ) -> RocCurveDisplay | PrecisionRecallCurveDisplay | PredictionErrorDisplay:
        """Get the display from the cache or compute it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.

        y : array-like of shape (n_samples,)
            The target.

        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        response_method : str
            The response method.

        display_class : class
            The display class.

        display_kwargs : dict
            The display kwargs used by `display_class._from_predictions`.

        Returns
        -------
        display : display_class
            The display.
        """
        if "seed" in display_kwargs and display_kwargs["seed"] is None:
            cache_key = None
        else:
            # build the cache key components to finally create a tuple that will be used
            # to check if the metric has already been computed
            cache_key_parts: list[Any] = [self._parent._hash, display_class.__name__]
            cache_key_parts.extend(display_kwargs.values())
            cache_key_parts.append(data_source)
            cache_key = tuple(cache_key_parts)

        assert self._parent._progress_info is not None, "Progress info not set"
        progress = self._parent._progress_info["current_progress"]
        main_task = self._parent._progress_info["current_task"]
        total_estimators = len(self._parent.reports_)
        progress.update(main_task, total=total_estimators)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
        else:
            y_true: list[YPlotData] = []
            y_pred: list[YPlotData] = []

            if self._parent._reports_type == "EstimatorReport":
                for report_name, report in self._parent.reports_.items():
                    report_X, report_y, _ = (
                        report.metrics._get_X_y_and_data_source_hash(
                            data_source=data_source,
                            X=X,
                            y=y,
                        )
                    )

                    y_true.append(
                        YPlotData(
                            estimator_name=report_name,
                            data_source=data_source,
                            split=None,
                            y=report_y,
                        )
                    )
                    results = _get_cached_response_values(
                        cache=report._cache,
                        estimator_hash=report._hash,
                        estimator=report._estimator,
                        X=report_X,
                        response_method=response_method,
                        data_source=data_source,
                        data_source_hash=None,
                        pos_label=display_kwargs.get("pos_label"),
                    )
                    for key, value, is_cached in results:
                        if not is_cached:
                            report._cache[key] = value
                        if key[-1] != "predict_time":
                            y_pred.append(
                                YPlotData(
                                    estimator_name=report_name,
                                    data_source=data_source,
                                    split=None,
                                    y=value,
                                )
                            )

                    progress.update(main_task, advance=1, refresh=True)

                display = display_class._compute_data_for_display(
                    y_true=y_true,
                    y_pred=y_pred,
                    report_type="comparison-estimator",
                    estimators=[
                        report.estimator_ for report in self._parent.reports_.values()
                    ],
                    ml_task=self._parent._ml_task,
                    data_source=data_source,
                    **display_kwargs,
                )

            else:
                for report_name, report in self._parent.reports_.items():
                    for split, estimator_report in enumerate(report.estimator_reports_):
                        report_X, report_y, _ = (
                            estimator_report.metrics._get_X_y_and_data_source_hash(
                                data_source=data_source,
                                X=X,
                                y=y,
                            )
                        )

                        y_true.append(
                            YPlotData(
                                estimator_name=report_name,
                                data_source=data_source,
                                split=split,
                                y=report_y,
                            )
                        )

                        results = _get_cached_response_values(
                            cache=estimator_report._cache,
                            estimator_hash=estimator_report._hash,
                            estimator=estimator_report.estimator_,
                            X=report_X,
                            response_method=response_method,
                            data_source=data_source,
                            data_source_hash=None,
                            pos_label=display_kwargs.get("pos_label"),
                        )
                        for key, value, is_cached in results:
                            if not is_cached:
                                report._cache[key] = value
                            if key[-1] != "predict_time":
                                y_pred.append(
                                    YPlotData(
                                        estimator_name=report_name,
                                        data_source=data_source,
                                        split=split,
                                        y=value,
                                    )
                                )

                    progress.update(main_task, advance=1, refresh=True)

                display = display_class._compute_data_for_display(
                    y_true=y_true,
                    y_pred=y_pred,
                    report_type="comparison-cross-validation",
                    estimators=[
                        estimator_report.estimator_
                        for report in self._parent.reports_.values()
                        for estimator_report in report.estimator_reports_
                    ],
                    ml_task=self._parent._ml_task,
                    data_source=data_source,
                    **display_kwargs,
                )

            if cache_key is not None:
                # Unless seed is an int (i.e. the call is deterministic),
                # we do not cache
                self._parent._cache[cache_key] = display

        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> display = comparison_report.metrics.roc()
        >>> display.plot()
        """
        if pos_label == _DEFAULT:
            pos_label = self._parent.pos_label

        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display = cast(
            RocCurveDisplay,
            self._get_display(
                X=X,
                y=y,
                data_source=data_source,
                response_method=response_method,
                display_class=RocCurveDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision_recall(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> PrecisionRecallCurveDisplay:
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        Returns
        -------
        PrecisionRecallCurveDisplay
            The precision-recall curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> display = comparison_report.metrics.precision_recall()
        >>> display.plot()
        """
        if pos_label == _DEFAULT:
            pos_label = self._parent.pos_label

        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        display = cast(
            PrecisionRecallCurveDisplay,
            self._get_display(
                X=X,
                y=y,
                data_source=data_source,
                response_method=response_method,
                display_class=PrecisionRecallCurveDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def prediction_error(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        subsample: int = 1_000,
        seed: int | None = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            display on the scatter plot. If `None`, no subsampling will be
            applied. by default, 1,000 samples or less will be displayed.

        seed : int, default=None
            The seed used to initialize the random number generator used for the
            subsampling.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_report_1 = EstimatorReport(estimator_1, **split_data)
        >>> estimator_2 = Ridge(random_state=43)
        >>> estimator_report_2 = EstimatorReport(estimator_2, **split_data)
        >>> comparison_report = ComparisonReport(
        ...     [estimator_report_1, estimator_report_2]
        ... )
        >>> display = comparison_report.metrics.prediction_error()
        >>> display.plot(kind="actual_vs_predicted")
        """
        display_kwargs = {"subsample": subsample, "seed": seed}
        display = cast(
            PredictionErrorDisplay,
            self._get_display(
                X=X,
                y=y,
                data_source=data_source,
                response_method="predict",
                display_class=PredictionErrorDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display
