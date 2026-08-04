from __future__ import annotations

import numbers
import warnings
from typing import Any, Literal

import pandas as pd
from joblib import Parallel
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import BaseMetricsAccessor, _summarize_report_metrics
from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._plot.metrics import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.metrics import Metric, MetricLike
from skore._sklearn.types import Aggregate
from skore._utils._accessor import (
    _check_any_sub_report_has_metric,
    _check_supported_ml_task,
)
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._parallel import delayed
from skore._utils._progress_bar import track

DataSource = Literal["test", "train", "both"]


class _MetricsAccessor(BaseMetricsAccessor[ComparisonReport], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def __init__(self, parent: ComparisonReport) -> None:
        super().__init__(parent)

    def summarize(
        self,
        *,
        data_source: DataSource = "test",
        metric: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for the estimators.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

        metric : str or list of str or None, default=None
            The metrics to report, from the list of registered metrics. None means show
            all registered metrics. To add a custom metric, see :meth:`add`.

        Returns
        -------
        :class:`MetricsSummaryDisplay`
            A display containing the statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate(
        ...     [estimator_1, estimator_2], X, y, splitter=0.2, pos_label=1
        ... )
        >>> comparison_report.metrics.summarize(metric=["precision", "recall"]).frame()
        estimator  LogisticRegression_1  LogisticRegression_2
        metric
        precision              0.98...              0.98...
        recall                 0.92...              0.92...
        """
        parallel = Parallel(
            **_validate_joblib_parallel_params(
                n_jobs=self._parent.n_jobs, return_as="generator"
            )
        )

        summaries = list(
            track(
                parallel(
                    delayed(_summarize_report_metrics)(
                        report,
                        data_source=data_source,
                        metric=metric,
                    )
                    for report in self._parent.reports_.values()
                ),
                description="Compute metric for each estimator",
                total=len(self._parent.reports_),
            )
        )

        extra_rows_data = [
            {"estimator": estimator_name} for estimator_name in self._parent.reports_
        ]
        return MetricsSummaryDisplay._concatenate(
            summaries,
            report_type=self._parent._report_type,
            extra_rows_data=extra_rows_data,
        )

    def _formatted_summary_frame(
        self,
        *,
        data_source: DataSource = "test",
        metric: str | list[str] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Wide metric summary frame used for accessor display.

        Comparison reports always return a :class:`pandas.DataFrame` with one
        column per compared estimator.
        """
        frame = self.summarize(data_source=data_source, metric=metric).frame(
            flat_index=False,
            verbose_name=True,
        )
        frame = frame.rename_axis(
            None
            if self._parent._report_type == "comparison-estimator"
            else [None, None],
            axis="columns",
        )
        if self._parent._report_type == "comparison-cross-validation":
            frame = frame.swaplevel(axis="columns")
        return frame

    def _repr_html_(self) -> str:
        frame = self._formatted_summary_frame()
        html = (
            frame.to_frame()._repr_html_()
            if isinstance(frame, pd.Series)
            else frame._repr_html_()
        )
        return (
            "<p>Metrics summary:</p>"
            f"{html}"
            '<p role="note">Explore available methods with '
            "<code>.help()</code>.</p>"
        )

    def _metric(
        self, metric_name: str, *, data_source: DataSource, **kwargs: Any
    ) -> MetricsSummaryDisplay:
        """Compute a single metric across compared reports, forwarding *kwargs*."""
        summaries = [
            report.metrics._metric(metric_name, data_source=data_source, **kwargs)
            for report in self._parent.reports_.values()
        ]

        extra_rows_data = [
            {"estimator": estimator_name} for estimator_name in self._parent.reports_
        ]
        return MetricsSummaryDisplay._concatenate(
            summaries,
            report_type=self._parent._report_type,
            extra_rows_data=extra_rows_data,
        )

    def available(self, *, report_name: str | None = None) -> list[str]:
        """List available metric names in the registry.

        Parameters
        ----------
        report_name : str, default=None
            Name of the sub-report to list metrics from. If `None`, returns the
            union of metric names across all sub-reports.

        Returns
        -------
        list[str]
            The list of available metric names.
        """
        reports = self._parent.reports_
        if report_name is not None:
            if report_name not in reports:
                valid_names = ", ".join(reports)
                raise ValueError(
                    f"Unknown report name: {report_name!r}. "
                    f"Available report names are: {valid_names}."
                )
            return reports[report_name].metrics.available()

        keys = dict.fromkeys(
            metric
            for report in reports.values()
            for metric in report.metrics.available()
        )
        return list(keys)

    def _resolve_metric(self, name: str) -> Metric | None:
        """Return the :class:`~skore._sklearn.metrics.Metric` for ``name``, or None."""
        for report in self._parent.reports_.values():
            metric = report.metrics._resolve_metric(name)
            if metric is not None:
                return metric
        return None

    def add(
        self,
        metric: MetricLike,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool = True,
        position: Literal["first", "last"] = "first",
        **kwargs: Any,
    ) -> None:
        """Add a custom metric to :meth:`summarize`.

        Parameters
        ----------
        metric : str, sklearn scorer, or callable
            The metric to add.

            - If a string, it will be run through :func:`sklearn.metrics.get_scorer`.
              Metrics that require a ``neg_`` prefix (e.g. ``"neg_mean_squared_error"``)
              may also be passed without it (e.g. ``"mean_squared_error"``); the alias
              is resolved automatically.
            - If a callable, it must have the signature
              ``(estimator, X, y_true, **kw) -> float``. It may also return a ``dict``
              mapping class labels to floats (e.g. ``{0: 0.9, 1: 0.85}``), in which case
              :meth:`summarize` will show one row per class label under the metric name.
              If your metric has the form ``(y_true, y_pred, **kw) -> float``, see
              :func:`sklearn.metrics.make_scorer` to convert it to a scorer.

        name : str or None, default=None
            Custom name for the metric. If ``None``, the name is inferred
            from the metric (e.g. the function's ``__name__``).

        verbose_name : str or None, default=None
            Custom verbose name for the metric which will be used for display
            purposes. If ``None``, the verbose name is inferred from the metric
            name.

        greater_is_better : bool, default=True
            Whether higher values are better (only for callables).

        position : {"first", "last"}, default="first"
            Where to place the metric in default :meth:`summarize` ordering
            for each compared report. See :meth:`EstimatorReport.metrics.add`.

        **kwargs : Any
            Default keyword arguments passed to the score function at call
            time.  Only used when *metric* is a plain callable.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.metrics import make_scorer, mean_absolute_error
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10_000)
        >>> estimator_2 = LogisticRegression(max_iter=10_000, C=2)
        >>> report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> report.metrics.add(
        ...     make_scorer(mean_absolute_error, response_method="predict")
        ... )
        >>> report.metrics.summarize(metric="mean_absolute_error").frame(
        ...     verbose_name=True, flat_index=False
        ... )
        Estimator                  LogisticRegression_1  LogisticRegression_2
        Metric
        Mean Absolute Error                0.05...              0.05...
        >>> report.metrics.mean_absolute_error()
        Estimator             LogisticRegression_1  LogisticRegression_2
        Metric
        Mean Absolute Error                   ...                   ...
        """
        for report in self._parent.reports_.values():
            report.metrics.add(
                metric,
                name=name,
                verbose_name=verbose_name,
                greater_is_better=greater_is_better,
                position=position,
                **kwargs,
            )

    def remove(self, name: str) -> None:
        """Remove a metric from each underlying estimator report.

        Parameters
        ----------
        name : str
            The technical name of the metric to remove.

        See Also
        --------
        add : Add a custom metric.
        """
        for report in self._parent.reports_.values():
            report.metrics.remove(name)

    def get(
        self,
        name: str,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        """Get a metric value.

        Parameters
        ----------
        name : str
            Name of the metric to compute. Get all available metrics with
            :meth:`~ComparisonReport.metrics.available()`.

        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        Returns
        -------
        pd.DataFrame
            The metric values.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.get("precision")
        Estimator        LogisticRegression_1  LogisticRegression_2
        Metric    Label
        Precision 0                  0.901961              0.901961
                  1                  0.984127              0.984127
        """
        return self._metric(metric_name=name, data_source=data_source, **kwargs).frame(
            aggregate=aggregate,
            verbose_name=True,
            flat_index=False,
        )

    def timings(
        self,
        *,
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
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = make_classification(random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> report.metrics.timings()
                                LogisticRegression_1    LogisticRegression_2
        Fit time (s)                             ...                     ...
        Predict time test (s)                    ...                     ...
        """
        if self._parent._report_type == "comparison-estimator":
            timings = pd.concat(
                [
                    pd.Series(report.metrics.timings())
                    for report in self._parent.reports_.values()
                ],
                axis=1,
                keys=self._parent.reports_.keys(),
            )
            timings.index = timings.index.str.replace("_", " ").str.capitalize()
            timings.index = pd.Index([f"{idx} (s)" for idx in timings.index])

            return timings
        else:  # "comparison-cross-validation"
            timings = pd.concat(
                [
                    report.metrics.timings(aggregate=aggregate)
                    for report in self._parent.reports_.values()
                ],
                axis=1,
                keys=self._parent.reports_.keys(),
            )

            timings.index.name = "Metric"
            if aggregate is None:
                timings.columns.names = ["Estimator", "Split"]
            elif isinstance(timings.columns, pd.MultiIndex):
                timings.columns = timings.columns.swaplevel(0, 1)
                timings = timings.sort_index(axis=1)
                timings.columns.names = [None, "Estimator"]
            else:
                stat = (
                    aggregate[0] if isinstance(aggregate, (list, tuple)) else aggregate
                )
                timings.columns = pd.MultiIndex.from_tuples(
                    [(stat, estimator) for estimator in timings.columns],
                    names=[None, "Estimator"],
                )

            return timings

    @available_if(_check_any_sub_report_has_metric("score"))
    def score(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame | pd.Series:
        """Compute the estimator's default score.

        This calls the underlying estimator's ``score`` method on the chosen data
        source. For :class:`skrub.DataOp` estimators, scorings registered via
        :meth:`~skrub.DataOp.skb.with_scoring` are used.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

        Returns
        -------
        pd.DataFrame
            The estimator's default score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.score()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        Score                       0.94...               0.94...
        """
        return self._metric("score", data_source=data_source).frame(
            aggregate=aggregate,
            verbose_name=True,
            flat_index=False,
        )

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics.

        Returns
        -------
        :class:`RocCurveDisplay`
            The ROC curve display.

        See Also
        --------
        :class:`RocCurveDisplay` : Display class for ROC curve plots.

        Notes
        -----
        To keep the stored display lightweight, the ROC curve is downsampled to at most
        500 points per class and per child report. Sampling is performed by picking
        evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> display = comparison_report.metrics.roc()
        >>> display.plot()
        """
        child_displays = [
            report.metrics.roc(data_source=data_source)
            for report in track(
                list(self._parent.reports_.values()),
                description="Computing display for each report",
            )
        ]
        estimator_names = self._parent.reports_.keys()

        display = RocCurveDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"estimator": list(estimator_names)},
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
        data_source: DataSource | Literal["both"] = "test",
    ) -> PrecisionRecallCurveDisplay:
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics.

        Returns
        -------
        :class:`PrecisionRecallCurveDisplay`
            The precision-recall curve display.

        See Also
        --------
        :class:`PrecisionRecallCurveDisplay`
            Display class for precision-recall curve plots.

        Notes
        -----
        To keep the stored display lightweight, the precision-recall curve is
        downsampled to at most 500 points per class and per child report. Sampling is
        performed by picking evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> display = comparison_report.metrics.precision_recall()
        >>> display.plot()
        """
        child_displays = [
            report.metrics.precision_recall(data_source=data_source)
            for report in track(
                list(self._parent.reports_.values()),
                description="Computing display for each report",
            )
        ]
        estimator_names = self._parent.reports_.keys()

        display = PrecisionRecallCurveDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"estimator": list(estimator_names)},
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
        data_source: DataSource | Literal["both"] = "test",
        subsample: int = 1_000,
        seed: int | None = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics.

        subsample : int, default=1_000
            Maximum number of samples to show on the scatter plot.

        seed : int, default=None
            The seed used to initialize the random number generator used for the
            subsampling.

        Returns
        -------
        :class:`PredictionErrorDisplay`
            The prediction error display.

        See Also
        --------
        :class:`PredictionErrorDisplay` : Display class for prediction error plots.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_2 = Ridge(random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> display = comparison_report.metrics.prediction_error()
        >>> display.plot(kind="actual_vs_predicted")
        """
        if isinstance(subsample, numbers.Integral):
            # Preserve the total number of sub-samples:
            n_children = len(self._parent.reports_)
            if 0 < subsample < n_children:
                subsample = 1
            else:
                subsample //= n_children

        child_displays = [
            report.metrics.prediction_error(
                data_source=data_source,
                subsample=subsample,
                seed=seed,
            )
            for report in track(
                list(self._parent.reports_.values()),
                description="Computing display for each report",
            )
        ]
        estimator_names = self._parent.reports_.keys()

        display = PredictionErrorDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"estimator": list(estimator_names)},
        )
        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def confusion_matrix(
        self,
        *,
        data_source: DataSource = "test",
    ) -> ConfusionMatrixDisplay:
        """Plot the confusion matrix.

        The confusion matrix shows the counts of correct and incorrect classifications
        for each class.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        :class:`ConfusionMatrixDisplay`
            The confusion matrix display.

        See Also
        --------
        :class:`ConfusionMatrixDisplay` : Display class for confusion matrix plots.

        Notes
        -----
        To keep the stored display lightweight, the thresholded confusion matrices are
        downsampled to at most 500 points per class and per child report. Sampling is
        performed by picking evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.svm import SVC
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> comparison = evaluate(
        ...     [LogisticRegression(max_iter=10_000), SVC()],
        ...     X,
        ...     y,
        ...     splitter=2,
        ...     pos_label=1,
        ... )
        >>> display = comparison.metrics.confusion_matrix()
        >>> display.plot()
        """
        do_thresholds = True
        if not all(
            hasattr(report.learner_, "predict_proba")
            for report in self._parent.reports_.values()
        ) and not all(
            hasattr(report.learner_, "decision_function")
            for report in self._parent.reports_.values()
        ):
            warnings.warn(
                (
                    "Not all estimators have a `predict_proba` or a "
                    "`decision_function` method. Thresholded confusion matrices are "
                    "not available."
                ),
                stacklevel=2,
            )
            do_thresholds = False
        child_displays = [
            report.metrics.confusion_matrix(data_source=data_source)
            for report in track(
                list(self._parent.reports_.values()),
                description="Computing display for each report",
            )
        ]
        estimator_names = self._parent.reports_.keys()

        display = ConfusionMatrixDisplay._concatenate(
            child_displays,
            do_thresholds=do_thresholds,
            report_type=self._parent._report_type,
            column_data={"estimator": list(estimator_names)},
        )
        return display
