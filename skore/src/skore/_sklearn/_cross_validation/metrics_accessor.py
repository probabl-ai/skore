from __future__ import annotations

import numbers
from typing import Any, Literal, cast

import pandas as pd
from joblib import Parallel
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import BaseMetricsAccessor, _summarize_report_metrics
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn._plot.metrics.metrics_summary_display import MetricsSummaryRow
from skore._sklearn.metrics import Metric, MetricLike, Score
from skore._sklearn.types import Aggregate
from skore._utils._accessor import _check_estimator_report_has_method
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._index import squeeze_single_column
from skore._utils._parallel import delayed
from skore._utils._progress_bar import track

DataSource = Literal["test", "train"]


class _MetricsAccessor(BaseMetricsAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    def summarize(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        metric: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets, showing them together.

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
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(
        ...     classifier, X, y, splitter=2, pos_label=1
        ... )
        >>> report.metrics.summarize(
        ...     metric=["precision", "recall"],
        ... ).frame(favorability=True)
                   logisticregression_mean  logisticregression_std favorability
        metric
        precision                 0.94...                0.02...         (↗︎)
        recall                    0.96...                0.02...         (↗︎)
        """
        if data_source == "both":
            train_summary = self._summarize_display(data_source="train", metric=metric)
            test_summary = self._summarize_display(data_source="test", metric=metric)

            combined = pd.concat(
                [train_summary.summary, test_summary.summary], ignore_index=True
            )
            return MetricsSummaryDisplay(
                summary=combined,
                report_type="cross-validation",
                errors=train_summary.errors + test_summary.errors,
            )

        return self._summarize_display(data_source=data_source, metric=metric)

    def _summarize_display(
        self,
        *,
        data_source: DataSource | Literal["both"],
        metric: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        if data_source == "both":
            train_summary = self._summarize_display(data_source="train", metric=metric)
            test_summary = self._summarize_display(data_source="test", metric=metric)

            combined = pd.concat(
                [train_summary.summary, test_summary.summary], ignore_index=True
            )
            return MetricsSummaryDisplay(
                summary=combined,
                report_type="cross-validation",
                errors=train_summary.errors + test_summary.errors,
            )

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
                    for report in self._parent.reports_
                ),
                description="Compute metric for each split",
                total=len(self._parent.reports_),
            )
        )

        extra_rows_data = [{"split": i} for i in range(len(summaries))]
        summary = pd.concat(
            [
                display.summary.assign(**extra_data)
                for display, extra_data in zip(summaries, extra_rows_data, strict=True)
            ],
            ignore_index=True,
        )
        errors = [error for display in summaries for error in display.errors]
        return MetricsSummaryDisplay(
            summary, report_type="cross-validation", errors=errors
        )

    def available(self) -> list[str]:
        """List available metric names in the registry.

        Returns
        -------
        list[str]
            The list of available metric names.
        """
        return self._parent.reports_[0].metrics.available()

    def _resolve_metric(self, name: str) -> Metric | None:
        """Return the :class:`~skore._sklearn.metrics.Metric` for ``name``, or None."""
        return self._parent.reports_[0].metrics._resolve_metric(name)

    def add(
        self,
        metric: MetricLike | Metric,
        *,
        name: str | None = None,
        verbose_name: str | None = None,
        greater_is_better: bool = True,
        position: Literal["first", "last"] = "first",
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """Add a custom metric to :meth:`summarize`.

        Parameters
        ----------
        metric : str, sklearn scorer, callable, or Metric
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
            - If a :class:`~skore._sklearn.metrics.Metric`, it is registered as-is
              (or as a copy when ``name`` / ``verbose_name`` are set).

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
            for each split report. See :meth:`EstimatorReport.metrics.add`.

        force : bool, default=False
            If ``False`` and the metric's
            :meth:`~skore._sklearn.metrics.Metric.discouraged` returns a
            reason, raise instead of registering. Pass ``True`` to register it
            anyway. See :meth:`EstimatorReport.metrics.add`.

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
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2, pos_label=1)
        >>> report.metrics.add(
        ...     make_scorer(mean_absolute_error, response_method="predict")
        ... )
        >>> report.metrics.summarize(metric="mean_absolute_error").frame(
        ...     verbose_name=True, flat_index=False
        ... )
        Estimator           LogisticRegression
        Aggregate                         mean      std
        Metric
        Mean Absolute Error           0.05...  0.00...
        >>> report.metrics.mean_absolute_error()
        Estimator           LogisticRegression
        Aggregate                         mean      std
        Metric
        Mean Absolute Error           0.05...  0.00...
        """
        for report in self._parent.reports_:
            report.metrics.add(
                metric,
                name=name,
                verbose_name=verbose_name,
                greater_is_better=greater_is_better,
                position=position,
                force=force,
                **kwargs,
            )

    def remove(self, name: str) -> None:
        """Remove a metric from each underlying estimator report.

        Parameters
        ----------
        name : str
            The name of the metric to remove.
        """
        for report in self._parent.reports_:
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
            :meth:`~CrossValidationReport.metrics.available()`.

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
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.get("precision")
        Estimator       LogisticRegression
        Aggregate                     mean       std
        Metric    Label
        Precision 0               0.93...  0.04...
                  1               0.94...  0.02...
        """
        return self._metric(name, data_source=data_source, **kwargs).frame(
            aggregate=aggregate,
            verbose_name=True,
            flat_index=False,
        )

    def timings(
        self,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame | pd.Series:
        """Get all measured processing times related to the estimator.

        The index of the returned table is the name of the processing time. When
        the estimators were not used to predict, no timings regarding the prediction
        will be present.

        Parameters
        ----------
        aggregate : {"mean", "std"} or list of such str, default=("mean", "std")
            Function to aggregate the timings across the cross-validation splits.

        Returns
        -------
        pd.DataFrame
            A dataframe with the processing times.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(random_state=42)
        >>> estimator = LogisticRegression()
        >>> from skore import evaluate
        >>> report = evaluate(estimator, X, y, splitter=2)
        >>> report.metrics.timings()
                                    mean       std
        Fit time (s)                 ...       ...
        Predict time test (s)        ...       ...
        """
        timings: pd.DataFrame = pd.concat(
            [pd.Series(report.metrics.timings()) for report in self._parent.reports_],
            axis=1,
            keys=[f"Split #{i}" for i in range(len(self._parent.reports_))],
        )
        if aggregate:
            if isinstance(aggregate, str):
                aggregate = [aggregate]
            timings = timings.aggregate(func=aggregate, axis=1)

        timings.index = timings.index.str.replace("_", " ").str.capitalize()
        timings.index = pd.Index([f"{idx} (s)" for idx in timings.index])

        return squeeze_single_column(timings)

    def _metric(
        self,
        metric: str | Metric,
        *,
        data_source: DataSource,
        **kwargs: Any,
    ) -> MetricsSummaryDisplay:
        """Compute a single metric across cross-validation splits.

        This helper allows passing kwargs to the sub-reports, unlike :meth:`summarize`.
        A :class:`Metric` instance may be passed directly to compute a metric that is
        not registered, as :meth:`score` does for a discouraged default score.
        """
        rows: list[MetricsSummaryRow] = []
        for split_idx, report in enumerate(self._parent.reports_):
            resolved = (
                report._metric_registry[metric] if isinstance(metric, str) else metric
            )
            metric_rows = resolved.rows(
                report=report, data_source=data_source, **kwargs
            )
            rows.extend(
                cast(
                    MetricsSummaryRow,
                    {
                        "name": resolved.name,
                        "verbose_name": row["metric_verbose_name"],
                        "estimator": report.estimator_name_,
                        "data_source": data_source,
                        "split": split_idx,
                        "greater_is_better": row["greater_is_better"],
                        "score": row["score"],
                        "label": row["label"],
                        "average": row["average"],
                        "output": row["output"],
                    },
                )
                for row in metric_rows
            )

        return MetricsSummaryDisplay._compute_data_for_display(
            rows, report_type="cross-validation", errors=[]
        )

    @available_if(_check_estimator_report_has_method("metrics", "score"))
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
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.score()
        Estimator LogisticRegression
        Aggregate               mean      std
        Metric
        Score               0.94...  0.00...
        """
        return self._metric(Score(), data_source=data_source).frame(
            aggregate=aggregate,
            verbose_name=True,
            flat_index=False,
        )

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    @available_if(_check_estimator_report_has_method("metrics", "roc"))
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
            - "both" : use both the train and test and show them together.

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
        500 points per class and per split. Sampling is performed by picking
        evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> display = report.metrics.roc()
        >>> display.set_style(relplot_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        child_displays = [
            report.metrics.roc(data_source=data_source)
            for report in self._parent.reports_
        ]
        split_indices = range(len(self._parent.reports_))

        display = RocCurveDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"split": list(split_indices)},
        )
        return display

    @available_if(_check_estimator_report_has_method("metrics", "precision_recall"))
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
            - "both" : use both the train and test and show them together.

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
        downsampled to at most 500 points per class and per split. Sampling is performed
        by picking evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> display = report.metrics.precision_recall()
        >>> display.plot()
        """
        child_displays = [
            report.metrics.precision_recall(data_source=data_source)
            for report in self._parent.reports_
        ]
        split_indices = range(len(self._parent.reports_))

        display = PrecisionRecallCurveDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"split": list(split_indices)},
        )
        return display

    @available_if(_check_estimator_report_has_method("metrics", "prediction_error"))
    def prediction_error(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        subsample: float | int | None = 1_000,
        seed: int | None = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both train and test and display them together.

        subsample : float, int or None, default=1_000
            Sampling the samples to be shown on the scatter plot. If `float`,
            it should be between 0 and 1 and represents the proportion of the
            original dataset. If `int`, it represents the number of samples
            applied. by default, 1,000 samples or less will be displayed.

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
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=2)
        >>> display = report.metrics.prediction_error()
        >>> display.set_style(perfect_model_kwargs={"color": "tab:red"})
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
            for report in self._parent.reports_
        ]
        split_indices = range(len(self._parent.reports_))

        display = PredictionErrorDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"split": list(split_indices)},
        )
        return display

    @available_if(_check_estimator_report_has_method("metrics", "confusion_matrix"))
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
        downsampled to at most 500 points per class and per split. Sampling is performed
        by picking evenly-spaced indices on the sorted thresholds.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> display = report.metrics.confusion_matrix()
        >>> display.plot()

        With specific threshold for binary classification:

        >>> display = report.metrics.confusion_matrix()
        >>> display.plot(threshold_value=0.7, label=1)
        """
        child_displays = [
            report.metrics.confusion_matrix(data_source=data_source)
            for report in self._parent.reports_
        ]
        split_indices = range(len(self._parent.reports_))

        display = ConfusionMatrixDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"split": list(split_indices)},
        )
        return display
