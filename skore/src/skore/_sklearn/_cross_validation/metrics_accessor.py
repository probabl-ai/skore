from __future__ import annotations

import numbers
from typing import Any, Literal, cast

import pandas as pd
from joblib import Parallel
from numpy.typing import ArrayLike
from sklearn.metrics import auc
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import BaseMetricsAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn._plot.metrics.metrics_summary_display import MetricsSummaryRow
from skore._sklearn.metrics import MetricLike
from skore._sklearn.types import Aggregate
from skore._utils._accessor import _check_estimator_report_has_method
from skore._utils._fixes import _validate_joblib_parallel_params
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
            - "both" : use both the train and test sets, showing them side-by-side.

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
        ... ).frame(flat_index=False, favorability=True)
                  LogisticRegression           Favorability
                                mean       std
        Metric
        Precision           0.94...  0.02...         (↗︎)
        Recall              0.96...  0.02...         (↗︎)
        """
        if data_source == "both":
            train_summary = self.summarize(data_source="train", metric=metric)
            test_summary = self.summarize(data_source="test", metric=metric)

            combined = train_summary.rows + test_summary.rows
            return MetricsSummaryDisplay(rows=combined, report_type="cross-validation")

        parallel = Parallel(
            **_validate_joblib_parallel_params(
                n_jobs=self._parent.n_jobs, return_as="generator"
            )
        )

        summaries = list(
            track(
                parallel(
                    delayed(report.metrics.summarize)(
                        data_source=data_source,
                        metric=metric,
                    )
                    for report in self._parent.reports_
                ),
                description="Compute metric for each split",
                total=len(self._parent.reports_),
            )
        )

        return MetricsSummaryDisplay._concatenate(
            summaries,
            report_type="cross-validation",
            extra_rows_data=[{"split": i} for i in range(len(summaries))],
        )

    def available(self) -> list[str]:
        """List available metric names in the registry.

        Returns
        -------
        list[str]
            The list of available metric names.
        """
        return self._parent.reports_[0].metrics.available()

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
        """
        Add a custom metric to :meth:`~skore.CrossValidationReport.metrics.summarize`.

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
            for each split report. See :meth:`EstimatorReport.metrics.add`.

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
        >>> report.metrics.summarize().frame()
                            LogisticRegression
                                mean       std
        Metric
        ...
        Mean Absolute Error      ...       ...
        >>> report.metrics.mean_absolute_error()
                             LogisticRegression
                                          mean   std
        Metric
        Mean Absolute Error            0.05...   ...
        """
        for report in self._parent.reports_:
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
            The name of the metric to remove.
        """
        for report in self._parent.reports_:
            report.metrics.remove(name)

    def get(
        self,
        name: str,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
        **kwargs,
    ) -> pd.DataFrame | None:
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

        flat_index : bool, default=True
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The metric values, or None if the metric is not available.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.get("precision", flat_index=False)
                        LogisticRegression
                                      mean       std
        Metric    Label
        Precision 0                0.93...   0.04...
                  1                0.94...   0.02...
        """
        return self._metric(metric_name=name, data_source=data_source, **kwargs).frame(
            aggregate=aggregate, flat_index=flat_index
        )

    def timings(
        self,
        *,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Get all measured processing times related to the estimator.

        The index of the returned dataframe is the name of the processing time. When
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

        return timings

    def _metric(
        self, metric_name: str, *, data_source: DataSource, **kwargs: Any
    ) -> MetricsSummaryDisplay:
        """Compute a single metric across cross-validation splits.

        This helper allows passing kwargs to the sub-reports, unlike :meth:`summarize`.
        """
        rows: list[MetricsSummaryRow] = []
        for split_idx, report in enumerate(self._parent.reports_):
            metric = report._metric_registry[metric_name]
            metric_rows = metric.rows(report=report, data_source=data_source, **kwargs)
            rows.extend(
                cast(
                    MetricsSummaryRow,
                    row
                    | {
                        "metric_name": metric.name,
                        "estimator_name": report.estimator_name_,
                        "data_source": data_source,
                        "split": split_idx,
                    },
                )
                for row in metric_rows
            )

        return MetricsSummaryDisplay(rows=rows, report_type="cross-validation")

    @available_if(_check_estimator_report_has_method("metrics", "score"))
    def score(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
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

        flat_index : bool, default=True
            Whether to return a flat index or a multi-index.

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
        >>> report.metrics.score(flat_index=False)
                LogisticRegression
                            mean      std
        Metric
        Score              0.94...  0.00...
        """
        return self._metric("score", data_source=data_source).frame(
            aggregate=aggregate, flat_index=flat_index
        )

    @available_if(_check_estimator_report_has_method("metrics", "accuracy"))
    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.accuracy(flat_index=False)
                LogisticRegression
                            mean      std
        Metric
        Accuracy           0.94...  0.00...
        """
        return self._metric("accuracy", data_source=data_source).frame(
            aggregate=aggregate, flat_index=flat_index
        )

    @available_if(_check_estimator_report_has_method("metrics", "precision"))
    def precision(
        self,
        *,
        data_source: DataSource = "test",
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by the
              report's `pos_label`. This is applicable only if targets
              (`y_{true,pred}`) are binary.
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
                If the report's `pos_label` is specified and `average` is None,
                then we report only the statistics of the positive class (i.e.
                equivalent to `average="binary"`).

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.precision()
                LogisticRegression
                              mean       std
        Metric    Label
        Precision 0               0.93...  0.04...
                  1               0.94...  0.02...
        """
        return self._metric(
            "precision", data_source=data_source, average=average
        ).frame(aggregate=aggregate, flat_index=flat_index)

    @available_if(_check_estimator_report_has_method("metrics", "recall"))
    def recall(
        self,
        *,
        data_source: DataSource = "test",
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the recall score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"binary","macro", "micro", "weighted", "samples"} or None, \
                default=None
            Used with multiclass problems.
            If `None`, the metrics for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:

            - "binary": Only report results for the class specified by the
              report's `pos_label`. This is applicable only if targets
              (`y_{true,pred}`) are binary.
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
                If the report's `pos_label` is specified and `average` is None,
                then we report only the statistics of the positive class (i.e.
                equivalent to `average="binary"`).

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.recall()
             LogisticRegression
                           mean       std
        Metric Label
        Recall 0               0.91...  0.04...
               1               0.96...  0.02...
        """
        return self._metric("recall", data_source=data_source, average=average).frame(
            aggregate=aggregate, flat_index=flat_index
        )

    @available_if(_check_estimator_report_has_method("metrics", "brier_score"))
    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.brier_score()
                    LogisticRegression
                                mean       std
        Metric
        Brier score            0.04...  0.00...
        """
        return self._metric("brier_score", data_source=data_source).frame(
            aggregate=aggregate, flat_index=flat_index
        )

    @available_if(_check_estimator_report_has_method("metrics", "roc_auc"))
    def roc_auc(
        self,
        *,
        data_source: DataSource = "test",
        average: Literal["macro", "micro", "weighted", "samples"] | None = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the ROC AUC score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"macro", "micro", "weighted", "samples"}, default=None
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

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The ROC AUC score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.roc_auc()
                LogisticRegression
                            mean       std
        Metric
        ROC AUC           0.98...  0.00...
        """
        return self._metric(
            "roc_auc", data_source=data_source, average=average, multi_class=multi_class
        ).frame(aggregate=aggregate, flat_index=flat_index)

    @available_if(_check_estimator_report_has_method("metrics", "log_loss"))
    def log_loss(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the log loss.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> report.metrics.log_loss()
                LogisticRegression
                            mean       std
        Metric
        Log loss            0.14...  0.03...
        """
        return self._metric(
            "log_loss",
            data_source=data_source,
        ).frame(aggregate=aggregate, flat_index=flat_index)

    @available_if(_check_estimator_report_has_method("metrics", "r2"))
    def r2(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the R² score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=2)
        >>> report.metrics.r2()
                Ridge
                    mean       std
        Metric
        R²      0.37...  0.02...
        """
        return self._metric(
            "r2", data_source=data_source, multioutput=multioutput
        ).frame(aggregate=aggregate, flat_index=flat_index)

    @available_if(_check_estimator_report_has_method("metrics", "rmse"))
    def rmse(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the root mean squared error.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=2)
        >>> report.metrics.rmse()
                    Ridge
                    mean       std
        Metric
        RMSE    60.7...  1.0...
        """
        return self._metric(
            "rmse", data_source=data_source, multioutput=multioutput
        ).frame(aggregate=aggregate, flat_index=flat_index)

    @available_if(_check_estimator_report_has_method("metrics", "mae"))
    def mae(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"]
        | ArrayLike = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the mean absolute error.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The mean absolute error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=2)
        >>> report.metrics.mae()
                    Ridge
                    mean       std
        Metric
        MAE     5...       ...
        """
        return self._metric(
            "mae", data_source=data_source, multioutput=multioutput
        ).frame(aggregate=aggregate, flat_index=flat_index)

    @available_if(_check_estimator_report_has_method("metrics", "mape"))
    def mape(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"]
        | ArrayLike = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
        flat_index: bool = False,
    ) -> pd.DataFrame:
        """Compute the mean absolute percentage error.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        pd.DataFrame
            The mean absolute percentage error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=2)
        >>> report.metrics.mape()
                    Ridge
                    mean       std
        Metric
        MAPE      0....      ...
        """
        return self._metric(
            "mape", data_source=data_source, multioutput=multioutput
        ).frame(aggregate=aggregate, flat_index=flat_index)

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    @available_if(_check_estimator_report_has_method("metrics", "roc"))
    def roc(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        average: Literal["threshold"] | None = None,
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test and show them side-by-side.

        average : {"threshold"} or None, default=None
            If "threshold", computes the threshold-averaged ROC curve across the
            cross-validation folds. The resulting plot will display the individual
            cross-validation splits as faded lines in the background, with the bold
            threshold-averaged curve overlaid on top.
            Only valid for binary classification.

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

        When `average="threshold"` is used, the individual fold lines can be visually
        suppressed from the plot by passing `relplot_kwargs={"alpha": 0.0}` to the
        `set_style()` method before calling `plot()`.

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
        if average is None:
            split_indices = range(len(self._parent.reports_))
            display = RocCurveDisplay._concatenate(
                child_displays,
                report_type=self._parent._report_type,
                column_data={"split": list(split_indices)},
            )
            return display

        if average != "threshold":
            raise ValueError('average must be "threshold" or None.')
        if self._parent._ml_task != "binary-classification":
            raise ValueError(
                "Averaging is not implemented for multi class classification"
            )

        split_indices = range(len(self._parent.reports_))
        display = RocCurveDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"split": list(split_indices)},
        )

        roc_curve_df = display.roc_curve.copy()
        roc_auc_df = display.roc_auc.copy()

        roc_curve_df["average"] = None
        roc_auc_df["average"] = None

        roc_curve_records = []
        roc_auc_records = []

        for (label, ds), group in display.roc_curve.groupby(
            ["label", "data_source"], observed=True
        ):
            all_thresholds = []
            all_fprs = []
            all_tprs = []

            for _, split_group in group.groupby("split", observed=True):
                sorted_group = split_group.sort_values("threshold", ascending=False)
                all_thresholds.append(sorted_group["threshold"].values)
                all_fprs.append(sorted_group["fpr"].values)
                all_tprs.append(sorted_group["tpr"].values)

            average_fpr, average_tpr, average_threshold = (
                RocCurveDisplay._threshold_average(all_fprs, all_tprs, all_thresholds)
            )
            average_roc_auc = auc(average_fpr, average_tpr)

            roc_curve_records.append(
                pd.DataFrame(
                    {
                        "estimator": display.roc_curve["estimator"].iloc[0],
                        "data_source": ds,
                        "split": None,
                        "label": label,
                        "threshold": average_threshold,
                        "fpr": average_fpr,
                        "tpr": average_tpr,
                        "average": "threshold",
                    }
                )
            )

            roc_auc_records.append(
                {
                    "estimator": display.roc_auc["estimator"].iloc[0],
                    "data_source": ds,
                    "split": None,
                    "label": label,
                    "roc_auc": average_roc_auc,
                    "average": "threshold",
                }
            )

        avg_curve_df = (
            pd.concat(roc_curve_records, ignore_index=True)
            if roc_curve_records
            else pd.DataFrame(columns=roc_curve_df.columns)
        )
        roc_curve_df = pd.concat([roc_curve_df, avg_curve_df], ignore_index=True)

        avg_auc_df = (
            pd.DataFrame(roc_auc_records)
            if roc_auc_records
            else pd.DataFrame(columns=roc_auc_df.columns)
        )
        roc_auc_df = pd.concat([roc_auc_df, avg_auc_df], ignore_index=True)

        for col in display.roc_curve.columns:
            if isinstance(display.roc_curve[col].dtype, pd.CategoricalDtype):
                roc_curve_df[col] = roc_curve_df[col].astype(
                    display.roc_curve[col].dtype
                )

        for col in display.roc_auc.columns:
            if isinstance(display.roc_auc[col].dtype, pd.CategoricalDtype):
                roc_auc_df[col] = roc_auc_df[col].astype(display.roc_auc[col].dtype)

        roc_curve_df["average"] = roc_curve_df["average"].astype("category")
        roc_auc_df["average"] = roc_auc_df["average"].astype("category")

        average_display = RocCurveDisplay(
            roc_curve=roc_curve_df,
            roc_auc=roc_auc_df,
            report_pos_label=display.report_pos_label,
            data_source=display.data_source,
            ml_task=display.ml_task,
            report_type="cross-validation",
        )
        return average_display

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
            - "both" : use both the train and test and show them side-by-side.

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
        data_source: DataSource = "test",
        subsample: float | int | None = 1_000,
        seed: int | None = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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
