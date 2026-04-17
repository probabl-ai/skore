import numbers
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._comparison.report import ComparisonReport
from skore._sklearn._plot.metrics import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.types import Aggregate, MetricLike
from skore._utils._accessor import (
    _check_any_sub_report_has_metric,
    _check_supported_ml_task,
)
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._progress_bar import track

DataSource = Literal["test", "train", "both"]


class _MetricsAccessor(_BaseAccessor[ComparisonReport], DirNamesMixin):
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
        Estimator       LogisticRegression_1  LogisticRegression_2
        Metric
        Precision                    0.98...               0.98...
        Recall                       0.92...               0.92...
        """
        parallel = joblib.Parallel(
            **_validate_joblib_parallel_params(
                n_jobs=self._parent.n_jobs, return_as="generator"
            )
        )

        summaries = list(
            track(
                parallel(
                    joblib.delayed(report.metrics.summarize)(
                        data_source=data_source,
                        metric=metric,
                    )
                    for report in self._parent.reports_.values()
                ),
                description="Compute metric for each estimator",
                total=len(self._parent.reports_),
            )
        )

        return MetricsSummaryDisplay._concatenate(
            summaries,
            report_type=self._parent._report_type,
            extra_rows_data=[
                {"estimator_name": estimator_name}
                for estimator_name in self._parent.reports_
            ],
        )

    def _metric(
        self, metric_name: str, *, data_source: DataSource, **kwargs: Any
    ) -> MetricsSummaryDisplay:
        """Compute a single metric across compared reports, forwarding *kwargs*."""
        summaries = [
            report.metrics._metric(metric_name, data_source=data_source, **kwargs)
            for report in self._parent.reports_.values()
        ]

        return MetricsSummaryDisplay._concatenate(
            summaries,
            report_type=self._parent._report_type,
            extra_rows_data=[
                {"estimator_name": estimator_name}
                for estimator_name in self._parent.reports_
            ],
        )

    def add(
        self,
        metric: MetricLike,
        *,
        name: str | None = None,
        response_method: str | list[str] = "predict",
        greater_is_better: bool = True,
        **kwargs: Any,
    ) -> None:
        """Add a custom metric to be included in :meth:`summarize` by default.

        Parameters
        ----------
        metric : str, sklearn scorer, or callable
            The metric to add.

        name : str, optional
            Custom name for the metric. If not provided, the name is inferred
            from the metric (e.g. the function's ``__name__``).

        response_method : str or list of str, default="predict"
            Estimator method to get predictions (only for callables).

        greater_is_better : bool, default=True
            Whether higher values are better (only for callables).

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
        >>> report.metrics.summarize().frame()
        Estimator                            LogisticRegression_1  LogisticRegression_2
        Metric              Label / Average
        ...
        Mean Absolute Error                                   ...                   ...
        """
        for report in self._parent.reports_.values():
            report.metrics.add(
                metric,
                name=name,
                response_method=response_method,
                greater_is_better=greater_is_better,
                **kwargs,
            )

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
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = make_classification(random_state=42)
        >>> estimator_1 = LogisticRegression()
        >>> estimator_2 = LogisticRegression(C=2)  # Different regularization
        >>> report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> report.metrics.timings()
                        LogisticRegression_1    LogisticRegression_2
        Fit time (s)                     ...                     ...
        >>> report.cache_predictions()
        >>> report.metrics.timings()
                                LogisticRegression_1    LogisticRegression_2
        Fit time (s)                             ...                     ...
        Predict time test (s)                    ...                     ...
        Predict time train (s)                   ...                     ...
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
            else:
                timings.columns = timings.columns.swaplevel(0, 1)
                timings = timings.sort_index(axis=1)
                timings.columns.names = [None, "Estimator"]

            return timings

    @available_if(_check_any_sub_report_has_metric("accuracy"))
    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.accuracy()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        Accuracy                    0.94...               0.94...
        """
        return self._metric("accuracy", data_source=data_source).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("precision"))
    def precision(
        self,
        *,
        data_source: DataSource = "test",
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        aggregate: Aggregate | None = ("mean", "std"),
    ) -> pd.DataFrame:
        """Compute the precision score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        average : {"binary", "macro", "micro", "weighted", "samples"} or None, \
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.precision()
        Estimator                    LogisticRegression_1  LogisticRegression_2
        Metric      Label / Average
        Precision                 0               0.90...               0.90...
                                  1               0.98...               0.98...
        """
        return self._metric(
            "precision", data_source=data_source, average=average
        ).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("recall"))
    def recall(
        self,
        *,
        data_source: DataSource = "test",
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        aggregate: Aggregate | None = ("mean", "std"),
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.recall()
        Estimator                    LogisticRegression_1  LogisticRegression_2
        Metric      Label / Average
        Recall                    0              0.978...              0.978...
                                  1              0.925...              0.925...
        """
        return self._metric("recall", data_source=data_source, average=average).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("brier_score"))
    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.brier_score()
        Estimator         LogisticRegression_1  LogisticRegression_2
        Metric
        Brier score                   0.036...              0.036...
        """
        return self._metric("brier_score", data_source=data_source).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("roc_auc"))
    def roc_auc(
        self,
        *,
        data_source: DataSource = "test",
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
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.roc_auc()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        ROC AUC                     0.99...               0.99...
        """
        return self._metric(
            "roc_auc", data_source=data_source, average=average, multi_class=multi_class
        ).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("log_loss"))
    def log_loss(
        self,
        *,
        data_source: DataSource = "test",
        aggregate: Aggregate | None = ("mean", "std"),
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = LogisticRegression(max_iter=10000, random_state=42)
        >>> estimator_2 = LogisticRegression(max_iter=10000, random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.log_loss()
        Estimator      LogisticRegression_1  LogisticRegression_2
        Metric
        Log loss                   0.110...              0.110...
        """
        return self._metric("log_loss", data_source=data_source).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("r2"))
    def r2(
        self,
        *,
        data_source: DataSource = "test",
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
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_2 = Ridge(random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.r2()
        Estimator     Ridge_1    Ridge_2
        Metric
        R²            0.34...    0.34...
        """
        return self._metric(
            "r2", data_source=data_source, multioutput=multioutput
        ).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("rmse"))
    def rmse(
        self,
        *,
        data_source: DataSource = "test",
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
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_2 = Ridge(random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.rmse()
        Estimator       Ridge_1       Ridge_2
        Metric
        RMSE          58.132...     58.132...
        """
        return self._metric(
            "rmse", data_source=data_source, multioutput=multioutput
        ).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("mae"))
    def mae(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_2 = Ridge(random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.mae()
        Estimator     Ridge_1    Ridge_2
        Metric
        MAE        46.5...    46.5...
        """
        return self._metric(
            "mae", data_source=data_source, multioutput=multioutput
        ).frame(
            aggregate=aggregate,
        )

    @available_if(_check_any_sub_report_has_metric("mape"))
    def mape(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: Literal["raw_values", "uniform_average"] = "raw_values",
        aggregate: Aggregate | None = ("mean", "std"),
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
            Ignored when comparison is between :class:`~skore.EstimatorReport` instances

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
        >>> estimator_1 = Ridge(random_state=42)
        >>> estimator_2 = Ridge(random_state=43)
        >>> comparison_report = evaluate([estimator_1, estimator_2], X, y, splitter=0.2)
        >>> comparison_report.metrics.mape()
        Estimator     Ridge_1    Ridge_2
        Metric
        MAPE       0.3...     0.3...
        """
        return self._metric(
            "mape", data_source=data_source, multioutput=multioutput
        ).frame(
            aggregate=aggregate,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.ComparisonReport.metrics")

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

        Extra keyword arguments will be passed to matplotlib's `plot`.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics.

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
        :class:`PredictionErrorDisplay`
            The prediction error display.

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
        >>> display.plot(threshold_value=0.7)
        """
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
            report_type=self._parent._report_type,
            column_data={"estimator": list(estimator_names)},
        )
        return display
