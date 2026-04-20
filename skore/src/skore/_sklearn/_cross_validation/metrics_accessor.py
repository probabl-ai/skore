import numbers
from typing import Any, Literal

import pandas as pd
from joblib import Parallel
from numpy.typing import ArrayLike
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.types import Aggregate, MetricLike
from skore._utils._accessor import _check_estimator_report_has_method
from skore._utils._fixes import _validate_joblib_parallel_params
from skore._utils._metric_rows import metric_score_to_rows
from skore._utils._parallel import delayed
from skore._utils._progress_bar import track

DataSource = Literal["test", "train"]


class _MetricsAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    def summarize(
        self,
        *,
        data_source: DataSource = "test",
        metric: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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
            raise NotImplementedError(
                'data_source="both" is not yet supported for CrossValidationReport'
            )

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
                    for report in self._parent.estimator_reports_
                ),
                description="Compute metric for each split",
                total=len(self._parent.estimator_reports_),
            )
        )

        return MetricsSummaryDisplay._concatenate(
            summaries,
            report_type="cross-validation",
            extra_rows_data=[{"split": i} for i in range(len(summaries))],
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
        """
        for report in self._parent.estimator_reports_:
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
        """Get all measured processing times related to the estimator.

        The index of the returned dataframe is the name of the processing time. When
        the estimators were not used to predict, no timings regarding the prediction
        will be present.

        Parameters
        ----------
        aggregate : {"mean", "std"} or list of such str, default=None
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
        Fit time (s)       ...       ...
        >>> report.cache_predictions()
        >>> report.metrics.timings()
                                    mean       std
        Fit time (s)                 ...       ...
        Predict time test (s)        ...       ...
        Predict time train (s)       ...       ...
        """
        timings: pd.DataFrame = pd.concat(
            [
                pd.Series(report.metrics.timings())
                for report in self._parent.estimator_reports_
            ],
            axis=1,
            keys=[f"Split #{i}" for i in range(len(self._parent.estimator_reports_))],
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
        reports = self._parent.estimator_reports_
        metric = reports[0]._metric_registry[metric_name]

        rows: list[dict] = []
        for split_idx, report in enumerate(reports):
            score = getattr(report.metrics, metric_name)(
                data_source=data_source, **kwargs
            )
            split_rows = metric_score_to_rows(
                score,
                metric=metric,
                ml_task=report._ml_task,
                data_source=data_source,
                estimator_name=report.estimator_name_,
                pos_label=report.pos_label,
                kwargs=kwargs or None,
            )
            for r in split_rows:
                r["split"] = split_idx
            rows.extend(split_rows)

        return MetricsSummaryDisplay(rows=rows, report_type="cross-validation")

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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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
        Metric    Label / Average
        Precision 0                         0.93...  0.04...
                  1                         0.94...  0.02...
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

        flat_index : bool, default=True
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
        Metric Label / Average
        Recall 0                         0.91...  0.04...
               1                         0.96...  0.02...
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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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

        flat_index : bool, default=True
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
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.CrossValidationReport.metrics")

    @available_if(_check_estimator_report_has_method("metrics", "roc"))
    def roc(
        self,
        *,
        data_source: DataSource = "test",
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> display = report.metrics.roc()
        >>> display.set_style(relplot_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        child_displays = [
            report.metrics.roc(data_source=data_source)
            for report in self._parent.estimator_reports_
        ]
        split_indices = range(len(self._parent.estimator_reports_))

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
        data_source: DataSource = "test",
    ) -> PrecisionRecallCurveDisplay:
        """Plot the precision-recall curve.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

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
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=2)
        >>> display = report.metrics.precision_recall()
        >>> display.plot()
        """
        child_displays = [
            report.metrics.precision_recall(data_source=data_source)
            for report in self._parent.estimator_reports_
        ]
        split_indices = range(len(self._parent.estimator_reports_))

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

        Extra keyword arguments will be passed to matplotlib's `plot`.

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
            n_children = len(self._parent.estimator_reports_)
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
            for report in self._parent.estimator_reports_
        ]
        split_indices = range(len(self._parent.estimator_reports_))

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
            for report in self._parent.estimator_reports_
        ]
        split_indices = range(len(self._parent.estimator_reports_))

        display = ConfusionMatrixDisplay._concatenate(
            child_displays,
            report_type=self._parent._report_type,
            column_data={"split": list(split_indices)},
        )
        return display
