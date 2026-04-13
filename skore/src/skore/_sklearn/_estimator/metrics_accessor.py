from collections.abc import Callable, Sized
from typing import Any, Literal, cast

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.metrics import (
    BUILTIN_METRICS,
    R2,
    Accuracy,
    Brier,
    FitTime,
    LogLoss,
    Metric,
    Precision,
    PredictTime,
    Recall,
    Rmse,
    RocAuc,
)
from skore._sklearn.types import DataSource, MetricLike, PositiveLabel
from skore._utils._accessor import _check_supported_ml_task
from skore._utils._cache_key import make_cache_key


class _MetricsAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    def __getattribute__(self, name):
        """Hide some metric methods conditionally.

        When the registry is initialized, the report is analyzed to filter metrics
        depending on the report's characteristics (e.g. the ML task and the estimator's
        prediction methods).
        """
        if (
            name in {m.name for m in BUILTIN_METRICS}
            and name not in self._parent._metric_registry
        ):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )
        return super().__getattribute__(name)

    def _parse_metrics(
        self,
        metric: MetricLike | list[MetricLike] | dict[str, MetricLike] | None,
        metric_kwargs: dict[str, Any],
    ) -> dict[str, Metric]:
        """Normalize arguments into a mapping from verbose name to Metric.

        Parameters
        ----------
        metric : MetricLike, list of MetricLike, dict of MetricLike, or None
            The metrics to parse.
        metric_kwargs : dict
            Kwargs to pass to each metric.
        """
        items: list[tuple[str | None, MetricLike]]
        if metric is None or (isinstance(metric, Sized) and len(metric) == 0):
            items = [(None, m) for m in self._parent._metric_registry]
        elif isinstance(metric, dict):
            items = list(metric.items())
        elif isinstance(metric, list):
            items = [(None, m) for m in metric]
        else:
            items = [(None, metric)]

        result = {}
        for display_name, m in items:
            parsed_metric = self._parent._metric_registry.check_metric(m, metric_kwargs)
            key = (
                display_name if display_name is not None else parsed_metric.verbose_name
            )
            result[key] = parsed_metric

        return result

    def summarize(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        metric: MetricLike | list[MetricLike] | dict[str, MetricLike] | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        response_method: str | list[str] | None = None,
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for our estimator.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

        metric : str, callable, scorer, or list of such instances or dict of such \
            instances, default=None
            The metrics to report. The possible values are:

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
            - if a dict, the keys are used as metric names and the values are the
              metric functions (strings, callables, or scorers as described above).
            - if a list, each element can be any of the above types (strings, callables,
              scorers).

        metric_kwargs : dict, default=None
            The keyword arguments to pass to the metric functions.

        response_method : {"predict", "predict_proba", "predict_log_proba", \
            "decision_function"} or list of such str, default=None
            The estimator's method to be invoked to get the predictions. Only necessary
            for custom metrics.

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
        >>> report = evaluate(classifier, X, y, splitter=0.2, pos_label=1)
        >>> report.metrics.summarize().frame(favorability=True).drop(
        ...    ["Fit time (s)", "Predict time (s)"]
        ... )
                    LogisticRegression Favorability
        Metric
        Accuracy               0.94...         (↗︎)
        Precision              0.98...         (↗︎)
        Recall                 0.92...         (↗︎)
        ROC AUC                0.99...         (↗︎)
        Log loss               0.11...         (↘︎)
        Brier score            0.03...         (↘︎)
        >>> # Using scikit-learn metrics
        >>> report.metrics.summarize(metric="f1").frame(favorability=True)
                                  LogisticRegression Favorability
        Metric   Label / Average
        F1 Score               1             0.95...          (↗︎)
        >>> report.metrics.summarize(
        ...    data_source="both"
        ... ).frame(favorability=True).drop(["Fit time (s)", "Predict time (s)"])
                     LogisticRegression (train)  LogisticRegression (test)  Favorability
        Metric
        Accuracy                        0.96...                     0.94...          (↗︎)
        Precision                       0.96...                     0.98...          (↗︎)
        Recall                          0.97...                     0.92...          (↗︎)
        ROC AUC                         0.99...                     0.99...          (↗︎)
        Log loss                        0.08...                     0.11...          (↘︎)
        Brier score                     0.02...                     0.03...          (↘︎)
        """
        if data_source == "both":
            train_summary = self.summarize(
                data_source="train",
                metric=metric,
                metric_kwargs=metric_kwargs,
                response_method=response_method,
            )
            test_summary = self.summarize(
                data_source="test",
                metric=metric,
                metric_kwargs=metric_kwargs,
                response_method=response_method,
            )

            combined = pd.concat(
                [train_summary.data, test_summary.data], ignore_index=True
            )
            return MetricsSummaryDisplay(data=combined, report_type="estimator")

        parsed_metrics = self._parse_metrics(
            metric,
            (metric_kwargs or {})
            | ({"response_method": response_method} if response_method else {}),
        )

        rows = []
        for metric_name, parsed_metric in parsed_metrics.items():
            score = parsed_metric(
                report=self._parent,
                data_source=data_source,
                **parsed_metric.kwargs,
            )

            row = {
                "metric": metric_name,
                "estimator_name": self._parent.estimator_name_,
                "data_source": data_source,
                "favorability": parsed_metric.icon,
                "label": None,
                "average": None,
                "output": None,
                "score": score,
            }

            if (
                self._parent._ml_task == "binary-classification"
                and parsed_metric.kwargs.get("average") == "binary"
            ):
                rows.append({**row, "label": self._parent.pos_label})
            elif self._parent._ml_task in (
                "binary-classification",
                "multiclass-classification",
            ):
                if isinstance(score, dict):
                    for label in score:
                        rows.append({**row, "label": label, "score": score[label]})  # noqa: PERF401
                else:
                    rows.append({**row, "average": parsed_metric.kwargs.get("average")})
            elif self._parent._ml_task == "multioutput-regression":
                if isinstance(score, list):
                    for output_idx, output_score in enumerate(score):
                        rows.append(
                            {**row, "output": output_idx, "score": output_score}
                        )
                else:
                    rows.append(
                        {**row, "average": parsed_metric.kwargs.get("multioutput")}
                    )
            else:
                rows.append(row)

        data = pd.DataFrame(rows)

        # Preserve original types from being converted to float
        # (which is pandas' default behaviour when there are `None` in the columns)
        if any(isinstance(row["label"], bool) for row in rows):
            data["label"] = data["label"].astype(pd.BooleanDtype())
        elif any(isinstance(row["label"], int) for row in rows):
            data["label"] = data["label"].astype(pd.Int64Dtype())

        if any(isinstance(row["output"], int) for row in rows):
            data["output"] = data["output"].astype(pd.Int64Dtype())

        return MetricsSummaryDisplay(data=data, report_type="estimator")

    def fit_time(self, cast: bool = True, **kwargs) -> float | None:
        """Get time to fit the estimator.

        Parameters
        ----------
        cast : bool, default=True
            Whether to cast the return value to a float. If `False`, the return value
            is `None` when the estimator is not fitted.

        kwargs : dict
            Additional arguments that are ignored but present for compatibility with
            other metrics.
        """
        return FitTime()(report=self._parent, cast=cast)

    def predict_time(
        self,
        *,
        data_source: DataSource = "test",
        cast: bool = True,
    ) -> float | None:
        """Get prediction time if it has been already measured.

        Parameters
        ----------
        cast : bool, default=True
            Whether to cast the numbers to floats. If `False`, the return value
            is `None` when the predictions have never been computed.
        """
        return PredictTime()(report=self._parent, data_source=data_source, cast=cast)

    def timings(self) -> dict:
        """Get all measured processing times related to the estimator.

        When an estimator is fitted inside the :class:`~skore.EstimatorReport`, the time
        to fit is recorded. Prediction time is recorded when the estimator's
        `predict` method is computed and cached for a given data source. This function
        returns all the recorded times.

        Returns
        -------
        timings : dict
            The recorded times, in seconds,
            in the form of a `dict` with some or all of the following keys:

            - "fit_time", for the time to fit the estimator in the train set.
            - "predict_time_{data_source}", where data_source is "train" or "test"
              for the time to compute the predictions on the given data source.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = make_classification(random_state=42)
        >>> estimator = LogisticRegression()
        >>> report = evaluate(estimator, X, y, splitter=0.2)
        >>> report.metrics.timings()
        {'fit_time': ...}
        >>> report.cache_predictions()
        >>> report.metrics.timings()
        {'fit_time': ..., 'predict_time_test': ...}
        """
        fit_time_ = self.fit_time(cast=False)
        fit_time = {"fit_time": fit_time_} if fit_time_ is not None else {}

        # predict_time cache keys are of the form
        # (data_source, "predict_time", None)
        predict_times = {
            f"predict_time_{data_source}": v
            for (data_source, name, _), v in self._parent._cache.items()
            if name == "predict_time"
        }

        return fit_time | predict_times

    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
    ) -> float:
        """Compute the accuracy score.

        Parameters
        ----------
        data_source : {"test", "train",}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        float
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> report.metrics.accuracy()
        0.94...
        """
        return Accuracy()(report=self._parent, data_source=data_source)  # type: ignore[return-value]

    def precision(
        self,
        *,
        data_source: DataSource = "test",
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
    ) -> float | dict[PositiveLabel, float]:
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

            - "binary": Only report results for the class specified by the report's
              `pos_label`. This is applicable only if targets (`y_{true,pred}`) are
              binary.
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

        Returns
        -------
        float or dict
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2, pos_label=1)
        >>> report.metrics.precision()
        0.98...
        """
        return Precision()(
            report=self._parent, data_source=data_source, average=average
        )

    def recall(
        self,
        *,
        data_source: DataSource = "test",
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
    ) -> float | dict[PositiveLabel, float]:
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
                If `pos_label` is specified and `average` is None, then we report
                only the statistics of the positive class (i.e. equivalent to
                `average="binary"`).

        Returns
        -------
        float or dict
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2, pos_label=1)
        >>> report.metrics.recall()
        0.92...
        """
        return Recall()(report=self._parent, data_source=data_source, average=average)

    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
    ) -> float:
        """Compute the Brier score.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        float
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> report.metrics.brier_score()
        0.03...
        """
        return Brier()(report=self._parent, data_source=data_source)

    def roc_auc(
        self,
        *,
        data_source: DataSource = "test",
        average: Literal["macro", "micro", "weighted", "samples"] | None = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
    ) -> float | dict[PositiveLabel, float]:
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

        Returns
        -------
        float or dict
            The ROC AUC score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> report.metrics.roc_auc()
        0.99...
        """
        return RocAuc()(
            report=self._parent,
            data_source=data_source,
            average=average,
            multi_class=multi_class,
        )

    def log_loss(
        self,
        *,
        data_source: DataSource = "test",
    ) -> float:
        """Compute the log loss.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        float
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import evaluate
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> report.metrics.log_loss()
        0.11...
        """
        return LogLoss()(report=self._parent, data_source=data_source)  # type: ignore[return-value]

    def r2(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list:
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

        Returns
        -------
        float or list of ``n_outputs``
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=0.2)
        >>> report.metrics.r2()
        0.34...
        """
        return R2()(
            report=self._parent, data_source=data_source, multioutput=multioutput
        )

    def rmse(
        self,
        *,
        data_source: DataSource = "test",
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list:
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

        Returns
        -------
        float or list of ``n_outputs``
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=0.2)
        >>> report.metrics.rmse()
        58.1...
        """
        return Rmse()(
            report=self._parent, data_source=data_source, multioutput=multioutput
        )

    def custom_metric(
        self,
        metric_function: Callable,
        response_method: str | list[str],
        *,
        data_source: DataSource = "test",
        **kwargs: Any,
    ) -> float | dict[PositiveLabel, float] | list:
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

        response_method : {"predict", "predict_proba", "predict_log_proba", \
            "decision_function"} or list of such str
            The estimator's method to be invoked to get the predictions.

        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        **kwargs : dict
            Any additional keyword arguments to be passed to the metric function.

        Returns
        -------
        float, dict, or list of ``n_outputs``
            The custom metric. The output type depends on the metric function.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_absolute_error
        >>> from skore import evaluate
        >>> X, y = load_diabetes(return_X_y=True)
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=0.2)
        >>> report.metrics.custom_metric(
        ...     metric_function=mean_absolute_error,
        ...     response_method="predict",
        ... )
        46.5...
        >>> def metric_function(y_true, y_pred):
        ...     return {"output": float(mean_absolute_error(y_true, y_pred))}
        >>> report.metrics.custom_metric(
        ...     metric_function=metric_function,
        ...     response_method="predict",
        ... )
        {'output': 46.5...}
        """
        metric = self._parent._metric_registry.check_metric(
            metric_function,
            {"response_method": response_method},
        )
        return metric(report=self._parent, data_source=data_source, **kwargs)

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.EstimatorReport.metrics")

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    def _get_display(
        self,
        *,
        data_source: DataSource | Literal["both"],
        response_method: str | list[str] | tuple[str, ...],
        display_class: type[
            RocCurveDisplay
            | PrecisionRecallCurveDisplay
            | PredictionErrorDisplay
            | ConfusionMatrixDisplay
        ],
        display_kwargs: dict[str, Any],
    ) -> (
        RocCurveDisplay
        | PrecisionRecallCurveDisplay
        | PredictionErrorDisplay
        | ConfusionMatrixDisplay
    ):
        """Get the display from the cache or compute it.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train set and the test set to compute the metric.

        response_method : str, list of str or tuple of str
            The response method.

        display_class : class
            The display class.

        display_kwargs : dict
            The display kwargs used by `display_class._compute_data_for_display`.

        Returns
        -------
        display : display_class
            The display.
        """
        if data_source == "both":
            displays = [
                self._get_display(
                    data_source=cast(DataSource, ds),
                    response_method=response_method,
                    display_class=display_class,
                    display_kwargs=display_kwargs,
                )
                for ds in ["train", "test"]
            ]
            return display_class._concatenate(
                displays,  # type: ignore[arg-type]
                report_type=self._parent._report_type,
                data_source=data_source,
            )

        # Compute cache key
        if "seed" in display_kwargs and display_kwargs["seed"] is None:
            cache_key = None
        else:
            cache_key = make_cache_key(
                data_source, display_class.__name__, display_kwargs
            )

        cache_value = self._parent._cache.get(cache_key)
        if cache_value is not None:
            return cache_value

        data_source = cast(DataSource, data_source)
        _, y_true = self._parent._get_data_and_y_true(data_source=data_source)

        y_pred = self._parent._get_predictions(
            data_source=data_source, response_method=response_method
        )

        display = display_class._compute_data_for_display(
            y_true=y_true,
            y_pred=y_pred,
            report_type=self._parent._report_type,
            estimator=self._parent.estimator_,
            estimator_name=self._parent.estimator_name_,
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
        data_source: DataSource | Literal["both"] = "test",
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

        Parameters
        ----------
        data_source : {"test", "train", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

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
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.roc()
        >>> display.set_style(relplot_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"report_pos_label": self._parent.pos_label}
        display = cast(
            RocCurveDisplay,
            self._get_display(
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
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.precision_recall()
        >>> display.set_style(relplot_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"report_pos_label": self._parent.pos_label}
        display = cast(
            PrecisionRecallCurveDisplay,
            self._get_display(
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
        data_source: DataSource | Literal["both"] = "test",
        subsample: float | int | None = 1_000,
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
            - "both" : use both the train and test sets to compute the metrics and
              present them side-by-side.

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
        >>> regressor = Ridge()
        >>> report = evaluate(regressor, X, y, splitter=0.2)
        >>> display = report.metrics.prediction_error()
        >>> display.set_style(perfect_model_kwargs={"color": "tab:red"})
        >>> display.plot()
        """
        display_kwargs = {"subsample": subsample, "seed": seed}
        display = cast(
            PredictionErrorDisplay,
            self._get_display(
                data_source=data_source,
                response_method="predict",
                display_class=PredictionErrorDisplay,
                display_kwargs=display_kwargs,
            ),
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
        >>> report = evaluate(classifier, X, y, splitter=0.2)
        >>> display = report.metrics.confusion_matrix()
        >>> display.plot()

        With specific threshold for binary classification:

        >>> display = report.metrics.confusion_matrix()
        >>> display.plot(threshold_value=0.7)
        """
        if data_source == "both":
            raise ValueError(
                "data_source='both' is not supported for confusion_matrix."
            )

        if hasattr(self._parent._estimator, "predict_proba") or hasattr(
            self._parent._estimator, "decision_function"
        ):
            y_scores = self._parent._get_predictions(
                data_source=data_source,
                response_method=("predict_proba", "decision_function"),
            )
        else:
            y_scores = None

        display_kwargs: dict = {
            "report_pos_label": self._parent.pos_label,
            "y_scores": y_scores,
        }
        display = cast(
            ConfusionMatrixDisplay,
            self._get_display(
                data_source=data_source,
                response_method="predict",
                display_class=ConfusionMatrixDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display
