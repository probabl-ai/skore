import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import sklearn
from numpy.typing import ArrayLike
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor, _get_cached_response_values
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.types import (
    DataSource,
    Metric,
    PositiveLabel,
)
from skore._utils._accessor import (
    _check_all_checks,
    _check_estimator_has_method,
    _check_roc_auc,
    _check_supported_ml_task,
)
from skore._utils._cache_key import make_cache_key


class _MetricsAccessor(_BaseAccessor[EstimatorReport], DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    _score_or_loss_info: dict[str, dict[str, str]] = {
        "fit_time": {"name": "Fit time (s)", "icon": "(↘︎)"},
        "predict_time": {"name": "Predict time (s)", "icon": "(↘︎)"},
        "accuracy": {"name": "Accuracy", "icon": "(↗︎)"},
        "precision": {"name": "Precision", "icon": "(↗︎)"},
        "recall": {"name": "Recall", "icon": "(↗︎)"},
        "brier_score": {"name": "Brier score", "icon": "(↘︎)"},
        "roc_auc": {"name": "ROC AUC", "icon": "(↗︎)"},
        "log_loss": {"name": "Log loss", "icon": "(↘︎)"},
        "r2": {"name": "R²", "icon": "(↗︎)"},
        "rmse": {"name": "RMSE", "icon": "(↘︎)"},
        "custom_metric": {"name": "Custom metric", "icon": ""},
    }

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    def summarize(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        metric: Metric | list[Metric] | dict[str, Metric] | None = None,
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
        >>> report.metrics.summarize(
        ...     metric=["f1"],
        ... ).frame(favorability=True)
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
        >>> # Using scikit-learn metrics
        >>> report.metrics.summarize(
        ...     metric=["f1"],
        ... ).frame(favorability=True)
                                  LogisticRegression Favorability
        Metric   Label / Average
        F1 Score               1             0.95...          (↗︎)
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

        pos_label = self._parent.pos_label

        # Handle dictionary metrics
        metric_names: list[str | None] | None = None
        if isinstance(metric, dict):
            metric_names = list(metric.keys())
            metrics = list(metric.values())
        elif metric is not None and not isinstance(metric, list):
            metrics = [metric]
        elif isinstance(metric, list):
            metrics = metric

        # Treat empty list same as None - use defaults
        if metric is None or (isinstance(metric, list) and len(metric) == 0):
            if "classification" in self._parent._ml_task:
                default_metric_names: tuple[str, ...] = (
                    "accuracy",
                    "precision",
                    "recall",
                    "roc_auc",
                    "log_loss",
                    "brier_score",
                )
            else:  # regression
                default_metric_names = ("r2", "rmse")
            metrics = [m for m in default_metric_names if hasattr(self, m)]
            metrics += ["fit_time", "predict_time"]

        if metric_names is None:
            metric_names = [None] * len(metrics)

        rows = []
        for metric_name, metric_ in zip(metric_names, metrics, strict=True):
            if isinstance(metric_, str) and metric_ not in self._score_or_loss_info:
                try:
                    metric_ = sklearn.metrics.get_scorer(metric_)
                except ValueError as err:
                    raise ValueError(
                        f"Invalid metric: {metric_!r}. "
                        f"Please use a valid metric from the "
                        f"list of supported metrics: "
                        f"{list(self._score_or_loss_info.keys())} "
                        "or a valid scikit-learn metric string."
                    ) from err
                if metric_kwargs is not None:
                    raise ValueError(
                        "The `metric_kwargs` parameter is not supported when "
                        "`metric` is a scikit-learn scorer name. Use the function "
                        "`sklearn.metrics.make_scorer` to create a scorer with "
                        "additional parameters."
                    )

            # NOTE: we have to check specifically for `_BaseScorer` first because this
            # is also a callable but it has a special private API that we can leverage
            if isinstance(metric_, _BaseScorer):
                # scorers have the advantage to have scoped defined kwargs
                metric_fn = partial(
                    self.custom_metric,
                    metric_function=metric_._score_func,
                    response_method=metric_._response_method,
                )
                # forward the additional parameters specific to the scorer
                metrics_kwargs = {**metric_._kwargs}
                metrics_params = inspect.signature(metric_._score_func).parameters
                if "pos_label" in metrics_params:
                    if (
                        "pos_label" in metrics_kwargs
                        and pos_label != metrics_kwargs["pos_label"]
                    ):
                        raise ValueError(
                            "`pos_label` is passed both in the scorer: "
                            f"{metrics_kwargs['pos_label']!r} and when creating "
                            f"the report: {pos_label!r}. Please provide a consistent "
                            "`pos_label` or only pass it whether in the scorer or "
                            "when creating the report."
                        )
                    metrics_kwargs["pos_label"] = pos_label

                metric_favorability = "(↗︎)" if metric_._sign == 1 else "(↘︎)"
                if metric_name is None:
                    metric_name = metric_._score_func.__name__.replace("_", " ").title()

            elif isinstance(metric_, str) or callable(metric_):
                if isinstance(metric_, str):
                    metric_fn = getattr(
                        self,
                        (
                            f"_{metric_}"
                            if metric_ in ["fit_time", "predict_time"]
                            else metric_
                        ),
                    )
                    metrics_kwargs = {}
                    if metric_name is None:
                        metric_name = self._score_or_loss_info[metric_]["name"]
                    metric_favorability = self._score_or_loss_info[metric_]["icon"]
                else:
                    # Handle callable metrics
                    if response_method is None:
                        if (
                            metric_kwargs is None
                            or "response_method" not in metric_kwargs
                        ):
                            raise ValueError(
                                "response_method is required when the metric is a "
                                "callable. Pass it directly or through `metric_kwargs`."
                            )

                        response_method = metric_kwargs["response_method"]

                    metric_fn = partial(
                        self.custom_metric,
                        metric_function=metric_,
                        response_method=response_method,
                    )
                    if metric_kwargs is None:
                        metrics_kwargs = {}
                    else:
                        # check if we should pass any parameters specific to the metric
                        # callable
                        metric_callable_params = inspect.signature(metric_).parameters
                        metrics_kwargs = {
                            param: metric_kwargs[param]
                            for param in metric_callable_params
                            if param in metric_kwargs
                        }
                    if metric_name is None:
                        metric_name = metric_.__name__.replace("_", " ").title()
                    metric_favorability = ""

                metrics_params = inspect.signature(metric_fn).parameters
                if metric_kwargs is not None:
                    for param in metrics_params:
                        if param in metric_kwargs:
                            metrics_kwargs[param] = metric_kwargs[param]
                if "pos_label" in metrics_params:
                    metrics_kwargs["pos_label"] = pos_label
            else:
                raise ValueError(
                    f"Invalid type of metric: {type(metric_)} for {metric_!r}"
                )

            score = metric_fn(data_source=data_source, **metrics_kwargs)

            row = {
                "metric": metric_name,
                "estimator_name": self._parent.estimator_name_,
                "data_source": data_source,
                "favorability": metric_favorability,
                "label": None,
                "average": None,
                "output": None,
                "score": score,
            }

            if (
                self._parent._ml_task == "binary-classification"
                and metrics_kwargs.get("average") == "binary"
            ):
                rows.append({**row, "label": pos_label})
            elif self._parent._ml_task in (
                "binary-classification",
                "multiclass-classification",
            ):
                if isinstance(score, dict):
                    for label in score:
                        rows.append({**row, "label": label, "score": score[label]})  # noqa: PERF401
                else:
                    rows.append({**row, "average": metrics_kwargs.get("average")})
            elif self._parent._ml_task == "multioutput-regression":
                if isinstance(score, list):
                    for output_idx, output_score in enumerate(score):
                        rows.append(
                            {**row, "output": output_idx, "score": output_score}
                        )
                else:
                    rows.append({**row, "average": metrics_kwargs.get("multioutput")})
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

    def _compute_metric_scores(
        self,
        metric_fn: Callable,
        *,
        response_method: str | list[str] | tuple[str, ...],
        data_source: DataSource = "test",
        **metric_kwargs: Any,
    ) -> float | dict[PositiveLabel, float] | list:
        X, y_true = self._get_X_y(data_source=data_source)

        pos_label = self._parent.pos_label

        cache_key = make_cache_key(data_source, metric_fn.__name__, metric_kwargs)

        score = self._parent._cache.get(cache_key)
        if score is None:
            results = _get_cached_response_values(
                cache=self._parent._cache,
                estimator=self._parent.estimator_,
                X=X,
                response_method=response_method,
                pos_label=pos_label,
                data_source=data_source,
            )
            for key_tuple, value, is_cached in results:
                if not is_cached:
                    self._parent._cache[key_tuple] = value
                if key_tuple[1] != "predict_time":
                    y_pred = value

            metric_params = inspect.signature(metric_fn).parameters
            kwargs = {**metric_kwargs}
            if "pos_label" in metric_params and "pos_label" not in kwargs:
                kwargs.update(pos_label=pos_label)
            score = metric_fn(y_true, y_pred, **kwargs)

            if isinstance(score, np.ndarray):
                score = score.tolist()

            if hasattr(score, "item"):
                score = score.item()
            elif isinstance(score, list):
                if len(score) == 1:
                    score = score[0]
                elif "classification" in self._parent._ml_task:
                    score = dict(
                        zip(
                            self._parent._estimator.classes_.tolist(),
                            score,
                            strict=False,
                        )
                    )

            self._parent._cache[cache_key] = score

        return score

    def _fit_time(self, cast: bool = True, **kwargs) -> float | None:
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
        if cast and self._parent.fit_time_ is None:
            return float("nan")
        return self._parent.fit_time_

    def _predict_time(
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
        predict_time_cache_key = make_cache_key(data_source, "predict_time")

        return self._parent._cache.get(
            predict_time_cache_key, (float("nan") if cast else None)
        )

    def timings(self) -> dict:
        """Get all measured processing times related to the estimator.

        When an estimator is fitted inside the :class:`~skore.EstimatorReport`, the time
        to fit is recorded. Similarly, when predictions are computed on some data, the
        time to predict is recorded. This function returns all the recorded times.

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
        >>> report.cache_predictions(response_methods=["predict"])
        >>> report.metrics.timings()
        {'fit_time': ..., 'predict_time_test': ...}
        """
        fit_time_ = self._fit_time(cast=False)
        fit_time = {"fit_time": fit_time_} if fit_time_ is not None else {}

        # predict_time cache keys are of the form
        # (data_source, "predict_time", None)
        predict_times = {
            f"predict_time_{data_source}": v
            for (data_source, name, _), v in self._parent._cache.items()
            if name == "predict_time"
        }

        return fit_time | predict_times

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
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
        score = self._compute_metric_scores(
            sklearn.metrics.accuracy_score,
            data_source=data_source,
            response_method="predict",
        )
        return cast(float, score)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
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
        pos_label = self._parent.pos_label

        if self._parent._ml_task == "binary-classification" and pos_label is not None:
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        result = self._compute_metric_scores(
            sklearn.metrics.precision_score,
            data_source=data_source,
            response_method="predict",
            average=average,
        )
        if self._parent._ml_task == "binary-classification" and (
            pos_label is not None or average is not None
        ):
            return cast(float, result)
        return cast(dict[PositiveLabel, float], result)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
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
        pos_label = self._parent.pos_label

        if self._parent._ml_task == "binary-classification" and pos_label is not None:
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        result = self._compute_metric_scores(
            sklearn.metrics.recall_score,
            data_source=data_source,
            response_method="predict",
            average=average,
        )
        if self._parent._ml_task == "binary-classification" and (
            pos_label is not None or average is not None
        ):
            return cast(float, result)
        return cast(dict[PositiveLabel, float], result)

    @available_if(
        _check_all_checks(
            checks=[
                _check_supported_ml_task(supported_ml_tasks=["binary-classification"]),
                _check_estimator_has_method(method_name="predict_proba"),
            ]
        )
    )
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
        # The Brier score in scikit-learn request `pos_label` to ensure that the
        # integral encoding of `y_true` corresponds to the probabilities of the
        # `pos_label`. Since we get the predictions with `get_response_method`, we
        # can pass any `pos_label`, they will lead to the same result.
        result = self._compute_metric_scores(
            sklearn.metrics.brier_score_loss,
            data_source=data_source,
            response_method="predict_proba",
            pos_label=self._parent._estimator.classes_[-1],
        )
        return cast(float, result)

    @available_if(
        _check_roc_auc(
            ml_task_and_methods=[
                ("binary-classification", ["predict_proba", "decision_function"]),
                ("multiclass-classification", ["predict_proba"]),
            ]
        )
    )
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
        result = self._compute_metric_scores(
            sklearn.metrics.roc_auc_score,
            data_source=data_source,
            response_method=["predict_proba", "decision_function"],
            average=average,
            multi_class=multi_class,
        )
        if self._parent._ml_task == "multiclass-classification" and average is None:
            return cast(dict[PositiveLabel, float], result)
        return cast(float, result)

    @available_if(
        _check_all_checks(
            checks=[
                _check_supported_ml_task(
                    supported_ml_tasks=[
                        "binary-classification",
                        "multiclass-classification",
                    ]
                ),
                _check_estimator_has_method(method_name="predict_proba"),
            ]
        )
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
        result = self._compute_metric_scores(
            sklearn.metrics.log_loss,
            data_source=data_source,
            response_method="predict_proba",
        )
        return cast(float, result)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
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
        result = self._compute_metric_scores(
            sklearn.metrics.r2_score,
            data_source=data_source,
            response_method="predict",
            multioutput=multioutput,
        )
        if (
            self._parent._ml_task == "multioutput-regression"
            and multioutput == "raw_values"
        ):
            return cast(list, result)
        return cast(float, result)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
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
        result = self._compute_metric_scores(
            sklearn.metrics.root_mean_squared_error,
            data_source=data_source,
            response_method="predict",
            multioutput=multioutput,
        )
        if (
            self._parent._ml_task == "multioutput-regression"
            and multioutput == "raw_values"
        ):
            return cast(list, result)
        return cast(float, result)

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
        if isinstance(metric_function, _BaseScorer):
            metric_function = metric_function._score_func

        return self._compute_metric_scores(
            metric_function,
            data_source=data_source,
            response_method=response_method,
            **kwargs,
        )

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
        X, y_true = self._get_X_y(data_source=data_source)

        results = _get_cached_response_values(
            cache=self._parent._cache,
            estimator=self._parent.estimator_,
            X=X,
            response_method=response_method,
            pos_label=display_kwargs.get("pos_label"),
            data_source=data_source,
        )
        for key, value, is_cached in results:
            if not is_cached:
                self._parent._cache[key] = value
            if key[1] != "predict_time":
                y_pred = value

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
        display_kwargs = {"pos_label": self._parent.pos_label}
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
        display_kwargs = {"pos_label": self._parent.pos_label}
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
        response_method: str | list[str] | tuple[str, ...]
        if self._parent._ml_task == "binary-classification":
            response_method = ("predict_proba", "decision_function")
        else:
            response_method = "predict"

        display_kwargs = {
            "display_labels": tuple(self._parent.estimator_.classes_),
            "pos_label": self._parent.pos_label,
            "response_method": response_method,
        }
        display = cast(
            ConfusionMatrixDisplay,
            self._get_display(
                data_source=data_source,
                response_method=response_method,
                display_class=ConfusionMatrixDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display
