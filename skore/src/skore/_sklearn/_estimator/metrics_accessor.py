import inspect
from collections.abc import Callable, Iterable
from functools import partial
from operator import attrgetter
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn import metrics as sklearn_metrics
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import (
    _BaseAccessor,
    _BaseMetricsAccessor,
    _get_cached_response_values,
)
from skore._sklearn._estimator.report import EstimatorReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore._sklearn.types import (
    _DEFAULT,
    DataSource,
    Metric,
    PositiveLabel,
    YPlotData,
)
from skore._utils._accessor import (
    _check_all_checks,
    _check_estimator_has_method,
    _check_roc_auc,
    _check_supported_ml_task,
)
from skore._utils._index import flatten_multi_index


class _MetricsAccessor(
    _BaseMetricsAccessor, _BaseAccessor["EstimatorReport"], DirNamesMixin
):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    _score_or_loss_info: dict[str, dict[str, str]] = {
        **_BaseMetricsAccessor._score_or_loss_info,
        "confusion_matrix": {"name": "Confusion Matrix", "icon": ""},
    }

    def __init__(self, parent: EstimatorReport) -> None:
        super().__init__(parent)

    def summarize(
        self,
        *,
        data_source: DataSource | Literal["both"] = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        metric: Metric | list[Metric] | dict[str, Metric] | None = None,
        metric_kwargs: dict[str, Any] | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
        indicator_favorability: bool = False,
        flat_index: bool = False,
    ) -> MetricsSummaryDisplay:
        """Report a set of metrics for our estimator.

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

        pos_label : int, float, bool, str or None, default=_DEFAULT
            The label to consider as the positive class when computing the metric. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report. If `None`,
            the metric is computed considering each class as a positive class.

        indicator_favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=False
            Whether to flatten the multi-index columns. Flat index will always be lower
            case, do not include spaces and remove the hash symbol to ease indexing.

        Returns
        -------
        MetricsSummaryDisplay
            A display containing the statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data, pos_label=1)
        >>> report.metrics.summarize(indicator_favorability=True).frame()
                    LogisticRegression Favorability
        Metric
        Precision              0.98...         (↗︎)
        Recall                 0.93...         (↗︎)
        ROC AUC                0.99...         (↗︎)
        Brier score            0.03...         (↘︎)
        >>> # Using scikit-learn metrics
        >>> report.metrics.summarize(
        ...     metric=["f1"],
        ...     indicator_favorability=True,
        ... ).frame()
                                  LogisticRegression Favorability
        Metric   Label / Average
        F1 Score               1             0.96...          (↗︎)
        >>> report.metrics.summarize(
        ...    indicator_favorability=True,
        ...    data_source="both"
        ... ).frame().drop(["Fit time (s)", "Predict time (s)"])
                     LogisticRegression (train)  LogisticRegression (test)  Favorability
        Metric
        Precision                       0.96...                     0.98...          (↗︎)
        Recall                          0.97...                     0.93...          (↗︎)
        ROC AUC                         0.99...                     0.99...          (↗︎)
        Brier score                     0.02...                     0.03...          (↘︎)
        >>> # Using scikit-learn metrics
        >>> report.metrics.summarize(
        ...     metric=["f1"],
        ...     indicator_favorability=True,
        ... ).frame()
                                  LogisticRegression Favorability
        Metric   Label / Average
        F1 Score               1             0.96...          (↗︎)
        """
        if data_source == "both":
            train_summary = self.summarize(
                data_source="train",
                metric=metric,
                metric_kwargs=metric_kwargs,
                pos_label=pos_label,
                indicator_favorability=False,
                flat_index=flat_index,
            )
            test_summary = self.summarize(
                data_source="test",
                metric=metric,
                metric_kwargs=metric_kwargs,
                pos_label=pos_label,
                indicator_favorability=indicator_favorability,
                flat_index=flat_index,
            )
            # Add suffix to the dataframes to distinguish train and test.
            train_df = train_summary.frame().add_suffix(" (train)")
            test_df = test_summary.frame().add_suffix(" (test)")
            combined = pd.concat([train_df, test_df], axis=1).rename(
                columns={"Favorability (test)": "Favorability"}
            )
            return MetricsSummaryDisplay(summarize_data=combined)

        if pos_label is _DEFAULT:
            pos_label = self._parent.pos_label

        # Handle dictionary metrics
        metric_names = None
        if isinstance(metric, dict):
            metric_names = list(metric.keys())
            metrics = list(metric.values())
        elif metric is not None and not isinstance(metric, list):
            metrics = [metric]
        elif isinstance(metric, list):
            metrics = metric

        if data_source == "X_y":
            # optimization of the hash computation to avoid recomputing it
            # FIXME: we are still recomputing the hash for all the metrics that we
            # support in the report because we don't call `_compute_metric_scores`
            # here. We should fix it.
            X, y, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )
        else:
            data_source_hash = None

        if metric is None:
            # Equivalent to _get_scorers_to_add
            if self._parent._ml_task == "binary-classification":
                metrics = ["_precision", "_recall", "_roc_auc"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    metrics.append("_brier_score")
            elif self._parent._ml_task == "multiclass-classification":
                metrics = ["_precision", "_recall"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    metrics += ["_roc_auc", "_log_loss"]
            else:
                metrics = ["_r2", "_rmse"]
            metrics += ["_fit_time", "_predict_time"]

        if metric_names is None:
            metric_names = [None] * len(metrics)  # type: ignore

        scores = []
        favorability_indicator = []
        for metric_name, metric_ in zip(metric_names, metrics, strict=False):
            if isinstance(metric_, str) and not (
                (metric_.startswith("_") and metric_[1:] in self._score_or_loss_info)
                or metric_ in self._score_or_loss_info
            ):
                try:
                    metric_ = sklearn_metrics.get_scorer(metric_)
                except ValueError as err:
                    raise ValueError(
                        f"Invalid metric: {metric_!r}. "
                        f"Please use a valid metric from the "
                        f"list of supported metrics: "
                        f"{list(self._score_or_loss_info.keys())} "
                        "or a valid scikit-learn scoring string."
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
                metric_function: Callable = metric_._score_func
                response_method: str | list[str] = metric_._response_method
                metric_fn = partial(
                    self._custom_metric,
                    metric_function=metric_function,
                    response_method=response_method,
                )
                # forward the additional parameters specific to the scorer
                metrics_kwargs = {**metric_._kwargs}
                metrics_kwargs["data_source_hash"] = data_source_hash
                metrics_params = inspect.signature(metric_._score_func).parameters
                if "pos_label" in metrics_params:
                    if pos_label is not None and "pos_label" in metrics_kwargs:
                        if pos_label != metrics_kwargs["pos_label"]:
                            raise ValueError(
                                "`pos_label` is passed both in the scorer and to the "
                                "`summarize` method. Please provide a consistent "
                                "`pos_label` or only pass it whether in the scorer or "
                                "to the `summarize` method."
                            )
                    elif pos_label is not None:
                        metrics_kwargs["pos_label"] = pos_label
                if metric_name is None:
                    metric_name = metric_._score_func.__name__.replace("_", " ").title()
                metric_favorability = "(↗︎)" if metric_._sign == 1 else "(↘︎)"
                favorability_indicator.append(metric_favorability)
            elif isinstance(metric_, str) or callable(metric_):
                if isinstance(metric_, str):
                    # Handle built-in metrics (with underscore prefix)
                    if (
                        metric_.startswith("_")
                        and metric_[1:] in self._score_or_loss_info
                    ):
                        metric_fn = getattr(self, metric_)
                        metrics_kwargs = {"data_source_hash": data_source_hash}
                        if metric_name is None:
                            metric_name = (
                                f"{self._score_or_loss_info[metric_[1:]]['name']}"
                            )
                        metric_favorability = self._score_or_loss_info[metric_[1:]][
                            "icon"
                        ]

                    # Handle built-in metrics (without underscore prefix)
                    elif metric_ in self._score_or_loss_info:
                        metric_fn = getattr(self, f"_{metric_}")
                        metrics_kwargs = {"data_source_hash": data_source_hash}
                        if metric_name is None:
                            metric_name = f"{self._score_or_loss_info[metric_]['name']}"
                        metric_favorability = self._score_or_loss_info[metric_]["icon"]
                else:
                    # Handle callable metrics
                    metric_fn = partial(self._custom_metric, metric_function=metric_)
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
                    metrics_kwargs["data_source_hash"] = data_source_hash
                    if metric_name is None:
                        metric_name = metric_.__name__
                    metric_favorability = ""
                    favorability_indicator.append(metric_favorability)

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

            score = metric_fn(data_source=data_source, X=X, y=y, **metrics_kwargs)

            index: pd.Index | pd.MultiIndex | list[str] | None
            score_array: NDArray
            if self._parent._ml_task == "binary-classification":
                if isinstance(score, dict):
                    classes = list(score.keys())
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name] * len(classes), classes],
                        names=["Metric", "Label / Average"],
                    )
                    score_array = np.hstack([score[c] for c in classes]).reshape(-1, 1)
                elif "average" in metrics_kwargs:
                    if metrics_kwargs["average"] == "binary":
                        index = pd.MultiIndex.from_arrays(
                            [[metric_name], [pos_label]],
                            names=["Metric", "Label / Average"],
                        )
                    elif metrics_kwargs["average"] is not None:
                        index = pd.MultiIndex.from_arrays(
                            [[metric_name], [metrics_kwargs["average"]]],
                            names=["Metric", "Label / Average"],
                        )
                    else:
                        index = pd.Index([metric_name], name="Metric")
                    score_array = np.array(score).reshape(-1, 1)
                else:
                    index = pd.Index([metric_name], name="Metric")
                    score_array = np.array(score).reshape(-1, 1)
            elif self._parent._ml_task == "multiclass-classification":
                if isinstance(score, dict):
                    classes = list(score.keys())
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name] * len(classes), classes],
                        names=["Metric", "Label / Average"],
                    )
                    score_array = np.hstack([score[c] for c in classes]).reshape(-1, 1)
                elif (
                    "average" in metrics_kwargs
                    and metrics_kwargs["average"] is not None
                ):
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name], [metrics_kwargs["average"]]],
                        names=["Metric", "Label / Average"],
                    )
                    score_array = np.array(score).reshape(-1, 1)
                else:
                    index = pd.Index([metric_name], name="Metric")
                    score_array = np.array(score).reshape(-1, 1)
            elif self._parent._ml_task in ("regression", "multioutput-regression"):
                if isinstance(score, list):
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name] * len(score), list(range(len(score)))],
                        names=["Metric", "Output"],
                    )
                    score_array = np.array(score).reshape(-1, 1)
                else:
                    index = pd.Index([metric_name], name="Metric")
                    score_array = np.array(score).reshape(-1, 1)
            else:  # unknown task - try our best
                index = None if isinstance(score, Iterable) else [metric_name]

            score_df = pd.DataFrame(
                score_array, index=index, columns=[self._parent.estimator_name_]
            )
            if indicator_favorability:
                score_df["Favorability"] = metric_favorability

            scores.append(score_df)

        has_multilevel = any(
            isinstance(df, pd.DataFrame) and isinstance(df.index, pd.MultiIndex)
            for df in scores
        )

        if has_multilevel:
            # Convert single-level dataframes to multi-level
            for i, df in enumerate(scores):
                if hasattr(df, "index") and not isinstance(df.index, pd.MultiIndex):
                    if self._parent._ml_task in (
                        "regression",
                        "multioutput-regression",
                    ):
                        name_index = ["Metric", "Output"]
                    else:
                        name_index = ["Metric", "Label / Average"]

                    scores[i].index = pd.MultiIndex.from_tuples(
                        [(idx, "") for idx in df.index],
                        names=name_index,
                    )

        results = pd.concat(scores, axis=0)
        if flat_index:
            if isinstance(results.columns, pd.MultiIndex):
                results.columns = flatten_multi_index(results.columns)
            if isinstance(results.index, pd.MultiIndex):
                results.index = flatten_multi_index(results.index)
            if isinstance(results.index, pd.Index):
                results.index = results.index.str.replace(
                    r"\((.*)\)$", r"\1", regex=True
                )
        return MetricsSummaryDisplay(summarize_data=results)

    def _compute_metric_scores(
        self,
        metric_fn: Callable,
        X: ArrayLike | None,
        y_true: ArrayLike | None,
        *,
        response_method: str | list[str] | tuple[str, ...],
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        pos_label: PositiveLabel | None = None,
        **metric_kwargs: Any,
    ) -> float | dict[PositiveLabel, float] | list:
        if data_source_hash is None:
            X, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y_true
            )

        # build the cache key components to finally create a tuple that will be used
        # to check if the metric has already been computed
        cache_key_parts: list[Any] = [
            self._parent._hash,
            metric_fn.__name__,
            data_source,
        ]

        if data_source_hash is not None:
            cache_key_parts.append(data_source_hash)

        metric_params = inspect.signature(metric_fn).parameters
        if "pos_label" in metric_params:
            cache_key_parts.append(pos_label)

        # add the ordered metric kwargs to the cache key
        ordered_metric_kwargs = sorted(metric_kwargs.keys())
        for key in ordered_metric_kwargs:
            value = metric_kwargs[key]
            if isinstance(value, np.ndarray):
                cache_key_parts.append(joblib.hash(value))
            else:
                cache_key_parts.append(value)

        cache_key = tuple(cache_key_parts)

        if cache_key in self._parent._cache:
            score = self._parent._cache[cache_key]
        else:
            metric_params = inspect.signature(metric_fn).parameters
            kwargs = {**metric_kwargs}
            if "pos_label" in metric_params:
                kwargs.update(pos_label=pos_label)

            results = _get_cached_response_values(
                cache=self._parent._cache,
                estimator_hash=int(self._parent._hash),
                estimator=self._parent.estimator_,
                X=X,
                response_method=response_method,
                pos_label=pos_label,
                data_source=data_source,
                data_source_hash=data_source_hash,
            )
            for key_tuple, value, is_cached in results:
                if not is_cached:
                    self._parent._cache[key_tuple] = value
                if key_tuple[-1] != "predict_time":
                    y_pred = value

            score = metric_fn(y_true, y_pred, **kwargs)

            if isinstance(score, np.ndarray):
                score = score.tolist()
            elif hasattr(score, "item"):
                score = score.item()

            self._parent._cache[cache_key] = score

        if isinstance(score, list):
            if "classification" in self._parent._ml_task:
                return dict(
                    zip(self._parent._estimator.classes_.tolist(), score, strict=False)
                )

            if len(score) == 1:
                return score[0]
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
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        cast: bool = True,
    ) -> float | None:
        """Get prediction time if it has been already measured.

        Parameters
        ----------
        cast : bool, default=True
            Whether to cast the numbers to floats. If `False`, the return value
            is `None` when the predictions have never been computed.
        """
        if data_source_hash is None:
            X, _, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

        predict_time_cache_key = (
            self._parent._hash,
            data_source,
            data_source_hash,
            "predict_time",
        )

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
            - "predict_time_{data_source}", where data_source is "train", "test" or
              "X_y_{data_source_hash}", for the time to compute the predictions on the
              given data source.

        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> from skore import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> X, y = make_classification(random_state=42)
        >>> split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
        >>> estimator = LogisticRegression()
        >>> from skore import EstimatorReport
        >>> report = EstimatorReport(estimator, **split_data)
        >>> report.metrics.timings()
        {'fit_time': ...}
        >>> report.cache_predictions(response_methods=["predict"])
        >>> report.metrics.timings()
        {'fit_time': ..., 'predict_time_test': ...}
        """
        fit_time_ = self._fit_time(cast=False)
        fit_time = {"fit_time": fit_time_} if fit_time_ is not None else {}

        # predict_time cache keys are of the form
        # (self._parent._hash, data_source, data_source_hash, "predict_time")

        def make_key(k: tuple) -> str:
            data_source, data_source_hash = k[1], k[2]
            if data_source == "X_y":
                return f"predict_time_X_y_{data_source_hash}"
            return f"predict_time_{data_source}"

        predict_times = {
            make_key(k): v
            for k, v in self._parent._cache.items()
            if k[-1] == "predict_time"
        }

        return fit_time | predict_times

    @available_if(attrgetter("_accuracy"))
    def accuracy(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
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

        Returns
        -------
        float
            The accuracy score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.metrics.accuracy()
        0.95...
        """
        return self._accuracy(data_source=data_source, data_source_hash=None, X=X, y=y)

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def _accuracy(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Private interface of `accuracy` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        score = self._compute_metric_scores(
            sklearn_metrics.accuracy_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
        )
        return cast(float, score)

    @available_if(attrgetter("_precision"))
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
    ) -> float | dict[PositiveLabel, float]:
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

        Returns
        -------
        float or dict
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.metrics.precision(pos_label=1)
        0.98...
        """
        return self._precision(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            average=average,
            pos_label=pos_label,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def _precision(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> float | dict[PositiveLabel, float]:
        """Private interface of `precision` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        if pos_label is _DEFAULT:
            pos_label = self._parent.pos_label

        if self._parent._ml_task == "binary-classification" and pos_label is not None:
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        result = self._compute_metric_scores(
            sklearn_metrics.precision_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            pos_label=pos_label,
            average=average,
        )
        if self._parent._ml_task == "binary-classification" and (
            pos_label is not None or average is not None
        ):
            return cast(float, result)
        return cast(dict[PositiveLabel, float], result)

    @available_if(attrgetter("_recall"))
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
    ) -> float | dict[PositiveLabel, float]:
        """Compute the recall score.

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

        Returns
        -------
        float or dict
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.metrics.recall(pos_label=1)
        0.93...
        """
        return self._recall(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            average=average,
            pos_label=pos_label,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def _recall(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: (
            Literal["binary", "macro", "micro", "weighted", "samples"] | None
        ) = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> float | dict[PositiveLabel, float]:
        """Private interface of `recall` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        if pos_label is _DEFAULT:
            pos_label = self._parent.pos_label

        if self._parent._ml_task == "binary-classification" and pos_label is not None:
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        result = self._compute_metric_scores(
            sklearn_metrics.recall_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            pos_label=pos_label,
            average=average,
        )
        if self._parent._ml_task == "binary-classification" and (
            pos_label is not None or average is not None
        ):
            return cast(float, result)
        return cast(dict[PositiveLabel, float], result)

    @available_if(attrgetter("_brier_score"))
    def brier_score(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Compute the Brier score.

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

        Returns
        -------
        float
            The Brier score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.metrics.brier_score()
        0.03...
        """
        return self._brier_score(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
        )

    @available_if(
        _check_all_checks(
            checks=[
                _check_supported_ml_task(supported_ml_tasks=["binary-classification"]),
                _check_estimator_has_method(method_name="predict_proba"),
            ]
        )
    )
    def _brier_score(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Private interface of `brier_score` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        # The Brier score in scikit-learn request `pos_label` to ensure that the
        # integral encoding of `y_true` corresponds to the probabilities of the
        # `pos_label`. Since we get the predictions with `get_response_method`, we
        # can pass any `pos_label`, they will lead to the same result.
        result = self._compute_metric_scores(
            sklearn_metrics.brier_score_loss,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict_proba",
            pos_label=self._parent._estimator.classes_[-1],
        )
        return cast(float, result)

    @available_if(attrgetter("_roc_auc"))
    def roc_auc(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: Literal["macro", "micro", "weighted", "samples"] | None = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
    ) -> float | dict[PositiveLabel, float]:
        """Compute the ROC AUC score.

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
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.metrics.roc_auc()
        0.99...
        """
        return self._roc_auc(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            average=average,
            multi_class=multi_class,
        )

    @available_if(
        _check_roc_auc(
            ml_task_and_methods=[
                ("binary-classification", ["predict_proba", "decision_function"]),
                ("multiclass-classification", ["predict_proba"]),
            ]
        )
    )
    def _roc_auc(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        average: Literal["macro", "micro", "weighted", "samples"] | None = None,
        multi_class: Literal["raise", "ovr", "ovo"] = "ovr",
    ) -> float | dict[PositiveLabel, float]:
        """Private interface of `roc_auc` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        result = self._compute_metric_scores(
            sklearn_metrics.roc_auc_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method=["predict_proba", "decision_function"],
            average=average,
            multi_class=multi_class,
        )
        if self._parent._ml_task == "multiclass-classification" and average is None:
            return cast(dict[PositiveLabel, float], result)
        return cast(float, result)

    @available_if(attrgetter("_log_loss"))
    def log_loss(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Compute the log loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            New data on which to compute the metric. By default, we use the validation
            set provided when creating the report.

        y : array-like of shape (n_samples,), default=None
            New target on which to compute the metric. By default, we use the target
            provided when creating the report.

        Returns
        -------
        float
            The log-loss.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> report.metrics.log_loss()
        0.10...
        """
        return self._log_loss(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
        )

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
    def _log_loss(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> float:
        """Private interface of `log_loss` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        result = self._compute_metric_scores(
            sklearn_metrics.log_loss,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict_proba",
        )
        return cast(float, result)

    @available_if(attrgetter("_r2"))
    def r2(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list:
        """Compute the R² score.

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
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)
        >>> report.metrics.r2()
        0.35...
        """
        return self._r2(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            multioutput=multioutput,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def _r2(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list:
        """Private interface of `r2` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        result = self._compute_metric_scores(
            sklearn_metrics.r2_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            multioutput=multioutput,
        )
        if (
            self._parent._ml_task == "multioutput-regression"
            and multioutput == "raw_values"
        ):
            return cast(list, result)
        return cast(float, result)

    @available_if(attrgetter("_rmse"))
    def rmse(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list:
        """Compute the root mean squared error.

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
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)
        >>> report.metrics.rmse()
        56.5...
        """
        return self._rmse(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            multioutput=multioutput,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["regression", "multioutput-regression"]
        )
    )
    def _rmse(
        self,
        *,
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        multioutput: (
            Literal["raw_values", "uniform_average"] | ArrayLike
        ) = "raw_values",
    ) -> float | list:
        """Private interface of `rmse` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        result = self._compute_metric_scores(
            sklearn_metrics.root_mean_squared_error,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
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
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
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

        response_method : str or list of str
            The estimator's method to be invoked to get the predictions. The possible
            values are: `predict`, `predict_proba`, `predict_log_proba`, and
            `decision_function`.

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
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)
        >>> report.metrics.custom_metric(
        ...     metric_function=mean_absolute_error,
        ...     response_method="predict",
        ... )
        44.9...
        >>> def metric_function(y_true, y_pred):
        ...     return {"output": float(mean_absolute_error(y_true, y_pred))}
        >>> report.metrics.custom_metric(
        ...     metric_function=metric_function,
        ...     response_method="predict",
        ... )
        {'output': 44.9...}
        """
        return self._custom_metric(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            metric_function=metric_function,
            response_method=response_method,
            **kwargs,
        )

    def _custom_metric(
        self,
        *,
        metric_function: Callable,
        response_method: str | list[str],
        data_source: DataSource = "test",
        data_source_hash: int | None = None,
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        **kwargs: Any,
    ) -> Any:
        """Private interface of `custom_metric` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        pos_label = kwargs.pop("pos_label", self._parent.pos_label)

        if isinstance(metric_function, _BaseScorer):
            metric_function = metric_function._score_func

        return self._compute_metric_scores(
            metric_function,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method=response_method,
            pos_label=pos_label,
            **kwargs,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _get_methods_for_help(self) -> list[tuple[str, Callable]]:
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.EstimatorReport.metrics")

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    def _get_display(
        self,
        *,
        X: ArrayLike | None,
        y: ArrayLike | None,
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
        X : array-like of shape (n_samples, n_features)
            The data.

        y : array-like of shape (n_samples,)
            The target.

        data_source : {"test", "train", "X_y", "both"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.
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
        pos_label = display_kwargs.get("pos_label")

        def get_ys(
            *,
            X,
            y_true,
            data_source,
            data_source_hash,
            cache=self._parent._cache,
            estimator_hash=int(self._parent._hash),
            estimator=self._parent.estimator_,
            estimator_name=self._parent.estimator_name_,
            response_method=response_method,
            pos_label=pos_label,
        ) -> tuple[list[YPlotData], list[YPlotData]]:
            """Get predictions and format y_true and y_pred using YPlotData."""
            results = _get_cached_response_values(
                cache=cache,
                estimator_hash=estimator_hash,
                estimator=estimator,
                X=X,
                response_method=response_method,
                pos_label=pos_label,
                data_source=data_source,
                data_source_hash=data_source_hash,
            )
            for key, value, is_cached in results:
                key = cast(tuple[Any, ...], key)
                if not is_cached:
                    cache[key] = value
                if key[-1] != "predict_time":
                    y_pred = value

            y_true = [
                YPlotData(
                    estimator_name=estimator_name,
                    data_source=data_source,
                    split=None,
                    y=y_true,
                )
            ]
            y_pred = [
                YPlotData(
                    estimator_name=estimator_name,
                    data_source=data_source,
                    split=None,
                    y=y_pred,
                )
            ]
            return y_true, y_pred

        if data_source == "both":
            X_train, y_train, data_source_hash_train = (
                self._get_X_y_and_data_source_hash(data_source="train", X=X, y=y)
            )
            y_train_true, y_train_pred = get_ys(
                X=X_train,
                y_true=y_train,
                data_source="train",
                data_source_hash=data_source_hash_train,
            )
            assert y_train_true is not None, "y must be provided"

            X_test, y_test, data_source_hash_test = self._get_X_y_and_data_source_hash(
                data_source="test", X=X, y=y
            )
            y_test_true, y_test_pred = get_ys(
                X=X_test,
                y_true=y_test,
                data_source="test",
                data_source_hash=data_source_hash_test,
            )
            assert y_test_true is not None, "y must be provided"

            y_true = y_train_true + y_test_true
            y_pred = y_train_pred + y_test_pred
            data_source_hash = None
        else:
            X, y_data_source, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y
            )

            y_true, y_pred = get_ys(
                X=X,
                y_true=y_data_source,
                data_source=data_source,
                data_source_hash=data_source_hash,
            )

            assert y_true is not None, "y must be provided"

        # Compute cache key
        if "seed" in display_kwargs and display_kwargs["seed"] is None:
            cache_key = None
        else:
            cache_key_parts: list[Any] = [self._parent._hash, display_class.__name__]
            for kwarg in display_kwargs.values():
                # NOTE: We cannot use lists in cache keys because they are not hashable
                if isinstance(kwarg, list):
                    kwarg = tuple(kwarg)
                cache_key_parts.append(kwarg)
            if data_source_hash is not None:
                cache_key_parts.append(data_source_hash)
            else:
                cache_key_parts.append(data_source)
            cache_key = tuple(cache_key_parts)

        if cache_key and cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
        else:
            display = display_class._compute_data_for_display(
                y_true=y_true,
                y_pred=y_pred,
                report_type="estimator",
                estimators=[self._parent.estimator_],
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
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> RocCurveDisplay:
        """Plot the ROC curve.

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
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.roc()
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        if pos_label is _DEFAULT:
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
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.precision_recall()
        >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
        """
        if pos_label is _DEFAULT:
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
        data_source: DataSource | Literal["both"] = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        subsample: float | int | None = 1_000,
        seed: int | None = None,
    ) -> PredictionErrorDisplay:
        """Plot the prediction error of a regression model.

        Extra keyword arguments will be passed to matplotlib's `plot`.

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
        >>> from skore import EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> regressor = Ridge()
        >>> report = EstimatorReport(regressor, **split_data)
        >>> display = report.metrics.prediction_error()
        >>> display.plot(perfect_model_kwargs={"color": "tab:red"})
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

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def confusion_matrix(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        pos_label: PositiveLabel | None = _DEFAULT,
    ) -> ConfusionMatrixDisplay:
        """Plot the confusion matrix.

        The confusion matrix shows the counts of correct and incorrect classifications
        for each class.

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
            The label to consider as the positive class when displaying the matrix. Use
            this parameter to override the positive class. By default, the positive
            class is set to the one provided when creating the report.

        Returns
        -------
        display : :class:`~skore._sklearn._plot.ConfusionMatrixDisplay`
            The confusion matrix display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from skore import train_test_split
        >>> from skore import EstimatorReport
        >>> X, y = load_breast_cancer(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, random_state=0, as_dict=True)
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(classifier, **split_data)
        >>> display = report.metrics.confusion_matrix()
        >>> display.plot()

        With specific threshold for binary classification:

        >>> display = report.metrics.confusion_matrix()
        >>> display.plot(threshold_value=0.7)
        """
        if pos_label is _DEFAULT:
            pos_label = self._parent.pos_label

        response_method: str | list[str] | tuple[str, ...]
        if self._parent._ml_task == "binary-classification":
            response_method = ("predict_proba", "decision_function")
        else:
            response_method = "predict"

        display_kwargs = {
            "display_labels": self._parent.estimator_.classes_.tolist(),
            "pos_label": pos_label,
            "response_method": response_method,
        }
        display = cast(
            ConfusionMatrixDisplay,
            self._get_display(
                X=X,
                y=y,
                data_source=data_source,
                response_method=response_method,
                display_class=ConfusionMatrixDisplay,
                display_kwargs=display_kwargs,
            ),
        )
        return display
