import inspect
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.metaestimators import available_if

from skore.externals._pandas_accessors import DirNamesMixin
from skore.sklearn._base import _BaseAccessor, _get_cached_response_values
from skore.sklearn._plot import (
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
)
from skore.utils._accessor import _check_supported_ml_task
from skore.utils._index import flatten_multi_index


class _MetricsAccessor(_BaseAccessor, DirNamesMixin):
    """Accessor for metrics-related operations.

    You can access this accessor using the `metrics` attribute.
    """

    _SCORE_OR_LOSS_INFO = {
        "accuracy": {"name": "Accuracy", "icon": "(↗︎)"},
        "precision": {"name": "Precision", "icon": "(↗︎)"},
        "recall": {"name": "Recall", "icon": "(↗︎)"},
        "brier_score": {"name": "Brier score", "icon": "(↘︎)"},
        "roc_auc": {"name": "ROC AUC", "icon": "(↗︎)"},
        "log_loss": {"name": "Log loss", "icon": "(↘︎)"},
        "r2": {"name": "R²", "icon": "(↗︎)"},
        "rmse": {"name": "RMSE", "icon": "(↘︎)"},
        "custom_metric": {"name": "Custom metric", "icon": ""},
        "report_metrics": {"name": "Report metrics", "icon": ""},
    }

    def __init__(self, parent):
        super().__init__(parent)

    def report_metrics(
        self,
        *,
        data_source="test",
        X=None,
        y=None,
        scoring=None,
        scoring_names=None,
        scoring_kwargs=None,
        pos_label=None,
        indicator_favorability=False,
        flat_index=False,
    ):
        """Report a set of metrics for our estimator.

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

        scoring : list of str, callable, or scorer, default=None
            The metrics to report. You can get the possible list of string by calling
            `report.metrics.help()`. When passing a callable, it should take as
            arguments `y_true`, `y_pred` as the two first arguments. Additional
            arguments can be passed as keyword arguments and will be forwarded with
            `scoring_kwargs`. If the callable API is too restrictive (e.g. need to pass
            same parameter name with different values), you can use scikit-learn scorers
            as provided by :func:`sklearn.metrics.make_scorer`.

        scoring_names : list of str, default=None
            Used to overwrite the default scoring names in the report. It should be of
            the same length as the `scoring` parameter.

        scoring_kwargs : dict, default=None
            The keyword arguments to pass to the scoring functions.

        pos_label : int, float, bool or str, default=None
            The positive class.

        indicator_favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=False
            Whether to flatten the multiindex columns. Flat index will always be lower
            case, do not include spaces and remove the hash symbol to ease indexing.

        Returns
        -------
        pd.DataFrame
            The statistics for the metrics.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.metrics.report_metrics(pos_label=1, indicator_favorability=True)
                    LogisticRegression Favorability
        Metric
        Precision              0.98...         (↗︎)
        Recall                 0.93...         (↗︎)
        ROC AUC                0.99...         (↗︎)
        Brier score            0.03...         (↘︎)
        """
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

        scoring_was_none = scoring is None
        if scoring is None:
            # Equivalent to _get_scorers_to_add
            if self._parent._ml_task == "binary-classification":
                scoring = ["_precision", "_recall", "_roc_auc"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    scoring.append("_brier_score")
            elif self._parent._ml_task == "multiclass-classification":
                scoring = ["_precision", "_recall"]
                if hasattr(self._parent._estimator, "predict_proba"):
                    scoring += ["_roc_auc", "_log_loss"]
            else:
                scoring = ["_r2", "_rmse"]

        if scoring_names is not None and len(scoring_names) != len(scoring):
            if scoring_was_none:
                # we raise a better error message since we decide the default scores
                raise ValueError(
                    "The `scoring_names` parameter should be of the same length as "
                    "the `scoring` parameter. In your case, `scoring` was set to None "
                    f"and you are using our default scores that are {len(scoring)}. "
                    "The list is the following: {scoring}."
                )
            else:
                raise ValueError(
                    "The `scoring_names` parameter should be of the same length as "
                    f"the `scoring` parameter. Got {len(scoring_names)} names for "
                    f"{len(scoring)} scoring functions."
                )
        elif scoring_names is None:
            scoring_names = [None] * len(scoring)

        scores = []
        favorability_indicator = []
        for metric_name, metric in zip(scoring_names, scoring):
            # NOTE: we have to check specifically for `_BaseScorer` first because this
            # is also a callable but it has a special private API that we can leverage
            if isinstance(metric, _BaseScorer):
                # scorers have the advantage to have scoped defined kwargs
                metric_fn = partial(
                    self._custom_metric,
                    metric_function=metric._score_func,
                    response_method=metric._response_method,
                )
                # forward the additional parameters specific to the scorer
                metrics_kwargs = {**metric._kwargs}
                metrics_kwargs["data_source_hash"] = data_source_hash
                metrics_params = inspect.signature(metric._score_func).parameters
                if "pos_label" in metrics_params:
                    if pos_label is not None and "pos_label" in metrics_kwargs:
                        if pos_label != metrics_kwargs["pos_label"]:
                            raise ValueError(
                                "`pos_label` is passed both in the scorer and to the "
                                "`report_metrics` method. Please provide a consistent "
                                "`pos_label` or only pass it whether in the scorer or "
                                "to the `report_metrics` method."
                            )
                    elif pos_label is not None:
                        metrics_kwargs["pos_label"] = pos_label
                if metric_name is None:
                    metric_name = metric._score_func.__name__
                metric_favorability = "↗︎" if metric._sign == 1 else "↘︎"
                favorability_indicator.append(metric_favorability)
            elif isinstance(metric, str) or callable(metric):
                if isinstance(metric, str):
                    err_msg = (
                        f"Invalid metric: {metric!r}. Please use a valid metric"
                        " from the list of supported metrics: "
                        f"{list(self._SCORE_OR_LOSS_INFO.keys())}"
                    )
                    if (
                        metric.startswith("_")
                        and metric[1:] not in self._SCORE_OR_LOSS_INFO
                    ):
                        raise ValueError(err_msg)
                    if not metric.startswith("_"):
                        if metric not in self._SCORE_OR_LOSS_INFO:
                            raise ValueError(err_msg)
                        metric = f"_{metric}"
                    metric_fn = getattr(self, metric)
                    metrics_kwargs = {"data_source_hash": data_source_hash}
                    if metric_name is None:
                        metric_name = f"{self._SCORE_OR_LOSS_INFO[metric[1:]]['name']}"
                    metric_favorability = self._SCORE_OR_LOSS_INFO[metric[1:]]["icon"]
                    favorability_indicator.append(metric_favorability)
                else:
                    metric_fn = partial(self._custom_metric, metric_function=metric)
                    if scoring_kwargs is None:
                        metrics_kwargs = {}
                    else:
                        # check if we should pass any parameters specific to the metric
                        # callable
                        metric_callable_params = inspect.signature(metric).parameters
                        metrics_kwargs = {
                            param: scoring_kwargs[param]
                            for param in metric_callable_params
                            if param in scoring_kwargs
                        }
                    metrics_kwargs["data_source_hash"] = data_source_hash
                    if metric_name is None:
                        metric_name = metric.__name__
                    metric_favorability = ""
                    favorability_indicator.append(metric_favorability)
                metrics_params = inspect.signature(metric_fn).parameters
                if scoring_kwargs is not None:
                    for param in metrics_params:
                        if param in scoring_kwargs:
                            metrics_kwargs[param] = scoring_kwargs[param]
                if "pos_label" in metrics_params:
                    metrics_kwargs["pos_label"] = pos_label
            else:
                raise ValueError(
                    f"Invalid type of metric: {type(metric)} for {metric!r}"
                )

            score = metric_fn(data_source=data_source, X=X, y=y, **metrics_kwargs)

            if self._parent._ml_task == "binary-classification":
                if isinstance(score, dict):
                    classes = list(score.keys())
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name] * len(classes), classes],
                        names=["Metric", "Label / Average"],
                    )
                    score = np.hstack([score[c] for c in classes]).reshape(-1, 1)
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
                    score = np.array(score).reshape(-1, 1)
                else:
                    index = pd.Index([metric_name], name="Metric")
                    score = np.array(score).reshape(-1, 1)
            elif self._parent._ml_task == "multiclass-classification":
                if isinstance(score, dict):
                    classes = list(score.keys())
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name] * len(classes), classes],
                        names=["Metric", "Label / Average"],
                    )
                    score = np.hstack([score[c] for c in classes]).reshape(-1, 1)
                elif (
                    "average" in metrics_kwargs
                    and metrics_kwargs["average"] is not None
                ):
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name], [metrics_kwargs["average"]]],
                        names=["Metric", "Label / Average"],
                    )
                    score = np.array(score).reshape(-1, 1)
                else:
                    index = pd.Index([metric_name], name="Metric")
                    score = np.array(score).reshape(-1, 1)
            elif self._parent._ml_task == "regression":
                if isinstance(score, np.ndarray):
                    index = pd.MultiIndex.from_arrays(
                        [[metric_name] * len(score), list(range(len(score)))],
                        names=["Metric", "Output"],
                    )
                    score = score.reshape(-1, 1)
                else:
                    index = pd.Index([metric_name], name="Metric")
                    score = np.array(score).reshape(-1, 1)
            else:  # unknown task - try our best
                index = [metric_name] if len(score) == 1 else None

            score = pd.DataFrame(
                score, index=index, columns=[self._parent.estimator_name_]
            )
            if indicator_favorability:
                score["Favorability"] = metric_favorability

            scores.append(score)

        has_multilevel = any(
            isinstance(score, pd.DataFrame) and isinstance(score.index, pd.MultiIndex)
            for score in scores
        )

        if has_multilevel:
            # Convert single-level dataframes to multi-level
            for i, score in enumerate(scores):
                if hasattr(score, "index") and not isinstance(
                    score.index, pd.MultiIndex
                ):
                    if self._parent._ml_task == "regression":
                        name_index = ["Metric", "Output"]
                    else:
                        name_index = ["Metric", "Label / Average"]

                    scores[i].index = pd.MultiIndex.from_tuples(
                        [(idx, "") for idx in score.index],
                        names=name_index,
                    )

        results = pd.concat(scores, axis=0)
        if flat_index:
            if isinstance(results.columns, pd.MultiIndex):
                results.columns = flatten_multi_index(results.columns)
            if isinstance(results.index, pd.MultiIndex):
                results.index = flatten_multi_index(results.index)
        return results

    def _compute_metric_scores(
        self,
        metric_fn,
        X,
        y_true,
        *,
        data_source="test",
        data_source_hash=None,
        response_method,
        pos_label=None,
        **metric_kwargs,
    ):
        if data_source_hash is None:
            X, y_true, data_source_hash = self._get_X_y_and_data_source_hash(
                data_source=data_source, X=X, y=y_true
            )

        cache_key = (self._parent._hash, metric_fn.__name__, data_source)
        if data_source_hash:
            cache_key += (data_source_hash,)

        metric_params = inspect.signature(metric_fn).parameters
        if "pos_label" in metric_params:
            cache_key += (pos_label,)

        # we need to enforce the order of the parameter for a specific metric
        # to make sure that we hit the cache in a consistent way
        ordered_metric_kwargs = sorted(metric_kwargs.keys())
        cache_key += tuple(
            (
                joblib.hash(metric_kwargs[key])
                if isinstance(metric_kwargs[key], np.ndarray)
                else metric_kwargs[key]
            )
            for key in ordered_metric_kwargs
        )

        if cache_key in self._parent._cache:
            score = self._parent._cache[cache_key]
        else:
            metric_params = inspect.signature(metric_fn).parameters
            kwargs = {**metric_kwargs}
            if "pos_label" in metric_params:
                kwargs.update(pos_label=pos_label)

            y_pred = _get_cached_response_values(
                cache=self._parent._cache,
                estimator_hash=self._parent._hash,
                estimator=self._parent.estimator_,
                X=X,
                response_method=response_method,
                pos_label=pos_label,
                data_source=data_source,
                data_source_hash=data_source_hash,
            )

            score = metric_fn(y_true, y_pred, **kwargs)
            if isinstance(score, np.floating):
                # convert np.float64 and np.float32 to float for consistency across
                # functions
                score = float(score)
            self._parent._cache[cache_key] = score

        if self._parent._ml_task in (
            "binary-classification",
            "multiclass-classification",
        ) and isinstance(score, np.ndarray):
            return dict(zip(self._parent._estimator.classes_, score))
        elif isinstance(score, np.ndarray):
            return score[0] if score.size == 1 else score
        return score

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def accuracy(self, *, data_source="test", X=None, y=None):
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.metrics.accuracy()
        0.95...
        """
        return self._accuracy(data_source=data_source, data_source_hash=None, X=X, y=y)

    def _accuracy(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
    ):
        """Private interface of `accuracy` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        return self._compute_metric_scores(
            metrics.accuracy_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision(
        self, *, data_source="test", X=None, y=None, average=None, pos_label=None
    ):
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

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        float or dict
            The precision score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
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

    def _precision(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
        average=None,
        pos_label=None,
    ):
        """Private interface of `precision` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        if self._parent._ml_task == "binary-classification" and pos_label is not None:
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        return self._compute_metric_scores(
            metrics.precision_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            pos_label=pos_label,
            average=average,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def recall(
        self, *, data_source="test", X=None, y=None, average=None, pos_label=None
    ):
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

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        float or dict
            The recall score.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
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

    def _recall(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
        average=None,
        pos_label=None,
    ):
        """Private interface of `recall` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        if self._parent._ml_task == "binary-classification" and pos_label is not None:
            # if `pos_label` is specified by our user, then we can safely report only
            # the statistics of the positive class
            average = "binary"

        return self._compute_metric_scores(
            metrics.recall_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            pos_label=pos_label,
            average=average,
        )

    @available_if(
        _check_supported_ml_task(supported_ml_tasks=["binary-classification"])
    )
    def brier_score(self, *, data_source="test", X=None, y=None):
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.metrics.brier_score()
        0.03...
        """
        return self._brier_score(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
        )

    def _brier_score(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
    ):
        """Private interface of `brier_score` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        # The Brier score in scikit-learn request `pos_label` to ensure that the
        # integral encoding of `y_true` corresponds to the probabilities of the
        # `pos_label`. Since we get the predictions with `get_response_method`, we
        # can pass any `pos_label`, they will lead to the same result.
        return self._compute_metric_scores(
            metrics.brier_score_loss,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict_proba",
            pos_label=self._parent._estimator.classes_[-1],
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc_auc(
        self, *, data_source="test", X=None, y=None, average=None, multi_class="ovr"
    ):
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
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

    def _roc_auc(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
        average=None,
        multi_class="ovr",
    ):
        """Private interface of `roc_auc` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        return self._compute_metric_scores(
            metrics.roc_auc_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method=["predict_proba", "decision_function"],
            average=average,
            multi_class=multi_class,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def log_loss(self, *, data_source="test", X=None, y=None):
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
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.metrics.log_loss()
        0.10...
        """
        return self._log_loss(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
        )

    def _log_loss(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
    ):
        """Private interface of `log_loss` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        return self._compute_metric_scores(
            metrics.log_loss,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict_proba",
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def r2(self, *, data_source="test", X=None, y=None, multioutput="raw_values"):
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
        float or ndarray of shape (n_outputs,)
            The R² score.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.metrics.r2()
        np.float64(0.35...)
        """
        return self._r2(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            multioutput=multioutput,
        )

    def _r2(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
        multioutput="raw_values",
    ):
        """Private interface of `r2` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        return self._compute_metric_scores(
            metrics.r2_score,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            multioutput=multioutput,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def rmse(self, *, data_source="test", X=None, y=None, multioutput="raw_values"):
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
        float or ndarray of shape (n_outputs,)
            The root mean squared error.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> report.metrics.rmse()
        np.float64(56.5...)
        """
        return self._rmse(
            data_source=data_source,
            data_source_hash=None,
            X=X,
            y=y,
            multioutput=multioutput,
        )

    def _rmse(
        self,
        *,
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
        multioutput="raw_values",
    ):
        """Private interface of `rmse` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        return self._compute_metric_scores(
            metrics.root_mean_squared_error,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method="predict",
            multioutput=multioutput,
        )

    def custom_metric(
        self,
        metric_function,
        response_method,
        *,
        data_source="test",
        X=None,
        y=None,
        **kwargs,
    ):
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
        float, dict, or ndarray of shape (n_outputs,)
            The custom metric. The output type depends on the metric function.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.metrics import mean_absolute_error
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
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
        data_source="test",
        data_source_hash=None,
        X=None,
        y=None,
        metric_function=None,
        response_method=None,
        **kwargs,
    ):
        """Private interface of `custom_metric` to be able to pass `data_source_hash`.

        `data_source_hash` is either an `int` when we already computed the hash
        and are able to pass it around or `None` and thus trigger its computation
        in the underlying process.
        """
        return self._compute_metric_scores(
            metric_function,
            X=X,
            y_true=y,
            data_source=data_source,
            data_source_hash=data_source_hash,
            response_method=response_method,
            **kwargs,
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _sort_methods_for_help(self, methods):
        """Override sort method for metrics-specific ordering.

        In short, we display the `report_metrics` first and then the `custom_metric`.
        """

        def _sort_key(method):
            name = method[0]
            if name == "custom_metric":
                priority = 1
            elif name == "report_metrics":
                priority = 2
            else:
                priority = 0
            return priority, name

        return sorted(methods, key=_sort_key)

    def _format_method_name(self, name):
        """Override format method for metrics-specific naming."""
        method_name = f"{name}(...)"
        method_name = method_name.ljust(22)
        if name in self._SCORE_OR_LOSS_INFO and self._SCORE_OR_LOSS_INFO[name][
            "icon"
        ] in ("(↗︎)", "(↘︎)"):
            if self._SCORE_OR_LOSS_INFO[name]["icon"] == "(↗︎)":
                method_name += f"[cyan]{self._SCORE_OR_LOSS_INFO[name]['icon']}[/cyan]"
                return method_name.ljust(43)
            else:  # (↘︎)
                method_name += (
                    f"[orange1]{self._SCORE_OR_LOSS_INFO[name]['icon']}[/orange1]"
                )
                return method_name.ljust(49)
        else:
            return method_name.ljust(29)

    def _get_methods_for_help(self):
        """Override to exclude the plot accessor from methods list."""
        methods = super()._get_methods_for_help()
        return [(name, method) for name, method in methods if name != "plot"]

    def _get_help_panel_title(self):
        return "[bold cyan]Available metrics methods[/bold cyan]"

    def _get_help_legend(self):
        return (
            "[cyan](↗︎)[/cyan] higher is better [orange1](↘︎)[/orange1] lower is better"
        )

    def _get_help_tree_title(self):
        return "[bold cyan]report.metrics[/bold cyan]"

    def __repr__(self):
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.EstimatorReport.metrics",
            help_method_name="report.metrics.help()",
        )

    ####################################################################################
    # Methods related to displays
    ####################################################################################

    def _get_display(
        self,
        *,
        X,
        y,
        data_source,
        response_method,
        display_class,
        display_kwargs,
    ):
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
        X, y, data_source_hash = self._get_X_y_and_data_source_hash(
            data_source=data_source, X=X, y=y
        )

        cache_key = (self._parent._hash, display_class.__name__)
        cache_key += tuple(display_kwargs.values())
        cache_key += (data_source_hash,) if data_source_hash else (data_source,)

        if cache_key in self._parent._cache:
            display = self._parent._cache[cache_key]
        else:
            y_pred = _get_cached_response_values(
                cache=self._parent._cache,
                estimator_hash=self._parent._hash,
                estimator=self._parent.estimator_,
                X=X,
                response_method=response_method,
                data_source=data_source,
                data_source_hash=data_source_hash,
                pos_label=display_kwargs.get("pos_label", None),
            )

            display = display_class._from_predictions(
                [y],
                [y_pred],
                estimator=self._parent.estimator_,
                estimator_name=self._parent.estimator_name_,
                ml_task=self._parent._ml_task,
                data_source=data_source,
                **display_kwargs,
            )
            self._parent._cache[cache_key] = display

        return display

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def roc(self, *, data_source="test", X=None, y=None, pos_label=None):
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

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        RocCurveDisplay
            The ROC curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> display = report.metrics.roc()
        >>> display.plot(roc_curve_kwargs={"color": "tab:red"})
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        return self._get_display(
            X=X,
            y=y,
            data_source=data_source,
            response_method=response_method,
            display_class=RocCurveDisplay,
            display_kwargs=display_kwargs,
        )

    @available_if(
        _check_supported_ml_task(
            supported_ml_tasks=["binary-classification", "multiclass-classification"]
        )
    )
    def precision_recall(
        self,
        *,
        data_source="test",
        X=None,
        y=None,
        pos_label=None,
    ):
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

        pos_label : int, float, bool or str, default=None
            The positive class.

        Returns
        -------
        PrecisionRecallCurveDisplay
            The precision-recall curve display.

        Examples
        --------
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_breast_cancer(return_X_y=True), random_state=0
        ... )
        >>> classifier = LogisticRegression(max_iter=10_000)
        >>> report = EstimatorReport(
        ...     classifier,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> display = report.metrics.precision_recall()
        >>> display.plot(pr_curve_kwargs={"color": "tab:red"})
        """
        response_method = ("predict_proba", "decision_function")
        display_kwargs = {"pos_label": pos_label}
        return self._get_display(
            X=X,
            y=y,
            data_source=data_source,
            response_method=response_method,
            display_class=PrecisionRecallCurveDisplay,
            display_kwargs=display_kwargs,
        )

    @available_if(_check_supported_ml_task(supported_ml_tasks=["regression"]))
    def prediction_error(
        self,
        *,
        data_source="test",
        X=None,
        y=None,
        subsample=1_000,
        random_state=None,
    ):
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

        random_state : int, default=None
            The random state to use for the subsampling.

        Returns
        -------
        PredictionErrorDisplay
            The prediction error display.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.model_selection import train_test_split
        >>> from skore import EstimatorReport
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     *load_diabetes(return_X_y=True), random_state=0
        ... )
        >>> regressor = Ridge()
        >>> report = EstimatorReport(
        ...     regressor,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     X_test=X_test,
        ...     y_test=y_test,
        ... )
        >>> display = report.metrics.prediction_error()
        >>> display.plot(line_kwargs={"color": "tab:red"})
        """
        display_kwargs = {"subsample": subsample, "random_state": random_state}
        return self._get_display(
            X=X,
            y=y,
            data_source=data_source,
            response_method="predict",
            display_class=PredictionErrorDisplay,
            display_kwargs=display_kwargs,
        )
