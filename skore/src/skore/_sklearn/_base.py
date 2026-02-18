from io import StringIO
from typing import Any, Generic, Literal, TypeVar, cast

import joblib
from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.panel import Panel
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore._externals._sklearn_compat import is_clusterer
from skore._sklearn.types import PositiveLabel
from skore._utils._cache import Cache
from skore._utils._measure_time import MeasureTime
from skore._utils.repr.base import AccessorHelpMixin, ReportHelpMixin


class _BaseReport(ReportHelpMixin):
    """Base class for all reports.

    This class centralizes shared report logic (e.g. configuration, accessors) and
    inherits from ``ReportHelpMixin`` to provide a consistent ``help()`` and rich/HTML
    representation across all report types.
    """

    _ACCESSOR_CONFIG: dict[str, dict[str, str]]
    _report_type: Literal[
        "estimator",
        "cross-validation",
        "comparison-estimator",
        "comparison-cross-validation",
    ]


ParentT = TypeVar("ParentT", bound="_BaseReport")


class _BaseAccessor(AccessorHelpMixin, Generic[ParentT]):
    """Base class for all accessors.

    Accessors expose additional views on a report (e.g. data, metrics) and inherit from
    ``AccessorHelpMixin`` to provide a dedicated ``help()`` and rich/HTML help tree.
    """

    def __init__(self, parent: ParentT) -> None:
        self._parent = parent

    def _rich_repr(self, class_name: str) -> str:
        """Return a string representation using rich for accessors."""
        string_buffer = StringIO()
        console = Console(file=string_buffer, force_terminal=False)
        console.print(
            Panel(
                "Get guidance using the help() method",
                title=f"[cyan]{class_name}[/cyan]",
                border_style="orange1",
                expand=False,
            )
        )
        return string_buffer.getvalue()

    def _get_X_y_and_data_source_hash(
        self,
        *,
        data_source: Literal["test", "train", "X_y"],
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike | None, int | None]:
        """Get the requested dataset and mention if we should hash before caching.

        Parameters
        ----------
        data_source : {"test", "train", "X_y"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.
            - "X_y" : use the provided `X` and `y` to compute the metric.

        X : array-like of shape (n_samples, n_features) or None, default=None
            The input data.

        y : array-like of shape (n_samples,) or None, default=None
            The target data.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The requested dataset.

        y : array-like of shape (n_samples,)
            The requested dataset.

        data_source_hash : int or None
            The hash of the data source. None when we are able to track the data, and
            thus relying on X_train, y_train, X_test, y_test.
        """
        is_cluster = is_clusterer(self._parent.estimator_)
        if data_source == "test":
            if not (X is None or y is None):
                raise ValueError("X and y must be None when data_source is test.")
            if self._parent._X_test is None or (
                not is_cluster and self._parent._y_test is None
            ):
                missing_data = "X_test" if is_cluster else "X_test and y_test"
                raise ValueError(
                    f"No {data_source} data (i.e. {missing_data}) were provided "
                    f"when creating the report. Please provide the {data_source} "
                    "data either when creating the report or by setting data_source "
                    "to 'X_y' and providing X and y."
                )
            return self._parent._X_test, self._parent._y_test, None
        elif data_source == "train":
            if not (X is None or y is None):
                raise ValueError("X and y must be None when data_source is train.")
            if self._parent._X_train is None or (
                not is_cluster and self._parent._y_train is None
            ):
                missing_data = "X_train" if is_cluster else "X_train and y_train"
                raise ValueError(
                    f"No {data_source} data (i.e. {missing_data}) were provided "
                    f"when creating the report. Please provide the {data_source} "
                    "data either when creating the report or by setting data_source "
                    "to 'X_y' and providing X and y."
                )
            return self._parent._X_train, self._parent._y_train, None
        elif data_source == "X_y":
            if X is None or (not is_cluster and y is None):
                missing_data = "X" if is_cluster else "X and y"
                raise ValueError(
                    f"{missing_data} must be provided when data_source is X_y."
                )
            return X, y, joblib.hash((X, y))
        else:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: "
                "test, train, X_y."
            )


def _get_cached_response_values(
    *,
    cache: Cache,
    estimator_hash: int,
    estimator: BaseEstimator,
    X: ArrayLike | None,
    response_method: str | list[str] | tuple[str, ...],
    pos_label: PositiveLabel | None = None,
    data_source: Literal["test", "train", "X_y"] = "test",
    data_source_hash: int | None = None,
) -> list[tuple[tuple[Any, ...], Any, bool]]:
    """Compute or load from local cache the response values.

    Be aware that the predictions will be loaded from the cache if present, but they
    will not be added to it. The reason is that we want to be able to run this function
    in parallel settings in a thread-safe manner. The update should be done outside of
    this function.

    Parameters
    ----------
    cache : Cache
        The cache backend to use.

    estimator_hash : int
        A hash associated with the estimator such that we can retrieve the data from
        the cache.

    estimator : estimator object
        The estimator used to generate the predictions.

    X : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        The input data on which to compute the responses when needed.

    response_method : str, list of str or tuple of str
        The response method.

    pos_label : int, float, bool or str, default=None
        The positive label.

    data_source : {"test", "train", "X_y"}, default="test"
        The data source to use.

        - "test" : use the test set provided when creating the report.
        - "train" : use the train set provided when creating the report.
        - "X_y" : use the provided `X` and `y` to compute the metric.

    data_source_hash : int or None
        The hash of the data source when `data_source` is "X_y".

    Returns
    -------
    list of tuples
        A list of tuples, each containing:

        - cache_key : tuple
            The cache key.

        - cache_value : Any
            The cache value. It corresponds to the predictions but also to the predict
            time when it has not been cached yet.

        - is_cached : bool
            Whether the cache value was loaded from the cache.
    """
    prediction_method = _check_response_method(estimator, response_method).__name__

    if data_source == "X_y" and data_source_hash is None:
        # Only trigger hash computation if it was not previously done.
        # If data_source_hash is not None, we internally computed ourself the hash
        # and it is trustful
        data_source_hash = joblib.hash(X)

    if prediction_method not in ("predict_proba", "decision_function"):
        # pos_label is only important in classification and with probabilities
        # and decision functions
        pos_label = None

    cache_key: tuple[Any, ...] = (
        estimator_hash,
        pos_label,
        prediction_method,
        data_source,
        data_source_hash,
    )

    if cache_key in cache:
        cached_predictions = cast(NDArray, cache[cache_key])
        return [(cache_key, cached_predictions, True)]

    with MeasureTime() as predict_time:
        predictions, _ = _get_response_values(
            estimator,
            X=X,
            response_method=prediction_method,
            pos_label=pos_label,
            return_response_method_used=False,
        )

    predict_time_cache_key: tuple[Any, ...] = (
        estimator_hash,
        data_source,
        data_source_hash,
        "predict_time",
    )

    return [
        (cache_key, predictions, False),
        (predict_time_cache_key, predict_time(), False),
    ]


class _BaseMetricsAccessor:
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
        "report_metrics": {"name": "Report metrics", "icon": ""},
    }

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _get_favorability_text(self, name: str) -> str | None:
        """Get favorability text for a method, or None if not applicable."""
        if name not in self._score_or_loss_info:
            return None
        icon = self._score_or_loss_info[name]["icon"]
        if icon == "(↗︎)":
            return "Higher value is better."
        elif icon == "(↘︎)":
            return "Lower value is better."
        return None
