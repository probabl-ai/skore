from functools import cached_property
from io import StringIO
from typing import Any, Generic, Literal, TypeVar, cast
from uuid import uuid4

from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.panel import Panel
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore._sklearn.types import PositiveLabel
from skore._utils._cache import Cache
from skore._utils._cache_key import make_cache_key
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

    @cached_property
    def id(self) -> int:
        return uuid4().int

    @property
    def _hash(self) -> int:
        # FIXME: only for backward compatibility
        return self.id


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

    def _get_data_and_y_true(
        self,
        *,
        data_source: Literal["test", "train"],
    ) -> tuple[dict, ArrayLike]:
        """Get the requested dataset.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        data : dict of input data
            The requested dataset.

        y : array-like of shape (n_samples,)
            The target labels.
        """
        if data_source not in ["train", "test"]:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: test, train."
            )
        if getattr(self._parent, f"{data_source}_data") is None:
            raise ValueError(
                f"No {data_source} data were provided when creating the report."
            )
        if data_source == "test":
            return self._parent.test_data, self._parent.y_test
        assert data_source == "train"
        return self._parent.train_data, self._parent.y_train


def _get_cached_response_values(
    *,
    cache: Cache,
    estimator: BaseEstimator,
    X: ArrayLike | dict | None,
    response_method: str | list[str] | tuple[str, ...],
    pos_label: PositiveLabel | None = None,
    data_source: Literal["test", "train"] = "test",
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

    estimator : estimator object
        The estimator used to generate the predictions.

    X : {array-like, sparse matrix} of shape (n_samples, n_features) or None
        The input data on which to compute the responses when needed.

    response_method : str, list of str or tuple of str
        The response method.

    pos_label : int, float, bool or str, default=None
        The positive label.

    data_source : {"test", "train"}, default="test"
        The data source to use.

        - "test" : use the test set provided when creating the report.
        - "train" : use the train set provided when creating the report.

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

    if prediction_method not in ("predict_proba", "decision_function"):
        # pos_label is only important in classification and with probabilities
        # and decision functions
        pos_label = None

    kwargs = {"pos_label": pos_label}
    cache_key = make_cache_key(data_source, prediction_method, kwargs)

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

    predict_time_cache_key = make_cache_key(data_source, "predict_time")

    return [
        (cache_key, predictions, False),
        (predict_time_cache_key, predict_time(), False),
    ]
