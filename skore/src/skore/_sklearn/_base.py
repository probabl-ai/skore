from abc import abstractmethod
from functools import cached_property
from io import StringIO
from typing import Any, Generic, Literal, TypeVar, cast
from uuid import uuid4

from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.panel import Panel
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore._config import configuration
from skore._sklearn._diagnostics.base import DiagnosticsDisplay
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
    _diagnostics_cache: tuple[dict[str, dict], set[str]]

    @abstractmethod
    def _compute_diagnostics(self) -> tuple[dict[str, dict], set[str]]:
        """Return detected issues and the set of diagnostic codes that were checked."""

    def _get_diagnostics(self) -> tuple[dict[str, dict], set[str]]:
        """Get the diagnostics from the cache or compute them."""
        if not hasattr(self, "_diagnostics_cache"):
            self._diagnostics_cache = self._compute_diagnostics()
        return self._diagnostics_cache

    def diagnose(
        self,
        *,
        ignore: list[str] | tuple[str, ...] | None = None,
    ) -> DiagnosticsDisplay:
        """Run diagnostics and return a summary of detected issues.

        Diagnostics check for common modeling problems such as overfitting and
        underfitting. Codes can be muted per-call via `ignore` or globally via
        :func:`~skore.configuration(ignore_diagnostics=...)` .

        Parameters
        ----------
        ignore : list of str or tuple of str or None, default=None
            diagnostic codes to exclude from the results, e.g.
            `["SKD001"]`

        Returns
        -------
        DiagnosticsDisplay
            A display object with rich and HTML representations, with the full
            diagnostic result objects accessible via the
            :meth:`~DiagnosticsDisplay.frame` method.

        Examples
        --------
        >>> from skore import evaluate
        >>> from sklearn.dummy import DummyClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(random_state=42)
        >>> report = evaluate(DummyClassifier(), X, y, splitter=0.2)
        >>> report.diagnose()
        Diagnostics: 1 issue(s) detected, 2 check(s) ran, 0 ignored.
        - [SKD002] Potential underfitting. Train/test scores are on par and not
        significantly better than the dummy baseline for 8/8 comparable metrics. Read
        our documentation for more details:
        https://docs.skore.probabl.ai/dev/user_guide/diagnostics.html#skd002-underfitting.
        Mute with `ignore=['SKD002']`.
        >>> report.diagnose(ignore=["SKD002"])
        Diagnostics: 0 issue(s) detected, 1 check(s) ran, 1 ignored.
        - No issues were detected in your report!
        """
        ignored: set[str] = set()
        if ignore:
            ignored.update(code.strip().upper() for code in ignore if code.strip())
        if configuration.ignore_diagnostics:
            ignored.update(
                code.strip().upper()
                for code in configuration.ignore_diagnostics
                if code.strip()
            )
        diagnostics, checked_codes = self._get_diagnostics()
        filtered = {code: d for code, d in diagnostics.items() if code not in ignored}
        checks_ran = len(checked_codes - ignored)
        return DiagnosticsDisplay(filtered, checks_ran, n_ignored=len(ignored))

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

    def _get_X_y(
        self,
        *,
        data_source: Literal["test", "train"],
    ) -> tuple[ArrayLike, ArrayLike]:
        """Get the requested dataset.

        Parameters
        ----------
        data_source : {"test", "train"}, default="test"
            The data source to use.

            - "test" : use the test set provided when creating the report.
            - "train" : use the train set provided when creating the report.

        Returns
        -------
        X : array-like of shape (n_samples, n_features)
            The requested dataset.

        y : array-like of shape (n_samples,)
            The requested dataset.
        """
        if data_source == "test":
            if self._parent._X_test is None or self._parent._y_test is None:
                missing_data = "X_test and y_test"
                raise ValueError(
                    f"No {data_source} data (i.e. {missing_data}) were provided "
                    f"when creating the report. Please provide the {data_source} "
                    "data when creating the report."
                )
            return self._parent._X_test, self._parent._y_test
        elif data_source == "train":
            if self._parent._X_train is None or self._parent._y_train is None:
                missing_data = "X_train and y_train"
                raise ValueError(
                    f"No {data_source} data (i.e. {missing_data}) were provided "
                    f"when creating the report. Please provide the {data_source} "
                    "data when creating the report."
                )
            return self._parent._X_train, self._parent._y_train
        else:
            raise ValueError(
                f"Invalid data source: {data_source}. Possible values are: test, train."
            )


def _get_cached_response_values(
    *,
    cache: Cache,
    estimator: BaseEstimator,
    X: ArrayLike | None,
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
