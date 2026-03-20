from abc import abstractmethod
from io import StringIO
from typing import Any, Generic, Literal, TypeVar, cast

from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich.panel import Panel
from sklearn.base import BaseEstimator
from sklearn.utils._response import _check_response_method, _get_response_values

from skore._config import configuration
from skore._sklearn._diagnostics.base import (
    DiagnosticResult,
    DiagnosticResults,
    format_diagnostic_message,
    normalize_ignore_codes,
)
from skore._sklearn.types import PositiveLabel
from skore._utils._cache import Cache
from skore._utils._environment import (
    is_environment_notebook_like,
    is_environment_sphinx_build,
)
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
    _diagnostics_cache: list[DiagnosticResult]

    @abstractmethod
    def _collect_diagnostics(self) -> list[DiagnosticResult]:
        """Collect diagnostics."""

    def _get_diagnostics(self) -> list[DiagnosticResult]:
        if not hasattr(self, "_diagnostics_cache"):
            self._diagnostics_cache = self._collect_diagnostics()
        return self._diagnostics_cache

    def _display_diagnose_results(self, results: list[str]) -> None:
        if is_environment_sphinx_build():
            return
        if is_environment_notebook_like():
            from IPython.display import display

            display(results)
            return
        from skore import console

        console.print(results)

    def diagnose(
        self,
        *,
        ignore: list[str] | tuple[str, ...] | None = None,
    ) -> DiagnosticResults:
        ignored = normalize_ignore_codes(ignore) | normalize_ignore_codes(
            tuple(configuration.ignore_diagnostics)
        )
        diagnostics = [
            diagnostic
            for diagnostic in self._get_diagnostics()
            if diagnostic.code not in ignored
        ]
        self._latest_diagnostics_ = diagnostics
        messages = [format_diagnostic_message(diagnostic) for diagnostic in diagnostics]
        results = DiagnosticResults(messages, diagnostics)
        self._latest_diagnose_result_ = results
        return results

    def _diagnostics_panel_html(self) -> str:
        diagnostics = getattr(self, "_latest_diagnostics_", None)
        if diagnostics is None:
            details = "No diagnostics have run yet."
            summary = "0 issue(s) across 0 diagnostic(s)."
        else:
            issue_count = sum(diagnostic.is_issue for diagnostic in diagnostics)
            evaluated_count = sum(diagnostic.evaluated for diagnostic in diagnostics)
            details = f"{issue_count} issue(s) across {len(diagnostics)} diagnostic(s)."
            summary = f"{evaluated_count} diagnostic(s) evaluated in the latest run."
        return (
            '<div style="margin:10px 0;padding:10px;'
            "border:1px solid #f97316;border-radius:4px;"
            'font-family:monospace;font-size:13px;line-height:1.5;">'
            '<div style="font-weight:700;margin-bottom:6px;">Diagnostics</div>'
            f"<div>{details}</div>"
            f"<div>{summary}</div>"
            "<div>Run <code>.diagnose()</code> for details and "
            "<code>.diagnose(ignore=...)</code> to ignore specific diagnostics.</div>"
            "</div>"
        )

    def _repr_html_(self) -> str:
        return f"{self._create_help_html()}{self._diagnostics_panel_html()}"

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        return {"text/plain": repr(self), "text/html": self._repr_html_()}


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
    estimator_hash: int,
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

    cache_key: tuple[Any, ...] = (
        estimator_hash,
        pos_label,
        prediction_method,
        data_source,
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
