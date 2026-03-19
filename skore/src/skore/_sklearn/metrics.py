from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Metric:
    """Metadata for a metric in the registry.

    Parameters
    ----------
    name : str
        Technical name used for lookup (e.g. ``"accuracy"``).

    verbose_name : str
        Display name shown in reports (e.g. ``"Accuracy"``).

    greater_is_better : bool or None, default=None
        Whether a higher value is better (``True``), lower is better
        (``False``), or there is no preference or information (``None``).

    score_func : callable or None, default=None
        The scoring function. ``None`` for built-in metrics that are dispatched
        by name; a callable for custom metrics.

    response_method : {"predict", "predict_proba", "decision_function"} or None, \
        default="predict"
        The method to call to get the predicted values that will passed to
        ``score_func``.

    kwargs : dict, default={}
        Keyword arguments to pass to ``score_func``.

    is_builtin : bool, default=False
        Whether this metric is a skore built-in.  Built-ins cannot be
        overridden via :meth:`register`.

    source_code : str or None, default=None
        Source code of the score function, captured at registration time.
    """

    name: str
    verbose_name: str
    greater_is_better: bool | None = None
    score_func: Callable | None = None
    response_method: (
        Literal["predict", "predict_proba", "decision_function"] | list[str] | None
    ) = "predict"
    kwargs: dict[str, Any] = field(default_factory=dict)
    is_builtin: bool = False
    source_code: str | None = None

    @property
    def icon(self) -> str:
        """Favorability icon derived from ``greater_is_better``."""
        match self.greater_is_better:
            case True:
                return "(↗︎)"
            case False:
                return "(↘︎)"
            case _:
                return ""

    def is_callable(self) -> bool:
        """Return whether the score function is callable."""
        return self.score_func is not None and callable(self.score_func)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("score_func") is not None:
            import pickle as _pickle

            try:
                _pickle.dumps(state["score_func"])
            except Exception:
                state["score_func"] = None
                state["_score_func_lost"] = True
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        if state.pop("_score_func_lost", False):
            import warnings

            warnings.warn(
                f"The score function for metric {state.get('name', '?')!r} "
                "could not be restored after pickling (e.g., it was a lambda "
                "or closure).",
                UserWarning,
                stacklevel=2,
            )
        self.__dict__.update(state)


FitTime = Metric(
    name="fit_time",
    verbose_name="Fit time (s)",
    greater_is_better=False,
    response_method=None,
    is_builtin=True,
)
PredictTime = Metric(
    name="predict_time",
    verbose_name="Predict time (s)",
    greater_is_better=False,
    response_method=None,
    is_builtin=True,
)
Accuracy = Metric(
    name="accuracy",
    verbose_name="Accuracy",
    greater_is_better=True,
    is_builtin=True,
)
Precision = Metric(
    name="precision",
    verbose_name="Precision",
    greater_is_better=True,
    is_builtin=True,
)
Recall = Metric(
    name="recall",
    verbose_name="Recall",
    greater_is_better=True,
    is_builtin=True,
)
Brier = Metric(
    name="brier_score",
    verbose_name="Brier score",
    greater_is_better=False,
    is_builtin=True,
)
RocAuc = Metric(
    name="roc_auc",
    verbose_name="ROC AUC",
    greater_is_better=True,
    is_builtin=True,
)
LogLoss = Metric(
    name="log_loss",
    verbose_name="Log loss",
    greater_is_better=False,
    is_builtin=True,
)
R2 = Metric(
    name="r2",
    verbose_name="R²",
    greater_is_better=True,
    is_builtin=True,
)
Rmse = Metric(
    name="rmse",
    verbose_name="RMSE",
    greater_is_better=False,
    is_builtin=True,
)
