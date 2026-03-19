from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from sklearn.base import BaseEstimator

from skore._sklearn.types import MLTask


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
    """

    name: str
    verbose_name: str
    greater_is_better: bool | None = None
    score_func: Callable | None = None
    response_method: Literal["predict", "predict_proba", "decision_function"] | None = (
        "predict"
    )
    kwargs: dict[str, Any] = field(default_factory=dict)
    is_builtin: bool = False

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

BUILTIN_METRICS = [
    FitTime,
    PredictTime,
    Accuracy,
    Precision,
    Recall,
    Brier,
    RocAuc,
    LogLoss,
    R2,
    Rmse,
]


def _get_default_metrics(ml_task: MLTask, estimator: BaseEstimator) -> list[str]:
    if ml_task == "binary-classification":
        metrics = [Accuracy, Precision, Recall, RocAuc]
        if hasattr(estimator, "predict_proba"):
            metrics += [Brier, LogLoss]
    elif ml_task == "multiclass-classification":
        metrics = [Accuracy, Precision, Recall]
        if hasattr(estimator, "predict_proba"):
            metrics += [RocAuc, LogLoss]
    else:
        metrics = [R2, Rmse]
    metrics += [FitTime, PredictTime]
    return [m.name for m in metrics]
