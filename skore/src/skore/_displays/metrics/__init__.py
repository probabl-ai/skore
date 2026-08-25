from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "ConfusionMatrixDisplay",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "RocCurveDisplay",
    "MetricsSummaryDisplay",
]

__lazy__ = {
    "ConfusionMatrixDisplay": "skore._displays.metrics.confusion_matrix",
    "PrecisionRecallCurveDisplay": "skore._displays.metrics.precision_recall_curve",
    "PredictionErrorDisplay": "skore._displays.metrics.prediction_error",
    "RocCurveDisplay": "skore._displays.metrics.roc_curve",
    "MetricsSummaryDisplay": "skore._displays.metrics.metrics_summary_display",
}

if TYPE_CHECKING:
    from skore._displays.metrics.confusion_matrix import ConfusionMatrixDisplay
    from skore._displays.metrics.metrics_summary_display import MetricsSummaryDisplay
    from skore._displays.metrics.precision_recall_curve import (
        PrecisionRecallCurveDisplay,
    )
    from skore._displays.metrics.prediction_error import PredictionErrorDisplay
    from skore._displays.metrics.roc_curve import RocCurveDisplay


def __getattr__(name: str) -> Any:
    if module_name := __lazy__.get(name):
        value = getattr(import_module(module_name), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
