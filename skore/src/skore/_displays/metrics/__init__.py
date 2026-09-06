from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._displays.metrics.confusion_matrix import ConfusionMatrixDisplay
    from skore._displays.metrics.metrics_summary_display import MetricsSummaryDisplay
    from skore._displays.metrics.precision_recall_curve import (
        PrecisionRecallCurveDisplay,
    )
    from skore._displays.metrics.prediction_error import PredictionErrorDisplay
    from skore._displays.metrics.roc_curve import RocCurveDisplay

__all__ = [
    "ConfusionMatrixDisplay",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "RocCurveDisplay",
    "MetricsSummaryDisplay",
]

# Declare objects as importable from here, but lazy-load them to avoid slowdowns.
#
# For an object to be lazy-loaded, declare it:
# - in the ``if TYPE_CHECKING`` block, so type checkers can use it,
# - in ``__all__``, so the F401 linter does not fail,
# - in the ``lazy_loader.attach`` call below.
__getattr__, __dir__, _ = lazy_loader.attach(
    __name__,
    submod_attrs={
        "confusion_matrix": ["ConfusionMatrixDisplay"],
        "metrics_summary_display": ["MetricsSummaryDisplay"],
        "precision_recall_curve": ["PrecisionRecallCurveDisplay"],
        "prediction_error": ["PredictionErrorDisplay"],
        "roc_curve": ["RocCurveDisplay"],
    },
)
