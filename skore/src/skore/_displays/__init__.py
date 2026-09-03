from typing import TYPE_CHECKING

from skore._externals import lazy_loader

if TYPE_CHECKING:
    from skore._displays.data import TableReportDisplay
    from skore._displays.metrics import (
        ConfusionMatrixDisplay,
        MetricsSummaryDisplay,
        PrecisionRecallCurveDisplay,
        PredictionErrorDisplay,
        RocCurveDisplay,
    )

__all__ = [
    "ConfusionMatrixDisplay",
    "RocCurveDisplay",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "TableReportDisplay",
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
        "data": ["TableReportDisplay"],
        "metrics": [
            "ConfusionMatrixDisplay",
            "MetricsSummaryDisplay",
            "PrecisionRecallCurveDisplay",
            "PredictionErrorDisplay",
            "RocCurveDisplay",
        ],
    },
)
