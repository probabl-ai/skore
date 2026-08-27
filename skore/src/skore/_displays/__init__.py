from typing import TYPE_CHECKING

from skore._externals import lazy_loader as lazy

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

__getattr__, __dir__, _ = lazy.attach(
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
