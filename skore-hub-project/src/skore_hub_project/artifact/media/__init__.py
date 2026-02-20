"""Class definitions of the payloads used to send a media to ``hub``."""

from .data import TableReportTest, TableReportTrain
from .inspection import (
    Coefficients,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
)
from .model import EstimatorHtmlRepr
from .performance import (
    PrecisionRecallSVGTest,
    PrecisionRecallSVGTrain,
    PredictionErrorSVGTest,
    PredictionErrorSVGTrain,
    RocSVGTest,
    RocSVGTrain,
)

__all__ = [
    "Coefficients",
    "EstimatorHtmlRepr",
    "ImpurityDecrease",
    "PermutationImportanceTest",
    "PermutationImportanceTrain",
    "PrecisionRecallSVGTest",
    "PrecisionRecallSVGTrain",
    "PredictionErrorSVGTest",
    "PredictionErrorSVGTrain",
    "RocSVGTest",
    "RocSVGTrain",
    "TableReportTest",
    "TableReportTrain",
]
