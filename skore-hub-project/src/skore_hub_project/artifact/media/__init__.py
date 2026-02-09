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
    PrecisionRecallTest,
    PrecisionRecallTrain,
    PredictionErrorTest,
    PredictionErrorTrain,
    RocTest,
    RocTrain,
)

__all__ = [
    "Coefficients",
    "EstimatorHtmlRepr",
    "ImpurityDecrease",
    "PermutationImportanceTest",
    "PermutationImportanceTrain",
    "PrecisionRecallTest",
    "PrecisionRecallTrain",
    "PredictionErrorTest",
    "PredictionErrorTrain",
    "RocTest",
    "RocTrain",
    "TableReportTest",
    "TableReportTrain",
]
