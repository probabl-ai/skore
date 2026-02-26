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
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PrecisionRecallSVGTest,
    PrecisionRecallSVGTrain,
    PredictionErrorDataFrameTest,
    PredictionErrorDataFrameTrain,
    PredictionErrorSVGTest,
    PredictionErrorSVGTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
    RocSVGTest,
    RocSVGTrain,
)

__all__ = [
    "Coefficients",
    "EstimatorHtmlRepr",
    "ImpurityDecrease",
    "PermutationImportanceTest",
    "PermutationImportanceTrain",
    "PrecisionRecallDataFrameTest",
    "PrecisionRecallDataFrameTrain",
    "PrecisionRecallSVGTest",
    "PrecisionRecallSVGTrain",
    "PredictionErrorDataFrameTest",
    "PredictionErrorDataFrameTrain",
    "PredictionErrorSVGTest",
    "PredictionErrorSVGTrain",
    "RocDataFrameTest",
    "RocDataFrameTrain",
    "RocSVGTest",
    "RocSVGTrain",
    "TableReportTest",
    "TableReportTrain",
]
