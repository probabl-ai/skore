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
    ConfusionMatrixDataFrameTestAll,
    ConfusionMatrixDataFrameTestNone,
    ConfusionMatrixDataFrameTrainAll,
    ConfusionMatrixDataFrameTrainNone,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PredictionErrorDataFrameTest,
    PredictionErrorDataFrameTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
)

__all__ = [
    "Coefficients",
    "ConfusionMatrixDataFrameTestAll",
    "ConfusionMatrixDataFrameTestNone",
    "ConfusionMatrixDataFrameTrainAll",
    "ConfusionMatrixDataFrameTrainNone",
    "EstimatorHtmlRepr",
    "ImpurityDecrease",
    "PermutationImportanceTest",
    "PermutationImportanceTrain",
    "PrecisionRecallDataFrameTest",
    "PrecisionRecallDataFrameTrain",
    "PredictionErrorDataFrameTest",
    "PredictionErrorDataFrameTrain",
    "RocDataFrameTest",
    "RocDataFrameTrain",
    "TableReportTest",
    "TableReportTrain",
]
