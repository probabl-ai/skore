"""Class definitions of the payloads used to send a media to ``hub``."""

from .checks import ChecksSummary
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
    RocDataFrameCrossValidationTest,
    RocDataFrameCrossValidationTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
)

__all__ = [
    "ChecksSummary",
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
    "RocDataFrameCrossValidationTest",
    "RocDataFrameCrossValidationTrain",
    "RocDataFrameTest",
    "RocDataFrameTrain",
    "TableReportTest",
    "TableReportTrain",
]
