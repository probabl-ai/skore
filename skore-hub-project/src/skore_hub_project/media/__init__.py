"""Class definitions of the payloads used to send a media to ``hub``."""

from __future__ import annotations

from .data import TableReportTest, TableReportTrain
from .feature_importance import (
    Coefficients,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
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
    "MeanDecreaseImpurity",
    "PermutationTest",
    "PermutationTrain",
    "PrecisionRecallTest",
    "PrecisionRecallTrain",
    "PredictionErrorTest",
    "PredictionErrorTrain",
    "RocTest",
    "RocTrain",
    "TableReportTest",
    "TableReportTrain",
]
