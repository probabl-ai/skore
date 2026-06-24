"""Enhance `sklearn` functions."""

from skore._sklearn._comparison import ComparisonReport
from skore._sklearn._cross_validation import CrossValidationReport
from skore._sklearn._estimator import EstimatorReport
from skore._sklearn._plot import (
    ConfusionMatrixDisplay,
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    TableReportDisplay,
)
from skore._sklearn.compare import compare
from skore._sklearn.evaluate import evaluate
from skore._sklearn.train_test_split import TrainTestSplit

__all__ = [
    "ComparisonReport",
    "compare",
    "ConfusionMatrixDisplay",
    "CrossValidationReport",
    "EstimatorReport",
    "evaluate",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "TableReportDisplay",
    "RocCurveDisplay",
    "MetricsSummaryDisplay",
    "TrainTestSplit",
]
