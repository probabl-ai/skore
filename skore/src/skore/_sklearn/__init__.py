"""Enhance `sklearn` functions."""

from skore._sklearn._comparison import ComparisonReport
from skore._sklearn._cross_validation import CrossValidationReport
from skore._sklearn._estimator import EstimatorReport
from skore._sklearn._plot import (
    MetricsSummaryDisplay,
    PrecisionRecallCurveDisplay,
    PredictionErrorDisplay,
    RocCurveDisplay,
    TableReportDisplay,
)
from skore._sklearn.find_estimators import find_estimators
from skore._sklearn.train_test_split.train_test_split import train_test_split

__all__ = [
    "ComparisonReport",
    "CrossValidationReport",
    "EstimatorReport",
    "PrecisionRecallCurveDisplay",
    "PredictionErrorDisplay",
    "TableReportDisplay",
    "RocCurveDisplay",
    "MetricsSummaryDisplay",
    "train_test_split",
    "find_estimators",
]
