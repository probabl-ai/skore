"""Enhance `sklearn` functions."""

from skore.sklearn._cross_validation import CrossValidationReport
from skore.sklearn._estimator import EstimatorReport
from skore.sklearn.comparator import Comparator
from skore.sklearn.train_test_split.train_test_split import train_test_split

__all__ = [
    "train_test_split",
    "CrossValidationReport",
    "EstimatorReport",
    "Comparator",
]
