"""Enhance `sklearn` functions."""

from skore.sklearn.cross_validate import CrossValidationReporter, cross_validate

__all__ = [
    "cross_validate",
    "CrossValidationReporter",
]
