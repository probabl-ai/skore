"""Protocols definition used to remove adherence to ``skore`` for type checking."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EstimatorReport(Protocol):
    """Protocol equivalent to ``skore.EstimatorReport``."""

    _cache: Any
    metrics: Any
    data: Any
    ml_task: Any
    estimator: Any
    estimator_name_: Any
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    fit: Any


@runtime_checkable
class CrossValidationReport(Protocol):
    """Protocol equivalent to ``skore.CrossValidationReport``."""

    _cache: Any
    metrics: Any
    data: Any
    estimator_reports_: Any
    ml_task: Any
    estimator: Any
    estimator_name_: Any
    X: Any
    y: Any
    splitter: Any
    split_indices: Any
