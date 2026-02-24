"""Protocols definition used to remove adherence to ``skore`` for type checking."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EstimatorReport(Protocol):
    """Protocol equivalent to ``skore.EstimatorReport``."""

    _report_type: str
    ml_task: str
    estimator_: Any
    estimator_name_: str
    y_test: Any


@runtime_checkable
class CrossValidationReport(Protocol):
    """Protocol equivalent to ``skore.CrossValidationReport``."""

    _report_type: str
    ml_task: str
    estimator_: Any
    estimator_name_: str
    estimator_reports_: Any
    y: Any
