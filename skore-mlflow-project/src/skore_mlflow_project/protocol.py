"""Protocols definition used to remove adherence to ``skore`` for type checking."""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

DatasetLike: TypeAlias = NDArray[np.generic] | pd.DataFrame | pd.Series


@runtime_checkable
class EstimatorReport(Protocol):
    """Protocol equivalent to ``skore.EstimatorReport``."""

    ml_task: str
    estimator: BaseEstimator
    estimator_: BaseEstimator
    estimator_name_: str
    X_train: DatasetLike | None
    X_test: DatasetLike
    y_train: DatasetLike | None
    y_test: DatasetLike
    metrics: Any
    data: Any


@runtime_checkable
class CrossValidationReport(Protocol):
    """Protocol equivalent to ``skore.CrossValidationReport``."""

    ml_task: str
    estimator: BaseEstimator
    estimator_: BaseEstimator
    estimator_name_: str
    estimator_reports_: list[EstimatorReport]
    X: DatasetLike
    y: DatasetLike
    metrics: Any
    data: Any
    splitter: Any

    def create_estimator_report(self) -> EstimatorReport:
        """Create a representative estimator report."""
