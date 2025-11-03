"""Protocols definition used to remove adherence to ``skore`` for type checking."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pandas import DataFrame

DataSource = Literal["train", "test"] | None
Aggregate = Literal["mean", "std"]


@runtime_checkable
class Display(Protocol):
    """Protocol equivalent to ``skore.Display``."""

    frame: Callable[..., DataFrame]
    plot: Callable[..., None]
    figure_: Any


@runtime_checkable
class EstimatorReport(Protocol):
    """Protocol equivalent to ``skore.EstimatorReport``."""

    _hash: int
    clear_cache: Any
    _cache: Any
    metrics: Any
    data: Any
    ml_task: str
    estimator: Any
    estimator_name_: str
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    fit: Any


@runtime_checkable
class CrossValidationReport(Protocol):
    """Protocol equivalent to ``skore.CrossValidationReport``."""

    _hash: int
    clear_cache: Any
    _cache: Any
    metrics: Any
    data: Any
    estimator_reports_: Any
    ml_task: str
    estimator: Any
    estimator_name_: str
    X: Any
    y: Any
    splitter: Any
    split_indices: Any
