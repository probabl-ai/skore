"""Protocols definition used to remove adherence to ``skore`` for type checking."""

from typing import Any, Literal, Protocol, runtime_checkable

DataSource = Literal["train", "test"] | None
Aggregate = Literal["mean", "std"]


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


class EstimatorReportMetricFunction(Protocol):
    """Protocol equivalent to a functions in ``skore.EstimatorReport.metrics``."""

    def __call__(self, data_source: DataSource) -> float | None:  # noqa: D102
        ...


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


class CrossValidationReportMetricFunction(Protocol):
    """Protocol equivalent to functions in ``skore.CrossValidationReport.metrics``."""

    def __call__(self, data_source: DataSource, aggregate: Aggregate) -> Any:  # noqa: D102
        ...
