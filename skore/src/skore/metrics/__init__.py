"""Metrics available for `skore` reports."""

from skore.metrics.metrics import (
    BUILTIN_METRICS,
    FitTime,
    FunctionKind,
    Metric,
    MetricLike,
    MetricRow,
    MissingKwargsError,
    PredictTime,
    Score,
    SKLearnScorer,
)
from skore.metrics.registry import MetricRegistry

__all__ = [
    "BUILTIN_METRICS",
    "FitTime",
    "FunctionKind",
    "Metric",
    "MetricLike",
    "MetricRegistry",
    "MetricRow",
    "MissingKwargsError",
    "PredictTime",
    "SKLearnScorer",
    "Score",
]
