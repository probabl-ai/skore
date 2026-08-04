"""Class definition and helpers for hub metric payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, Literal, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from skore import CrossValidationReport, EstimatorReport

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


class Metric(BaseModel, Generic[Report]):
    """
    Payload used to send a metric.

    Attributes
    ----------
    name : str
        Name of the metric.
    verbose_name : str
        Verbose name of the metric.
    data_source : Literal["train", "test"] | None, optional
        Data source of the metric when it can be declined in several ways, default None.
    greater_is_better: bool | None, optional
        Indicator of "greater value is better", default None.
    value : float
        Value of the metric.
    label : bool | int | float | str | None, optional
        Class label for per-class classification metrics, default None.
    output : int | None, optional
        Output index for multioutput regression metrics, default None.
    average : str | None, optional
        Averaging mode for metrics aggregated across labels or outputs, default None.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    verbose_name: str
    data_source: Literal["train", "test"] | None
    greater_is_better: bool | None
    value: float
    label: bool | int | float | str | None = None
    output: int | None = None
    average: str | None = None
    # See https://github.com/probabl-ai/skore/issues/3025
    position: None = Field(default=None)


def select_exportable_metrics(
    report: EstimatorReport | CrossValidationReport,
) -> pd.DataFrame:
    """Select metric summary rows suitable for hub export.

    Drops rows with missing scores.
    """
    metrics_summary = report.metrics.summarize(data_source="both").summary
    return metrics_summary[metrics_summary["score"].notna()]


def find_multimetric_scalar_names(metrics_summary: pd.DataFrame) -> frozenset[str]:
    """Registry names that expand to multiple scalar submetrics.

    Multimetric scorers produce several summary rows that share the same registry
    ``name`` but differ by ``verbose_name``, with null ``label`` / ``output`` /
    ``average``. Hub ``MetricType`` is unique on ``name``, so those rows need
    distinct hub identities.
    """
    if metrics_summary.empty:
        return frozenset()

    scalar = metrics_summary[
        metrics_summary["label"].isna()
        & metrics_summary["output"].isna()
        & metrics_summary["average"].isna()
    ]
    if scalar.empty:
        return frozenset()

    keys = scalar[["name", "data_source"]]
    return frozenset(keys[keys.duplicated()]["name"])


def get_hub_metric_name(
    row: Mapping[str, Any],
    *,
    multimetric_names: frozenset[str],
) -> str:
    """Return the hub ``Metric.name`` for a summarize / aggregate row.

    For multimetric scalar rows, use the submetric key (``verbose_name``),
    matching sklearn's multimetric convention. Otherwise keep the registry
    ``name`` so built-ins and single-value custom scorers are unchanged.
    """
    name = cast(str, row["name"])
    if name in multimetric_names:
        return cast(str, row["verbose_name"])
    return name
