"""Helpers for building hub metric payloads from reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd

from skore import CrossValidationReport, EstimatorReport


def select_exportable_metrics(
    report: EstimatorReport | CrossValidationReport,
) -> pd.DataFrame:
    """Select metric summary rows suitable for hub export.

    Drops rows with missing scores. For binary classification, drops
    averaged rows (keeps per-label rows only).
    """
    metrics_summary = report.metrics.summarize(data_source="both").summary

    if report._ml_task == "binary-classification":
        return metrics_summary[
            metrics_summary["score"].notna() & metrics_summary["average"].isna()
        ]

    return metrics_summary[metrics_summary["score"].notna()]


def multimetric_scalar_names(metrics_summary: pd.DataFrame) -> frozenset[str]:
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


def hub_metric_name(
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
