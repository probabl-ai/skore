"""Utilities for building hub report payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd


def select_exportable_summary_rows(
    summary: pd.DataFrame,
    *,
    ml_task: str,
) -> pd.DataFrame:
    """Filter a summarize summary for hub export.

    Drops rows with missing scores. For binary classification, drops
    averaged rows (keeps per-label rows only).
    """
    selected = summary[summary["score"].notna()]
    if ml_task == "binary-classification":
        selected = selected[selected["average"].isna()]
    return selected


def multimetric_scalar_names(summary: pd.DataFrame) -> frozenset[str]:
    """Registry names that expand to multiple scalar submetrics.

    Multimetric scorers produce several summary rows that share the same registry
    ``name`` but differ by ``verbose_name``, with null ``label`` / ``output`` /
    ``average``. Hub ``MetricType`` is unique on ``name``, so those rows need
    distinct hub identities.
    """
    if summary.empty:
        return frozenset()

    scalar = summary[
        summary["label"].isna() & summary["output"].isna() & summary["average"].isna()
    ]
    if scalar.empty:
        return frozenset()

    n_verbose = scalar.groupby("name", dropna=False)["verbose_name"].nunique()
    return frozenset(n_verbose[n_verbose > 1].index)


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
