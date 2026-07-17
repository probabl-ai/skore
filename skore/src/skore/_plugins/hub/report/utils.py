"""Utilities for building hub report payloads."""

from __future__ import annotations

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
