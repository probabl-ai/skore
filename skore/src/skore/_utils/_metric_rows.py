"""Helpers to convert metric scores into display-ready rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from skore._sklearn.types import DataSource, MLTask, PositiveLabel

if TYPE_CHECKING:
    from skore._sklearn.metrics import Metric


def metric_score_to_rows(
    score: float | list | dict,
    *,
    metric: Metric,
    ml_task: MLTask,
    data_source: DataSource,
    estimator_name: str,
    pos_label: PositiveLabel = None,
    kwargs: dict[str, Any] | None = None,
) -> list[dict]:
    """Expand a metric score into display rows based on the ML task.

    Parameters
    ----------
    score : float, dict, or list
        The metric score.

    metric : Metric
        The metric instance (provides ``verbose_name``, ``greater_is_better``,
        and default ``kwargs``).

    ml_task : str
        The ML task (e.g. ``"binary-classification"``).

    data_source : {"test", "train"}
        The data source to use.

    estimator_name : str
        Name shown in the display.

    pos_label : label, default=None
        Positive label for binary classification.

    kwargs : dict, optional
        Keyword arguments used for the score call. Default is ``metric.kwargs``.
    """
    if kwargs is None:
        kwargs = metric.kwargs

    row: dict = {
        "metric": metric.verbose_name,
        "estimator_name": estimator_name,
        "data_source": data_source,
        "greater_is_better": metric.greater_is_better,
        "label": None,
        "average": None,
        "output": None,
        "score": score,
    }

    if ml_task == "binary-classification" and kwargs.get("average") == "binary":
        return [{**row, "label": kwargs.get("pos_label", pos_label)}]
    if ml_task in ("binary-classification", "multiclass-classification"):
        if isinstance(score, dict):
            return [{**row, "label": label, "score": score[label]} for label in score]
        return [{**row, "average": kwargs.get("average")}]
    if ml_task == "multioutput-regression":
        if isinstance(score, list):
            return [{**row, "output": idx, "score": s} for idx, s in enumerate(score)]
        return [{**row, "average": kwargs.get("multioutput")}]
    return [row]


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert display rows to a DataFrame, preserving nullable dtypes."""
    data = pd.DataFrame(rows)

    if "greater_is_better" in data.columns:
        data["greater_is_better"] = data["greater_is_better"].astype(pd.BooleanDtype())

    if any(isinstance(r["label"], bool) for r in rows):
        data["label"] = data["label"].astype(pd.BooleanDtype())
    elif any(isinstance(r["label"], int) for r in rows):
        data["label"] = data["label"].astype(pd.Int64Dtype())

    if any(isinstance(r["output"], int) for r in rows):
        data["output"] = data["output"].astype(pd.Int64Dtype())

    return data
