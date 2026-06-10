from __future__ import annotations

from typing import Any

import skrub
from sklearn.base import MetaEstimatorMixin
from sklearn.pipeline import Pipeline

from skore._sklearn.types import EstimatorLike
from skore._utils._skrub import is_skrub_learner


def _markdown_estimator_kind(estimator: EstimatorLike) -> str:
    if isinstance(estimator, skrub.DataOp):
        return "skrub DataOp"
    if is_skrub_learner(estimator):
        return "skrub SkrubLearner"
    if isinstance(estimator, Pipeline):
        return "Pipeline"
    if isinstance(estimator, MetaEstimatorMixin):
        inner = getattr(estimator, "best_estimator_", None) or getattr(
            estimator, "estimator", None
        )
        if inner is not None:
            return (
                f"meta-estimator {type(estimator).__name__} "
                f"wrapping {type(inner).__name__}"
            )
        return f"meta-estimator {type(estimator).__name__}"
    if type(estimator).__module__.startswith("sklearn."):
        return "scikit-learn estimator"
    return f"{type(estimator).__module__.split('.')[0]} estimator"


def report_markdown_context(report: Any) -> dict[str, object]:
    return {
        "report_class_name": report.__class__.__name__,
        "estimator_name": report.estimator_name_,
        "estimator_kind": _markdown_estimator_kind(report.estimator),
        "ml_task": report.ml_task,
        "pos_label_repr": (
            repr(report._pos_label) if report._pos_label is not None else None
        ),
        "estimator_repr": repr(report.estimator_),
        "checks_text": repr(report.checks.summarize(fast_mode=True)),
    }


def markdown_data_section(summary: dict, *, data_label: str) -> dict[str, object]:
    return {
        "data_label": data_label,
        "data_n_rows": summary["n_rows"],
        "data_n_columns": summary["n_columns"],
        "data_n_constant_columns": summary["n_constant_columns"],
        "data_columns": [
            {
                "name": col["name"],
                "dtype": col["dtype"],
                "null_count": col.get("null_count", ""),
                "n_unique": col.get("n_unique", ""),
            }
            for col in summary["columns"]
        ],
    }
