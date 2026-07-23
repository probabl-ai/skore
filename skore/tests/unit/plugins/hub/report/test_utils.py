"""Tests for hub report payload utilities."""

import pandas as pd

from skore._plugins.hub.report.utils import (
    hub_metric_name,
    multimetric_scalar_names,
    select_exportable_summary_rows,
)


def _summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_drops_rows_with_missing_scores() -> None:
    summary = _summary(
        [
            {"name": "accuracy", "score": 0.9, "average": None},
            {"name": "precision", "score": float("nan"), "average": None},
        ]
    )

    selected = select_exportable_summary_rows(summary, ml_task="regression")

    assert selected["name"].tolist() == ["accuracy"]


def test_binary_classification_drops_averaged_rows() -> None:
    summary = _summary(
        [
            {"name": "precision", "score": 0.8, "label": 0, "average": None},
            {"name": "precision", "score": 0.7, "label": 1, "average": None},
            {"name": "precision", "score": 0.75, "label": None, "average": "macro"},
        ]
    )

    selected = select_exportable_summary_rows(summary, ml_task="binary-classification")

    assert len(selected) == 2
    assert selected["average"].isna().all()


def test_multiclass_keeps_averaged_rows() -> None:
    summary = _summary(
        [
            {"name": "precision", "score": 0.8, "label": 0, "average": None},
            {"name": "precision", "score": 0.75, "label": None, "average": "macro"},
        ]
    )

    selected = select_exportable_summary_rows(
        summary, ml_task="multiclass-classification"
    )

    assert len(selected) == 2
    assert selected["average"].tolist() == [None, "macro"]


def test_multimetric_scalar_names_detects_dict_submetrics() -> None:
    summary = _summary(
        [
            {
                "name": "my_multi_scorer",
                "verbose_name": "score_a",
                "label": None,
                "output": None,
                "average": None,
            },
            {
                "name": "my_multi_scorer",
                "verbose_name": "score_b",
                "label": None,
                "output": None,
                "average": None,
            },
            {
                "name": "r2",
                "verbose_name": "R²",
                "label": None,
                "output": None,
                "average": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "label": 0,
                "output": None,
                "average": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "label": 1,
                "output": None,
                "average": None,
            },
        ]
    )

    assert multimetric_scalar_names(summary) == frozenset({"my_multi_scorer"})


def test_hub_metric_name_uses_verbose_name_for_multimetric() -> None:
    multimetric_names = frozenset({"my_multi_scorer"})

    assert (
        hub_metric_name(
            {"name": "my_multi_scorer", "verbose_name": "score_a"},
            multimetric_names=multimetric_names,
        )
        == "score_a"
    )
    assert (
        hub_metric_name(
            {"name": "r2", "verbose_name": "R²"},
            multimetric_names=multimetric_names,
        )
        == "r2"
    )
