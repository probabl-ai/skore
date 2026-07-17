"""Tests for hub report payload utilities."""

import pandas as pd

from skore._plugins.hub.report.utils import select_exportable_summary_rows


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
