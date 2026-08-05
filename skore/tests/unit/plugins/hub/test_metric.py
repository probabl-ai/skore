"""Tests for hub metric helpers."""

from types import SimpleNamespace

import pandas as pd

from skore._plugins.hub.metric import (
    find_multimetric_scalar_names,
    get_hub_metric_name,
    select_exportable_metrics,
)


def _metrics_summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _report(ml_task: str, rows: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        _ml_task=ml_task,
        metrics=SimpleNamespace(
            summarize=lambda data_source: SimpleNamespace(
                summary=_metrics_summary(rows)
            )
        ),
    )


def test_drops_rows_with_missing_scores() -> None:
    report = _report(
        "regression",
        [
            {"name": "accuracy", "score": 0.9, "average": None},
            {"name": "precision", "score": float("nan"), "average": None},
        ],
    )

    selected = select_exportable_metrics(report)

    assert selected["name"].tolist() == ["accuracy"]


def test_binary_keeps_averaged_rows() -> None:
    report = _report(
        "binary-classification",
        [
            {"name": "precision", "score": 0.8, "label": 0, "average": None},
            {"name": "precision", "score": 0.7, "label": 1, "average": None},
            {"name": "precision", "score": 0.75, "label": None, "average": "macro"},
        ],
    )

    selected = select_exportable_metrics(report)

    assert len(selected) == 3
    assert selected["average"].tolist() == [None, None, "macro"]


def test_multiclass_keeps_averaged_rows() -> None:
    report = _report(
        "multiclass-classification",
        [
            {"name": "precision", "score": 0.8, "label": 0, "average": None},
            {"name": "precision", "score": 0.75, "label": None, "average": "macro"},
        ],
    )

    selected = select_exportable_metrics(report)

    assert len(selected) == 2
    assert selected["average"].tolist() == [None, "macro"]


def test_find_multimetric_scalar_names_detects_dict_submetrics() -> None:
    metrics_summary = _metrics_summary(
        [
            {
                "name": "my_multi_scorer",
                "verbose_name": "score_a",
                "data_source": "test",
                "label": None,
                "output": None,
                "average": None,
            },
            {
                "name": "my_multi_scorer",
                "verbose_name": "score_b",
                "data_source": "test",
                "label": None,
                "output": None,
                "average": None,
            },
            {
                "name": "r2",
                "verbose_name": "R²",
                "data_source": "test",
                "label": None,
                "output": None,
                "average": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "test",
                "label": 0,
                "output": None,
                "average": None,
            },
            {
                "name": "precision",
                "verbose_name": "Precision",
                "data_source": "test",
                "label": 1,
                "output": None,
                "average": None,
            },
        ]
    )

    assert find_multimetric_scalar_names(metrics_summary) == frozenset(
        {"my_multi_scorer"}
    )


def test_get_hub_metric_name_uses_verbose_name_for_multimetric() -> None:
    multimetric_names = frozenset({"my_multi_scorer"})

    assert (
        get_hub_metric_name(
            {"name": "my_multi_scorer", "verbose_name": "score_a"},
            multimetric_names=multimetric_names,
        )
        == "score_a"
    )
    assert (
        get_hub_metric_name(
            {"name": "r2", "verbose_name": "R²"},
            multimetric_names=multimetric_names,
        )
        == "r2"
    )
