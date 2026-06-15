from __future__ import annotations

from typing import Any

import skrub
from sklearn.base import MetaEstimatorMixin
from sklearn.pipeline import Pipeline
from skrub._reporting._summarize import summarize_dataframe

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


def _subreport_estimator_metadata(report: Any) -> dict[str, str | None]:
    timings = report.metrics.timings()
    metadata: dict[str, str | None] = {
        "estimator name": f"`{report.estimator_name_}`",
        "kind": _markdown_estimator_kind(report.estimator),
    }
    if report._report_type == "cross-validation":
        metadata["fit time"] = (
            f"{timings.loc['Fit time (s)', 'mean']:.3f} s"
            f" (± {timings.loc['Fit time (s)', 'std']:.3f})"
        )
        metadata["predict time (on test set)"] = (
            f"{timings.loc['Predict time test (s)', 'mean']:.3f} s"
            f" (± {timings.loc['Predict time test (s)', 'std']:.3f})"
        )
        metadata["cross-validation folds"] = str(len(report.reports_))
        metadata["splitter"] = (
            repr(report.splitter)
            if report.splitter is not None
            else f"{len(report.split_indices)} folds"
        )
    else:
        fit_time = timings.get("fit_time")
        predict_time = timings.get("predict_time_test")
        metadata["fit time"] = f"{fit_time:.3g} s" if fit_time is not None else None
        metadata["predict time (on test set)"] = (
            f"{predict_time:.3g} s" if predict_time is not None else None
        )
    return metadata


def comparison_estimator_markdown_context(comparison_report: Any) -> dict[str, object]:
    labels = list(comparison_report.reports_.keys())
    metadata_list = [
        _subreport_estimator_metadata(report)
        for report in comparison_report.reports_.values()
    ]
    row_labels = [
        "estimator name",
        "kind",
        "fit time",
        "predict time (on test set)",
    ]
    if comparison_report._report_type == "comparison-cross-validation":
        row_labels.extend(["cross-validation folds", "splitter"])
    return {
        "estimator_labels": labels,
        "ml_task": comparison_report._ml_task,
        "pos_label_repr": (
            repr(comparison_report._pos_label)
            if comparison_report._pos_label is not None
            else None
        ),
        "estimator_rows": [
            {
                "label": row_label,
                "cells": [metadata.get(row_label) or "" for metadata in metadata_list],
            }
            for row_label in row_labels
        ],
    }


def comparison_data_markdown_context(comparison_report: Any) -> dict[str, object]:
    all_report_data = []
    for report_name, report in comparison_report.reports_.items():
        if report._report_type == "cross-validation":
            all_report_data.append(
                {
                    "report_name": report_name,
                    "data_label": "full",
                    "dataframe": report.data._prepare_dataframe_for_display(
                        with_y=True,
                        subsample=None,
                        subsample_strategy="head",
                        seed=None,
                    ),
                }
            )
        else:  # "estimator"
            all_report_data.append(
                {
                    "report_name": report_name,
                    "data_label": "full" if report.X_train is not None else "test",
                    "dataframe": report.data._prepare_dataframe_for_display(
                        data_source="both" if report.X_train is not None else "test",
                        with_y=True,
                        subsample=None,
                        subsample_strategy="head",
                        seed=None,
                    ),
                }
            )
    first_report_data = all_report_data[0]

    if all(
        report_data["dataframe"].equals(first_report_data["dataframe"])
        and report_data["data_label"] == first_report_data["data_label"]
        for report_data in all_report_data[1:]
    ):
        summary = summarize_dataframe(
            first_report_data["dataframe"],
            with_plots=False,
            with_associations=False,
            verbose=0,
        )
        return {
            "data_mode": "unified",
            **markdown_data_section(
                summary, data_label=first_report_data["data_label"]
            ),
        }
    return {
        "data_mode": "shapes",
        "data_shapes": [
            {
                "report_name": report_data["report_name"],
                "n_rows": report_data["dataframe"].shape[0],
                "n_columns": report_data["dataframe"].shape[1],
            }
            for report_data in all_report_data
        ],
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
