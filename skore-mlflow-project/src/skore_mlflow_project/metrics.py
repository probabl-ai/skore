"""Metric and artifact iteration utilities inspired by ``skomlflow``."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt

from .protocol import CrossValidationReport, EstimatorReport


@dataclass
class Artifact:
    """Artifact payload and target name."""

    name: str
    payload: Any


@dataclass
class Metric:
    """Scalar metric payload."""

    name: str
    value: float


CLF_METRICS = {
    # metric -> kwargs
    "accuracy": {},
    "log_loss": {},
    "recall": {"average": "micro", "multi_class": "ovr"},
    "precision": {"average": "micro", "multi_class": "ovr"},
    "roc_auc": {"average": "micro", "multi_class": "ovr"},
}

REG_METRICS = {
    # metric -> kwargs
    "r2": {"multioutput": "uniform_average"},
    "rmse": {"multioutput": "uniform_average"},
}

# mappings per task type:
METRICS = {
    "binary-classification": CLF_METRICS,
    "multiclass-classification": CLF_METRICS,
    "regression": REG_METRICS,
    "multioutput-regression": REG_METRICS,
}

PLOTS = {
    "binary-classification": ["confusion_matrix", "roc", "precision_recall"],
    "multiclass-classification": ["confusion_matrix", "roc", "precision_recall"],
    "regression": ["prediction_error"],
    "multioutput-regression": [],
}


def iter_cv_metrics(
    report: CrossValidationReport,
) -> Generator[Artifact | Metric, Any, None]:
    """Yield metrics/artifacts for a cross-validation report."""
    ml_task = report.ml_task
    report_any = report
    # NOTE: we could use flat_index=True in summarize, but we have to flatten
    # other frames anyway, so we don't do it here.
    yield Artifact(
        "metrics_details/per_split",
        report_any.metrics.summarize(aggregate=None).frame(),
    )
    yield Artifact("all_metrics", report_any.metrics.summarize().frame())

    for name, kwargs in METRICS[ml_task].items():
        method = getattr(report_any.metrics, name)
        yield Metric(name, method(**kwargs, aggregate="mean").iloc[0, 0])
        yield Metric(f"{name}_std", method(**kwargs, aggregate="std").iloc[0, 0])
        if not kwargs or ml_task == "regression":
            continue
        yield Artifact(name, method())

    for name in PLOTS[ml_task]:
        method = getattr(report_any.metrics, name)
        display = method()
        yield Artifact(f"metrics_details/{name}", display.frame())
        display.plot()
        figure = display.figure_
        try:
            yield Artifact(name, figure)
        finally:
            plt.close(figure)
        continue

    timings = report_any.metrics.timings()
    fit_time = timings.loc["Fit time (s)"].loc["mean"]
    predict_time = timings.loc["Predict time test (s)"].loc["mean"]
    yield Metric("fit_time", fit_time)
    yield Metric("predict_time", predict_time)
    yield Artifact("timings", report_any.metrics.timings())


def iter_estimator_metrics(
    report: EstimatorReport,
) -> Generator[Artifact | Metric, Any, None]:
    """Yield metrics/artifacts for an estimator report."""
    ml_task = report.ml_task
    report_any = report
    # NOTE: we could do the same things with data_source="train"

    yield Artifact("all_metrics", report_any.metrics.summarize(flat_index=True).frame())

    for name, kwargs in METRICS[ml_task].items():
        method = getattr(report_any.metrics, name)
        yield Metric(name, method(**kwargs))
        if not kwargs or ml_task == "regression":
            continue
        yield Artifact(name, method())

    for name in PLOTS[ml_task]:
        method = getattr(report_any.metrics, name)
        display = method()
        yield Artifact(f"metrics_details/{name}", display.frame())
        display.plot()
        figure = display.figure_
        try:
            yield Artifact(name, figure)
        finally:
            plt.close(figure)
        continue

    timings = report_any.metrics.timings()
    yield Metric("fit_time", timings["fit_time"])
    yield Metric("predict_time", timings["predict_time_test"])
