"""Metric and artifact iteration utilities inspired by ``skomlflow``."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, cast

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


def _is_frame(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "iloc")


def _is_display(value: Any) -> bool:
    return callable(getattr(value, "frame", None)) and callable(
        getattr(value, "plot", None)
    )


def iter_cv_metrics(
    report: CrossValidationReport,
) -> Generator[Artifact | Metric, Any, None]:
    """Yield metrics/artifacts for a cross-validation report."""
    report_any = cast(Any, report)
    yield Artifact(
        "metrics_details/per_split",
        report_any.metrics.summarize(aggregate=None).frame(),
    )
    yield Artifact("all_metrics", report_any.metrics.summarize().frame())

    for name, method in report_any.metrics._get_methods_for_help():
        if name in ["custom_metric", "summarize"]:
            continue
        if (
            report_any.ml_task == "multioutput-regression"
            and name == "prediction_error"
        ):
            continue

        if name == "timings":
            (fit_time,) = method(aggregate="mean").loc["Fit time (s)"]
            yield Metric("fit_time", fit_time)
            continue

        out = method()

        if _is_display(out):
            yield Artifact(f"metrics_details/{name}", out.frame())
            out.plot()
            figure = out.figure_
            try:
                yield Artifact(name, figure)
            finally:
                plt.close(figure)
            continue

        if not _is_frame(out):
            raise TypeError(f"Unexpected type for metrics.{name}(): {type(out)}")

        kwargs = {"aggregate": "mean"}
        if "multioutput" in report_any.ml_task:
            kwargs["multioutput"] = "uniform_average"

        value = method(**kwargs)
        if value.shape != (1, 1) and "classification" in report_any.ml_task:
            kwargs["average"] = "micro"
            value = method(**kwargs)
        if value.shape != (1, 1):
            raise ValueError(f"Expected single-cell df but got: {value}")

        yield Metric(name, value.iloc[0, 0])

        kwargs["aggregate"] = "std"
        std = method(**kwargs).iloc[0, 0]
        yield Metric(f"{name}.std", std)

    yield Artifact("timings", report_any.metrics.timings())


def iter_estimator_metrics(
    report: EstimatorReport,
) -> Generator[Artifact | Metric, Any, None]:
    """Yield metrics/artifacts for an estimator report."""
    report_any = cast(Any, report)
    yield Artifact("all_metrics", report_any.metrics.summarize().frame())

    for name, method in report_any.metrics._get_methods_for_help():
        if name in ["custom_metric", "summarize"]:
            continue
        if (
            report_any.ml_task == "multioutput-regression"
            and name == "prediction_error"
        ):
            continue

        out = method()

        if name == "timings":
            if "fit_time" in out:
                yield Metric("fit_time", out["fit_time"])
            continue

        if _is_display(out):
            yield Artifact(f"metrics_details/{name}", out.frame())
            out.plot()
            figure = out.figure_
            try:
                yield Artifact(name, figure)
            finally:
                plt.close(figure)
            continue

        if isinstance(out, float):
            yield Metric(name, out)
            continue

        if isinstance(out, list):
            yield Artifact(name, out)
            yield Metric(name, method(multioutput="uniform_average"))
            continue

        if isinstance(out, dict):
            yield Artifact(name, out)
            yield Metric(name, method(average="micro"))
            continue

        raise TypeError(f"Unexpected type for metrics.{name}(): {type(out)}")

    yield Artifact("timings", report_any.metrics.timings())
