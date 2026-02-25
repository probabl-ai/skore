"""Report, metric and artifact iteration utilities."""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

from ._matplotlib import switch_mpl_backend
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


@dataclass
class Params:
    """Model parameter payload."""

    params: dict[str, Any]


@dataclass
class Tag:
    """Tag payload."""

    key: str
    value: str


@dataclass
class Model:
    """Model payload."""

    model: BaseEstimator


CLF_METRICS = {
    # metric -> kwargs
    "accuracy": {},
    "log_loss": {},
    "recall": {"average": "micro"},
    "precision": {"average": "micro"},
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


LogItem: TypeAlias = Params | Tag | Model | Artifact | Metric
NestedLogItem: TypeAlias = LogItem | tuple[str, Iterable[LogItem]]


def _data_analyze_html(report: CrossValidationReport | EstimatorReport) -> Any:
    return report.data.analyze()._repr_html_()


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
        with switch_mpl_backend(), plt.ioff():
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
        with switch_mpl_backend(), plt.ioff():
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


def iter_cv(report: CrossValidationReport) -> Generator[NestedLogItem, None, None]:
    """Yield loggable objects for a cross-validation report."""
    yield from iter_cv_metrics(report)

    estimator_report = report.create_estimator_report()
    estimator = estimator_report.estimator_
    yield Params(estimator.get_params())
    yield Model(estimator)

    yield Artifact("data.analyze", _data_analyze_html(report))

    for split_id, estimator_report in enumerate(report.estimator_reports_):
        yield (
            f"split_{split_id}",
            itertools.chain(
                [Tag("split_id", str(split_id))], iter_estimator(estimator_report)
            ),
        )

    yield Params(
        {
            "cv_splitter.class": report.splitter.__class__.__name__,
            # FIXME? try/except here:
            "cv_splitter.n_splits": report.splitter.get_n_splits(),
        }
    )


def iter_estimator(report: EstimatorReport) -> Generator[LogItem, None, None]:
    """Yield loggable objects for an estimator report."""
    yield from iter_estimator_metrics(report)

    estimator = report.estimator_
    yield Params(estimator.get_params())
    yield Model(estimator)

    yield Artifact("data.analyze", _data_analyze_html(report))
