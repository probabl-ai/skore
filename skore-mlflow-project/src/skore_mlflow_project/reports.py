"""Report iteration utilities inspired by ``skomlflow``."""

from __future__ import annotations

import itertools
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, cast

from .metrics import Artifact, iter_cv_metrics, iter_estimator_metrics
from .protocol import CrossValidationReport, EstimatorReport


@dataclass
class Params:
    """Model parameter payload."""

    params: dict[str, bool | int | float | str]


@dataclass
class Tag:
    """Tag payload."""

    key: str
    value: str


@dataclass
class Model:
    """Model payload."""

    model: Any


def _safe_param_value(value: Any) -> bool | int | float | str:
    if isinstance(value, bool | int | float | str):
        return value
    if value is None:
        return "None"
    return str(value)


def iter_cv(report: CrossValidationReport) -> Generator[Any, Any, None]:
    """Yield loggable objects for a cross-validation report."""
    report_any = cast(Any, report)
    yield from iter_cv_metrics(report_any)

    estimator_report = report_any.create_estimator_report()
    estimator = estimator_report.estimator
    yield Params({k: _safe_param_value(v) for k, v in estimator.get_params().items()})
    yield Model(estimator)

    yield Artifact("data.analyze", report_any.data.analyze()._repr_html_())

    for split_id, estimator_report in enumerate(report_any.estimator_reports_):
        yield (
            f"split_{split_id}",
            itertools.chain(
                [Tag("split_id", str(split_id))], iter_estimator(estimator_report)
            ),
        )

    yield Params(
        {
            "cv_splitter.class": report_any.splitter.__class__.__name__,
            "cv_splitter.n_splits": report_any.splitter.get_n_splits(),
        }
    )


def iter_estimator(report: EstimatorReport) -> Generator[Any, Any, None]:
    """Yield loggable objects for an estimator report."""
    report_any = cast(Any, report)
    yield from iter_estimator_metrics(report_any)

    estimator = report_any.estimator
    yield Params({k: _safe_param_value(v) for k, v in estimator.get_params().items()})
    yield Model(estimator)

    yield Artifact("data.analyze", report_any.data.analyze()._repr_html_())
