"""Report iteration utilities inspired by ``skomlflow``."""

from __future__ import annotations

import itertools
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

from sklearn.base import BaseEstimator

from .metrics import Artifact, Metric, iter_cv_metrics, iter_estimator_metrics
from .protocol import CrossValidationReport, EstimatorReport


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


LogItem: TypeAlias = Params | Tag | Model | Artifact | Metric
NestedLogItem: TypeAlias = LogItem | tuple[str, Iterable[LogItem]]


def iter_cv(report: CrossValidationReport) -> Generator[NestedLogItem, None, None]:
    """Yield loggable objects for a cross-validation report."""
    yield from iter_cv_metrics(report)

    estimator_report = report.create_estimator_report()
    estimator = estimator_report.estimator_
    yield Params(estimator.get_params())
    yield Model(estimator)

    yield Artifact("data.analyze", report.data.analyze()._repr_html_())

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
            "cv_splitter.n_splits": report.splitter.get_n_splits(),
        }
    )


def iter_estimator(report: EstimatorReport) -> Generator[LogItem, None, None]:
    """Yield loggable objects for an estimator report."""
    yield from iter_estimator_metrics(report)

    estimator = report.estimator_
    yield Params(estimator.get_params())
    yield Model(estimator)

    yield Artifact("data.analyze", report.data.analyze()._repr_html_())
