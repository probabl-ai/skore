"""Report iteration utilities inspired by ``skomlflow``."""

from __future__ import annotations

import itertools
from collections.abc import Generator
from dataclasses import dataclass
from importlib.metadata import version
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
class RawDataArtifact:
    """Raw data artifact payload."""

    name: str
    payload: Any


def _safe_param_value(value: Any) -> bool | int | float | str:
    if isinstance(value, bool | int | float | str):
        return value
    if value is None:
        return "None"
    return str(value)


def iter_cv(
    report: CrossValidationReport,
    *,
    subsample_dataset: bool = False,
    iter_children: bool = True,
) -> Generator[Any, Any, None]:
    """Yield loggable objects for a cross-validation report."""
    report_any = cast(Any, report)
    yield from iter_cv_metrics(report_any)

    estimator_report = report_any.create_estimator_report()
    estimator = estimator_report.estimator
    yield Params({k: _safe_param_value(v) for k, v in estimator.get_params().items()})
    yield estimator

    yield Artifact("data.analyze", report_any.data.analyze()._repr_html_())
    if not subsample_dataset:
        yield RawDataArtifact("X", report_any.X)
        yield RawDataArtifact("y", report_any.y)

    if iter_children:
        for split_id, estimator_report in enumerate(report_any.estimator_reports_):
            split_infos: list[Any] = [Tag("split_id", str(split_id))]
            if not subsample_dataset:
                train_indices, test_indices = report_any.split_indices[split_id]
                split_infos.append(
                    RawDataArtifact("split_train_indices", train_indices)
                )
                split_infos.append(RawDataArtifact("split_test_indices", test_indices))

            filtered_iter = (
                obj
                for obj in iter_estimator(
                    estimator_report, subsample_dataset=subsample_dataset
                )
                if not isinstance(obj, RawDataArtifact)
            )

            yield (f"split_{split_id}", itertools.chain(split_infos, filtered_iter))

    yield Params(
        {
            "cv_splitter.class": report_any.splitter.__class__.__name__,
            "cv_splitter.n_splits": report_any.splitter.get_n_splits(),
        }
    )
    import cloudpickle  # type: ignore[import-untyped]

    yield RawDataArtifact("splitter", cloudpickle.dumps(report_any.splitter))
    yield Tag("skore_version", version("skore"))
    yield Tag("ml_task", report_any.ml_task)


def iter_estimator(
    report: EstimatorReport, *, subsample_dataset: bool = False
) -> Generator[Any, Any, None]:
    """Yield loggable objects for an estimator report."""
    report_any = cast(Any, report)
    yield from iter_estimator_metrics(report_any)

    estimator = report_any.estimator
    yield Params({k: _safe_param_value(v) for k, v in estimator.get_params().items()})
    yield estimator

    yield Artifact("data.analyze", report_any.data.analyze()._repr_html_())
    if not subsample_dataset:
        if report_any.X_train is not None:
            yield RawDataArtifact("X_train", report_any.X_train)
            yield RawDataArtifact("y_train", report_any.y_train)
        yield RawDataArtifact("X_test", report_any.X_test)
        yield RawDataArtifact("y_test", report_any.y_test)

    yield Tag("skore_version", version("skore"))
    yield Tag("ml_task", report_any.ml_task)
