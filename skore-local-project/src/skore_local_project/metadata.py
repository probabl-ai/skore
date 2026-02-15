"""Class definition of the ``metadata`` objects used by ``project``."""

from __future__ import annotations

from abc import ABC
from contextlib import suppress
from dataclasses import InitVar, dataclass, field, fields
from datetime import datetime, timezone
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

from joblib import hash

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from skore import CrossValidationReport, EstimatorReport


def cast_to_float(value: Any) -> float | None:
    """Cast value to float."""
    with suppress(TypeError):
        float_value = float(value)

        if isfinite(float_value):
            return float_value

    return None


def report_type(
    report: EstimatorReport | CrossValidationReport,
) -> Literal["cross-validation", "estimator"]:
    """Human readable type of a report."""
    # _report_type is defined on skore._sklearn._base._BaseReport; both report
    # types inherit it (mypy may not resolve it when type-checking this package)
    return cast(
        Literal["cross-validation", "estimator"],
        report._report_type,  # type: ignore[union-attr]
    )


@dataclass(kw_only=True)
class ReportMetadata(ABC):
    """
    Metadata used to persist a report to local storage.

    Attributes
    ----------
    report : EstimatorReport | CrossValidationReport
        The report on which to calculate the metadata to persist.
    artifact_id : str
        ID of the artifact in the artifacts storage.
    project_name : str
        The name of the project the metadata should be associated with.
    date : str
        The date the metadata were created.
    key : str
        The key to associate to the report.
    learner : str
        The name of the report's estimator.
    ml_task : str
        The type of ML task covered by the report.
    report_type : str
        The type of the report.
    dataset : str
        The hash of the targets.
    """

    report: InitVar[EstimatorReport | CrossValidationReport]

    artifact_id: str
    project_name: str
    key: str
    date: str = field(init=False)
    learner: str = field(init=False)
    ml_task: str = field(init=False)
    report_type: str = field(init=False)
    dataset: str = field(init=False)

    def __iter__(self) -> Generator[tuple[str, str], None, None]:
        """Iterate over the metadata."""
        for field in fields(self):  # noqa: F402
            yield (field.name, getattr(self, field.name))

    def __post_init__(self, report: EstimatorReport | CrossValidationReport) -> None:
        """Initialize dynamic fields."""
        self.date = datetime.now(timezone.utc).isoformat()
        self.learner = report.estimator_name_
        self.ml_task = report.ml_task
        self.report_type = report_type(report)
        self.dataset = hash(report.y_test if hasattr(report, "y_test") else report.y)


@dataclass(kw_only=True)
class EstimatorReportMetadata(ReportMetadata):  # noqa: D101
    rmse: float | None = field(init=False)
    log_loss: float | None = field(init=False)
    roc_auc: float | None = field(init=False)
    fit_time: float | None = field(init=False)
    predict_time: float | None = field(init=False)

    @staticmethod
    def metric(report: EstimatorReport, name: str) -> float | None:
        """Compute metric."""
        if not hasattr(report.metrics, name):
            return None

        return cast_to_float(getattr(report.metrics, name)(data_source="test"))

    def __post_init__(self, report: EstimatorReport) -> None:  # type: ignore[override]
        """Initialize dynamic fields."""
        super().__post_init__(report)

        self.rmse = self.metric(report, "rmse")
        self.log_loss = self.metric(report, "log_loss")
        self.roc_auc = self.metric(report, "roc_auc")

        # timings must be calculated last
        self.fit_time = report.metrics.timings().get("fit_time")
        self.predict_time = report.metrics.timings().get("predict_time_test")


@dataclass(kw_only=True)
class CrossValidationReportMetadata(ReportMetadata):  # noqa: D101
    rmse_mean: float | None = field(init=False)
    log_loss_mean: float | None = field(init=False)
    roc_auc_mean: float | None = field(init=False)
    fit_time_mean: float | None = field(init=False)
    predict_time_mean: float | None = field(init=False)

    @staticmethod
    def metric(report: CrossValidationReport, name: str) -> float | None:
        """Compute metric."""
        if not hasattr(report.metrics, name):
            return None

        dataframe = getattr(report.metrics, name)(
            data_source="test",
            aggregate="mean",
        )

        return cast_to_float(dataframe.iloc[0, 0])

    @staticmethod
    def timing(report: CrossValidationReport, label: str) -> float | None:
        """Compute timing."""
        dataframe = report.metrics.timings(aggregate="mean")

        try:
            series = dataframe.loc[label]
        except KeyError:
            return None

        return cast_to_float(series.iloc[0])

    def __post_init__(self, report: CrossValidationReport) -> None:  # type: ignore[override]
        """Initialize dynamic fields."""
        super().__post_init__(report)

        self.rmse_mean = self.metric(report, "rmse")
        self.log_loss_mean = self.metric(report, "log_loss")
        self.roc_auc_mean = self.metric(report, "roc_auc")

        # timings must be calculated last
        self.fit_time_mean = self.timing(report, "Fit time (s)")
        self.predict_time_mean = self.timing(report, "Predict time test (s)")
