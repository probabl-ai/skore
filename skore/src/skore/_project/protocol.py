"""Project backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from skore._project.types import ExperimentRow, OnSameStatus, ReportMetadata

if TYPE_CHECKING:
    from skore import CrossValidationReport, EstimatorReport


class ProjectBackend(Protocol):
    """Contract implemented by project plugins loaded via entry points."""

    name: str

    def put(
        self, key: str, report: EstimatorReport | CrossValidationReport
    ) -> None: ...

    def get(self, id: str) -> EstimatorReport | CrossValidationReport: ...

    def summarize(self) -> list[ReportMetadata]: ...

    def experiments(self) -> list[ExperimentRow]: ...

    def create_experiment(self, row: ExperimentRow) -> ExperimentRow: ...

    def update_experiment(
        self,
        row: ExperimentRow,
        version: int,
        *,
        on_same_status: OnSameStatus = "write",
    ) -> ExperimentRow: ...

    @staticmethod
    def delete(**kwargs: Any) -> None: ...
