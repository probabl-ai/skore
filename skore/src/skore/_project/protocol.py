"""Project backend protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from skore._project.types import ExperimentRow, ReportMetadata

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

    def upsert_experiment(self, row: ExperimentRow) -> None: ...

    def objective(self) -> str: ...

    def set_objective(self, text: str) -> None: ...

    @staticmethod
    def delete(**kwargs: Any) -> None: ...
