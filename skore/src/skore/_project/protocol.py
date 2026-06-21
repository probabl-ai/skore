"""Private protocol for project backend plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from skore._project.types import ReportMetadata

if TYPE_CHECKING:
    from skore import CrossValidationReport, EstimatorReport


class ProjectBackend(Protocol):
    """Contract implemented by project plugins loaded via entry points."""

    @property
    def name(self) -> str: ...

    def put(
        self, key: str, report: EstimatorReport | CrossValidationReport
    ) -> None: ...

    def get(self, id: str) -> EstimatorReport | CrossValidationReport: ...

    def summarize(self) -> list[ReportMetadata]: ...

    @staticmethod
    def delete(**kwargs: Any) -> None: ...

    def __repr__(self) -> str: ...
