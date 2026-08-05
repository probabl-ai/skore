"""Cross-mode project synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pandas import isna

from skore._project.types import ProjectMode

if TYPE_CHECKING:
    from skore._project.project import Project


@dataclass(frozen=True)
class SyncOperation:
    """A report transfer in a synchronization result.

    Parameters
    ----------
    report_id : str
        Canonical Skore report ID.
    key : str
        Source key used when storing the report at the destination.
    source_mode : {"hub", "local", "mlflow"}
        Storage mode from which the report is loaded.
    destination_mode : {"hub", "local", "mlflow"}
        Storage mode in which the report is stored.
    """

    report_id: str
    key: str
    source_mode: ProjectMode
    destination_mode: ProjectMode


@dataclass(frozen=True)
class SyncResult:
    """The planned or completed operations of a project synchronization.

    Parameters
    ----------
    operations : tuple of SyncOperation
        Planned operations for a dry-run, otherwise completed operations.
    skipped : tuple of str
        Canonical report IDs already present in both projects.
    dry_run : bool
        Whether the operations were only planned.
    """

    operations: tuple[SyncOperation, ...]
    skipped: tuple[str, ...]
    dry_run: bool


@dataclass(frozen=True)
class _ReportRef:
    report_id: str
    backend_id: str
    key: str


@dataclass(frozen=True)
class _Transfer:
    operation: SyncOperation
    source: Project
    destination: Project
    backend_id: str


def _snapshot(project: Project) -> dict[str, _ReportRef]:
    frame = project.summarize().frame()
    if frame.empty:
        return {}

    reports: dict[str, _ReportRef] = {}
    missing_backend_ids: list[str] = []

    for backend_id, row in zip(
        frame.index.get_level_values("id"), frame.to_dict("records"), strict=True
    ):
        report_id = row.get("report_id")
        if report_id is None or isna(report_id):
            missing_backend_ids.append(str(backend_id))
            continue

        report_id = str(report_id)
        # Summaries are ordered by date. Reinsert duplicate IDs so the latest stored
        # copy also determines transfer order and the key used at the destination.
        reports.pop(report_id, None)
        reports[report_id] = _ReportRef(
            report_id=report_id,
            backend_id=str(backend_id),
            key=str(row["key"]),
        )

    if missing_backend_ids:
        raise ValueError(
            f"Cannot synchronize project {project.name!r} in {project.mode!r} mode "
            "because these reports have no canonical `report_id`: "
            f"{missing_backend_ids}."
        )

    return reports


def _transfers(
    source: Project,
    destination: Project,
    source_reports: dict[str, _ReportRef],
    destination_reports: dict[str, _ReportRef],
) -> list[_Transfer]:
    return [
        _Transfer(
            operation=SyncOperation(
                report_id=report.report_id,
                key=report.key,
                source_mode=source.mode,
                destination_mode=destination.mode,
            ),
            source=source,
            destination=destination,
            backend_id=report.backend_id,
        )
        for report_id, report in source_reports.items()
        if report_id not in destination_reports
    ]


def synchronize(
    source: Project,
    destination: Project,
    *,
    bidirectional: bool,
    dry_run: bool,
) -> SyncResult:
    """Synchronize two projects using canonical report IDs."""
    source_reports = _snapshot(source)
    destination_reports = _snapshot(destination)

    if (
        source.ml_task is not None
        and destination.ml_task is not None
        and source.ml_task != destination.ml_task
    ):
        raise ValueError(
            "Cannot synchronize projects with different ML tasks: "
            f"{source.ml_task!r} and {destination.ml_task!r}."
        )

    transfers = _transfers(
        source,
        destination,
        source_reports,
        destination_reports,
    )
    if bidirectional:
        transfers.extend(
            _transfers(
                destination,
                source,
                destination_reports,
                source_reports,
            )
        )

    operations = tuple(transfer.operation for transfer in transfers)
    skipped = tuple(
        report_id for report_id in source_reports if report_id in destination_reports
    )

    if not dry_run:
        for transfer in transfers:
            try:
                report = transfer.source.get(transfer.backend_id)
                if str(report.id) != transfer.operation.report_id:
                    raise RuntimeError(
                        "The loaded report ID does not match its project summary: "
                        f"expected {transfer.operation.report_id!r}, got "
                        f"{str(report.id)!r}."
                    )
                transfer.destination.put(transfer.operation.key, report)
            except Exception as exc:
                exc.add_note(
                    f"Failed to synchronize report {transfer.operation.report_id!r} "
                    f"from {transfer.operation.source_mode!r} to "
                    f"{transfer.operation.destination_mode!r}."
                )
                raise

    return SyncResult(operations=operations, skipped=skipped, dry_run=dry_run)
