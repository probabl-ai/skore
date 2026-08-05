"""Project synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import NA, DataFrame, concat

if TYPE_CHECKING:
    from skore._project.project import Project


_RESULT_COLUMNS = ["key", "direction", "status"]


def _snapshot(project: Project) -> DataFrame:
    frame = project.summarize().frame()
    if frame.empty:
        return DataFrame(columns=["backend_id", "key"]).rename_axis("report_id")

    return (
        frame.reset_index("id")
        .rename(columns={"id": "backend_id"})
        .dropna(subset=["report_id"])
        .drop_duplicates(subset=["report_id"], keep="last")
        .set_index("report_id")
    )


def _transfer(
    reports: DataFrame,
    source: Project,
    destination: Project,
    direction: str,
) -> None:
    for report_id, row in reports.iterrows():
        try:
            report = source.get(row["backend_id"])
            destination.put(row["key"], report)
        except Exception as exc:
            exc.add_note(f"Failed to synchronize report {report_id!r} {direction}.")
            raise


def _result(reports: DataFrame, *, direction: str | None, status: str) -> DataFrame:
    result = reports.loc[:, ["key"]].copy()
    result["direction"] = NA if direction is None else direction
    result["status"] = status
    return result


def synchronize(
    source: Project,
    destination: Project,
    *,
    bidirectional: bool,
    dry_run: bool,
) -> DataFrame:
    """Synchronize two projects using report IDs."""
    source_reports = _snapshot(source)
    destination_reports = _snapshot(destination)

    outbound = source_reports.loc[
        source_reports.index.difference(destination_reports.index, sort=False)
    ]
    inbound = destination_reports.loc[
        destination_reports.index.difference(source_reports.index, sort=False)
    ]
    skipped = source_reports.loc[
        source_reports.index.intersection(destination_reports.index, sort=False)
    ]

    if not dry_run:
        _transfer(outbound, source, destination, "outbound")
        if bidirectional:
            _transfer(inbound, destination, source, "inbound")

    transfer_status = "planned" if dry_run else "transferred"
    frames = [
        _result(outbound, direction="outbound", status=transfer_status),
    ]
    if bidirectional:
        frames.append(_result(inbound, direction="inbound", status=transfer_status))
    frames.append(_result(skipped, direction=None, status="skipped"))

    return concat(frames)[_RESULT_COLUMNS].rename_axis("report_id")
