"""Project synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

if TYPE_CHECKING:
    from skore.project.project import Project


Direction = Literal["outbound", "inbound"]
Status = Literal["planned", "transferred", "skipped"]


def _snapshot(project: Project) -> pd.DataFrame:
    frame = project.summarize().frame()
    if frame.empty or "report_id" not in frame:
        # Treat legacy summaries from before `report_id` was stored as empty snapshots.
        return pd.DataFrame(
            index=pd.Index([], name="report_id", dtype=object),
            columns=pd.Index(["backend_id", "key"]),
        )

    return (
        frame.reset_index("id")
        .rename(columns={"id": "backend_id"})
        .dropna(subset=["report_id"])
        .drop_duplicates(subset=["report_id"], keep="last")
        .set_index("report_id")
    )


def _transfer(
    reports: pd.DataFrame,
    source: Project,
    destination: Project,
) -> None:
    for report_id, row in reports.iterrows():
        try:
            report = source.get(cast(str, row["backend_id"]))
            destination.put(cast(str, row["key"]), report)
        except Exception as exc:
            exc.add_note(
                f"Failed to synchronize report {report_id!r} "
                f"from {source} to {destination}."
            )
            raise


def _build_result_frame(
    reports: pd.DataFrame,
    *,
    direction: Direction | None,
    status: Status,
) -> pd.DataFrame:
    result = reports.loc[:, ["key"]]
    result["direction"] = pd.NA if direction is None else direction
    result["status"] = status
    return result


def synchronize(
    source: Project,
    destination: Project,
    *,
    bidirectional: bool,
    dry_run: bool,
) -> pd.DataFrame:
    """Synchronize two projects."""
    source_reports = _snapshot(source)
    destination_reports = _snapshot(destination)

    outbound_reports = source_reports.loc[
        source_reports.index.difference(destination_reports.index, sort=False)
    ]
    inbound_reports = destination_reports.loc[
        destination_reports.index.difference(source_reports.index, sort=False)
    ]
    skipped_reports = source_reports.loc[
        source_reports.index.intersection(destination_reports.index, sort=False)
    ]

    if not dry_run:
        _transfer(outbound_reports, source, destination)
        if bidirectional:
            _transfer(inbound_reports, destination, source)

    transfer_status: Status = "planned" if dry_run else "transferred"
    frames = [
        _build_result_frame(
            outbound_reports,
            direction="outbound",
            status=transfer_status,
        ),
    ]
    if bidirectional:
        frames.append(
            _build_result_frame(
                inbound_reports,
                direction="inbound",
                status=transfer_status,
            )
        )
    frames.append(
        _build_result_frame(skipped_reports, direction=None, status="skipped")
    )

    result = pd.concat(frames).reindex(columns=["key", "direction", "status"])
    result = result.rename_axis("report_id")
    return cast(pd.DataFrame, result.astype("string"))
