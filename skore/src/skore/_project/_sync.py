"""Project synchronization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from pandas import NA, DataFrame, Index, concat

if TYPE_CHECKING:
    from typing import Any, Protocol

    class _Summary(Protocol):
        def frame(self) -> DataFrame: ...

    class _Project(Protocol):
        def summarize(self) -> _Summary: ...

        def get(self, id: str) -> Any: ...

        def put(self, key: str, report: Any) -> Any: ...


_Direction = Literal["outbound", "inbound"]
_Status = Literal["planned", "transferred", "skipped"]
_RESULT_COLUMNS = ("key", "direction", "status")


def _snapshot(project: _Project) -> DataFrame:
    frame = project.summarize().frame()
    if frame.empty or "report_id" not in frame:
        # Treat legacy summaries from before `report_id` was stored as empty snapshots.
        return DataFrame(
            index=Index([], name="report_id", dtype=object),
            columns=Index(["backend_id", "key"]),
        )

    return (
        frame.reset_index("id")
        .rename(columns={"id": "backend_id"})
        .dropna(subset=["report_id"])
        .drop_duplicates(subset=["report_id"], keep="last")
        .set_index("report_id")
    )


def _transfer(
    reports: DataFrame,
    source: _Project,
    destination: _Project,
    direction: _Direction,
) -> None:
    for report_id, row in reports.iterrows():
        try:
            report = source.get(cast(str, row["backend_id"]))
            destination.put(cast(str, row["key"]), report)
        except Exception as exc:
            exc.add_note(f"Failed to synchronize report {report_id!r} {direction}.")
            raise


def _result(
    reports: DataFrame,
    *,
    direction: _Direction | None,
    status: _Status,
) -> DataFrame:
    result = reports.loc[:, ["key"]].copy()
    result["direction"] = NA if direction is None else direction
    result["status"] = status
    return result


def synchronize(
    source: _Project,
    destination: _Project,
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

    transfer_status: _Status = "planned" if dry_run else "transferred"
    frames = [
        _result(outbound, direction="outbound", status=transfer_status),
    ]
    if bidirectional:
        frames.append(_result(inbound, direction="inbound", status=transfer_status))
    frames.append(_result(skipped, direction=None, status="skipped"))

    result = concat(frames).reindex(columns=_RESULT_COLUMNS).rename_axis("report_id")
    return cast(DataFrame, result.astype("string"))
