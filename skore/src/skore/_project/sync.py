"""Cross-mode project synchronization engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pandas import Timestamp, isna, to_datetime

from skore._plugins.local.sync_registry import SyncRegistry, project_uri
from skore._project._sync_result import (
    SyncConflict,
    SyncConflictError,
    SyncEntry,
    SyncResult,
    SyncSkip,
)
from skore._project.types import ConflictPolicy, SyncDirection

if TYPE_CHECKING:
    from skore._project.project import Project


@dataclass(frozen=True)
class _ReportRef:
    """The latest report for a key, with its summarize id and date."""

    key: str
    id: str
    date: Timestamp


@dataclass(frozen=True)
class _Op:
    """A planned transfer, executed in the second phase."""

    direction: Literal["put", "get"]
    origin: Project
    destination: Project
    origin_uri: str
    destination_uri: str
    ref: _ReportRef
    key: str


def _project_uri(project: Project) -> str:
    """Return a backend identifier that also distinguishes same-named projects.

    The discriminator is the local workspace path, the hub workspace name, or the
    mlflow tracking URI, so two backends sharing ``mode`` and ``name`` (but living
    in different workspaces/servers) never collide.
    """
    discriminator = (
        project.tracking_uri if project.mode == "mlflow" else project.workspace
    )
    return project_uri(mode=project.mode, name=project.name, workspace=discriminator)


def _sync_registry_for_pair(left: Project, right: Project) -> SyncRegistry | None:
    """Return a registry whose location is independent of argument order.

    When both projects are local, the owner is chosen deterministically so that
    ``a.sync_with(b)`` and ``b.sync_with(a)`` share the same registry file.
    """
    candidates: list[tuple[str, Path]] = []
    for project in (left, right):
        if project.mode != "local":
            continue
        # ``Project.workspace`` is typed as ``Path | str | None``; local always Path.
        workspace = cast(Path, project.workspace)
        candidates.append((_project_uri(project), workspace))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return SyncRegistry(candidates[0][1])


def _validate_ml_task_compatibility(left: Project, right: Project) -> None:
    """Raise if the two projects hold reports for different ML tasks."""
    tasks = {task for task in (left.ml_task, right.ml_task) if task is not None}
    if len(tasks) > 1:
        raise ValueError(
            f"Cannot synchronize projects with different ML tasks. Got {tasks}."
        )


def _row_id(index_value: object) -> str:
    """Return the report id from a summarize index value (plain or MultiIndex)."""
    if isinstance(index_value, tuple):
        return str(index_value[1])
    return str(index_value)


def _latest_by_key(project: Project) -> dict[str, _ReportRef]:
    """Return the most recent report reference for each key in ``project``."""
    frame = project.summarize().frame()
    if frame.empty:
        return {}

    latest: dict[str, _ReportRef] = {}
    for index, row in frame.iterrows():
        latest[row["key"]] = _ReportRef(
            key=row["key"],
            id=_row_id(index),
            date=to_datetime(row["date"], utc=True),
        )
    return latest


def _canonical_id(project: Project, ref: _ReportRef) -> str:
    """Return the report's stable ``report.id``.

    For local projects the summarize id already is the canonical id; for remote
    projects the report must be fetched.
    """
    if project.mode == "local":
        return ref.id
    return str(project.get(ref.id).id)


def _registry_linked(
    registry: SyncRegistry | None,
    *,
    self_proj: Project,
    self_ref: _ReportRef,
    other_proj: Project,
    other_ref: _ReportRef,
    self_uri: str,
    other_uri: str,
) -> bool:
    """Return whether the two refs are linked, without fetching reports.

    The local side's summarize id equals its canonical id, which is the registry
    key, so the lookup never downloads a report.
    """
    if registry is None:
        return False
    if self_proj.mode == "local":
        return registry.linked(
            report_id=self_ref.id,
            key=self_ref.key,
            remote_uri=other_uri,
            remote_id=other_ref.id,
        )
    if other_proj.mode == "local":
        return registry.linked(
            report_id=other_ref.id,
            key=other_ref.key,
            remote_uri=self_uri,
            remote_id=self_ref.id,
        )
    return False


def _register_pair_link(
    registry: SyncRegistry | None,
    *,
    self_proj: Project,
    self_ref: _ReportRef,
    other_proj: Project,
    other_ref: _ReportRef,
    self_uri: str,
    other_uri: str,
) -> None:
    """Record the link between two already-identical refs, for each local side."""
    if registry is None:
        return
    if self_proj.mode == "local":
        registry.link(
            report_id=self_ref.id,
            key=self_ref.key,
            remote_uri=other_uri,
            remote_id=other_ref.id,
        )
    if other_proj.mode == "local":
        registry.link(
            report_id=other_ref.id,
            key=other_ref.key,
            remote_uri=self_uri,
            remote_id=self_ref.id,
        )


def _resolve_conflict(
    *,
    on_conflict: ConflictPolicy,
    self_ref: _ReportRef,
    other_ref: _ReportRef,
) -> tuple[Literal["put", "get", "skip", "keep_both"], str | None]:
    """Decide the action and resolution label for a diverging key.

    Returns the action to take (``put``/``get``/``skip``/``keep_both``) together
    with a human-readable resolution label (or ``None`` when left unresolved),
    according to the ``on_conflict`` policy.
    """
    if on_conflict == "error":
        raise SyncConflictError(
            f"Conflict on key {self_ref.key!r}: "
            f"self id {self_ref.id!r}, other id {other_ref.id!r}."
        )
    if on_conflict == "skip":
        return "skip", None
    if on_conflict == "source_wins":
        return "put", "source_wins"
    if on_conflict == "destination_wins":
        return "get", "destination_wins"
    if on_conflict == "keep_both":
        return "keep_both", "keep_both"
    if on_conflict == "latest_wins":
        self_date, other_date = self_ref.date, other_ref.date
        if isna(self_date) and isna(other_date):
            return "put", "latest_wins_put"
        if isna(self_date):
            return "get", "latest_wins_get"
        if isna(other_date):
            return "put", "latest_wins_put"
        if self_date >= other_date:
            return "put", "latest_wins_put"
        return "get", "latest_wins_get"

    raise ValueError(f"Unknown conflict policy {on_conflict!r}.")


def _resolve_destination_id(
    project: Project,
    key: str,
    before_ids: set[str],
) -> str:
    """Find the destination summarize id for the report just written.

    Robust to backends that transform the key on write (e.g. slugification): the
    new id is identified by diffing the summarize ids captured before the write.
    """
    frame = project.summarize().frame()

    by_key = [_row_id(index) for index in frame[frame["key"] == key].index]
    new_by_key = [value for value in by_key if value not in before_ids]
    if new_by_key:
        return new_by_key[-1]
    if by_key:
        return by_key[-1]

    all_ids = [_row_id(index) for index in frame.index]
    new_ids = [value for value in all_ids if value not in before_ids]
    if new_ids:
        return new_ids[-1]

    raise RuntimeError(f"Could not locate the report for key {key!r} after writing it.")


def _execute(op: _Op, registry: SyncRegistry | None) -> SyncEntry:
    """Transfer one report to its destination and record the sync link."""
    report = op.origin.get(op.ref.id)
    report_id = str(report.id)

    before_ids = {_row_id(index) for index in op.destination.summarize().frame().index}
    op.destination.put(op.key, report)
    destination_id = _resolve_destination_id(op.destination, op.key, before_ids)

    if registry is not None:
        if op.origin.mode == "local":
            registry.link(
                report_id=report_id,
                key=op.ref.key,
                remote_uri=op.destination_uri,
                remote_id=destination_id,
            )
        if op.destination.mode == "local":
            registry.link(
                report_id=report_id,
                key=op.key,
                remote_uri=op.origin_uri,
                remote_id=op.ref.id,
            )

    return SyncEntry(
        key=op.key,
        report_id=report_id,
        source_id=op.ref.id,
        destination_id=destination_id,
        direction=op.direction,
    )


def _run_ops(
    ops: list[_Op],
    registry: SyncRegistry | None,
    dry_run: bool,
) -> tuple[list[SyncEntry], list[SyncEntry], list[tuple[str, BaseException]]]:
    """Execute planned ops, returning put/got entries and per-report failures.

    When ``dry_run`` is ``True``, placeholder entries are produced without
    transferring any report.
    """
    put: list[SyncEntry] = []
    got: list[SyncEntry] = []
    failed: list[tuple[str, BaseException]] = []

    for op in ops:
        if dry_run:
            entry = SyncEntry(
                key=op.key,
                report_id="",
                source_id=op.ref.id,
                destination_id="",
                direction=op.direction,
            )
        else:
            try:
                entry = _execute(op, registry)
            except Exception as exc:  # noqa: BLE001 - report per-report failures
                failed.append((op.key, exc))
                continue

        (put if op.direction == "put" else got).append(entry)

    return put, got, failed


def sync_with(
    self: Project,
    other: Project,
    *,
    direction: SyncDirection = "both",
    on_conflict: ConflictPolicy = "latest_wins",
    dry_run: bool = False,
) -> SyncResult:
    """Synchronize ``self`` with ``other``.

    See :meth:`~skore.Project.sync_with` for the public API documentation.
    """
    self_uri = _project_uri(self)
    other_uri = _project_uri(other)

    if self is other or self_uri == other_uri:
        raise ValueError("Cannot synchronize a project with itself.")

    _validate_ml_task_compatibility(self, other)

    registry = _sync_registry_for_pair(self, other)

    self_refs = _latest_by_key(self)
    other_refs = _latest_by_key(other)

    ops: list[_Op] = []
    skipped: list[SyncSkip] = []
    conflicts: list[SyncConflict] = []

    for key in sorted(set(self_refs) | set(other_refs)):
        self_ref = self_refs.get(key)
        other_ref = other_refs.get(key)

        if self_ref is not None and other_ref is None:
            if direction in ("put", "both"):
                ops.append(_Op("put", self, other, self_uri, other_uri, self_ref, key))
            continue

        if self_ref is None and other_ref is not None:
            if direction in ("get", "both"):
                ops.append(_Op("get", other, self, other_uri, self_uri, other_ref, key))
            continue

        assert self_ref is not None and other_ref is not None

        if self_ref.id == other_ref.id or _registry_linked(
            registry,
            self_proj=self,
            self_ref=self_ref,
            other_proj=other,
            other_ref=other_ref,
            self_uri=self_uri,
            other_uri=other_uri,
        ):
            skipped.append(SyncSkip(key=key, reason="already_synced"))
            continue

        if not dry_run and _canonical_id(self, self_ref) == _canonical_id(
            other, other_ref
        ):
            _register_pair_link(
                registry,
                self_proj=self,
                self_ref=self_ref,
                other_proj=other,
                other_ref=other_ref,
                self_uri=self_uri,
                other_uri=other_uri,
            )
            skipped.append(SyncSkip(key=key, reason="already_synced"))
            continue

        action, resolution = _resolve_conflict(
            on_conflict=on_conflict,
            self_ref=self_ref,
            other_ref=other_ref,
        )

        if action == "skip":
            conflicts.append(
                SyncConflict(
                    key=key,
                    self_id=self_ref.id,
                    other_id=other_ref.id,
                    resolution=None,
                )
            )
            continue

        if action == "keep_both":
            if direction in ("get", "both"):
                alt_key = f"{key} (from {other.name})"
                if alt_key not in self_refs:
                    ops.append(
                        _Op(
                            "get",
                            other,
                            self,
                            other_uri,
                            self_uri,
                            other_ref,
                            alt_key,
                        )
                    )
            if direction in ("put", "both"):
                alt_key = f"{key} (from {self.name})"
                if alt_key not in other_refs:
                    ops.append(
                        _Op(
                            "put",
                            self,
                            other,
                            self_uri,
                            other_uri,
                            self_ref,
                            alt_key,
                        )
                    )
            conflicts.append(
                SyncConflict(
                    key=key,
                    self_id=self_ref.id,
                    other_id=other_ref.id,
                    resolution="keep_both",
                )
            )
            continue

        if action == "put" and direction in ("put", "both"):
            ops.append(_Op("put", self, other, self_uri, other_uri, self_ref, key))
            conflicts.append(
                SyncConflict(
                    key=key,
                    self_id=self_ref.id,
                    other_id=other_ref.id,
                    resolution=resolution,
                )
            )
        elif action == "get" and direction in ("get", "both"):
            ops.append(_Op("get", other, self, other_uri, self_uri, other_ref, key))
            conflicts.append(
                SyncConflict(
                    key=key,
                    self_id=self_ref.id,
                    other_id=other_ref.id,
                    resolution=resolution,
                )
            )
        else:
            # The resolution is not allowed by ``direction``; leave it unresolved.
            conflicts.append(
                SyncConflict(
                    key=key,
                    self_id=self_ref.id,
                    other_id=other_ref.id,
                    resolution=None,
                )
            )

    put, got, failed = _run_ops(ops, registry, dry_run)

    return SyncResult(
        put=tuple(put),
        got=tuple(got),
        skipped=tuple(skipped),
        conflicts=tuple(conflicts),
        failed=tuple(failed),
    )
