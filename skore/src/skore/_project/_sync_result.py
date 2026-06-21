"""Result types for cross-mode project synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SyncEntry:
    """A report transferred during synchronization."""

    key: str
    report_id: str
    source_id: str
    destination_id: str
    direction: Literal["put", "get"]


@dataclass(frozen=True)
class SyncSkip:
    """A report skipped during synchronization."""

    key: str
    reason: str


@dataclass(frozen=True)
class SyncConflict:
    """An unresolved or resolved conflict between two projects."""

    key: str
    self_id: str
    other_id: str
    resolution: str | None


@dataclass(frozen=True)
class SyncResult:
    """Outcome of a :meth:`~skore.Project.sync_with` operation."""

    put: tuple[SyncEntry, ...]
    got: tuple[SyncEntry, ...]
    skipped: tuple[SyncSkip, ...]
    conflicts: tuple[SyncConflict, ...]
    failed: tuple[tuple[str, BaseException], ...]

    def summary(self) -> str:
        """Return a human-readable one-line summary.

        The conflict count reflects only unresolved conflicts (those left for the
        user to handle); conflicts auto-resolved by the ``on_conflict`` policy are
        reported as the resulting put/get instead.
        """
        unresolved = sum(
            1 for conflict in self.conflicts if conflict.resolution is None
        )
        parts = [
            f"put {len(self.put)}",
            f"got {len(self.got)}",
            f"skipped {len(self.skipped)}",
            f"conflicts {unresolved}",
        ]
        if self.failed:
            parts.append(f"failed {len(self.failed)}")
        return ", ".join(parts)


class SyncConflictError(Exception):
    """Raised when ``on_conflict='error'`` and a conflict is detected."""
