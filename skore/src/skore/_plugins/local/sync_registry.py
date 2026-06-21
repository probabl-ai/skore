"""Cross-mode sync registry persisted in the local workspace.

The registry maps a report (identified by its canonical ``report.id`` and the key
under which it is stored on the local side) to its counterpart on a remote
backend. It is keyed by ``(report_id, key)`` so the same report stored under
several keys is tracked independently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from skore._plugins.local.storage import DiskCacheStorage


class RemoteLink(TypedDict):
    """Mapping to a report's counterpart on a remote backend."""

    remote_id: str
    synced_at: str


class SyncRecord(TypedDict):
    """Sync state for a single ``(report_id, key)`` pair."""

    report_id: str
    key: str
    remotes: dict[str, RemoteLink]


def project_uri(*, mode: str, name: str, workspace: str | Path | None = None) -> str:
    """Return a stable identifier for a project backend.

    ``workspace`` is the per-mode discriminator that distinguishes two backends
    sharing the same ``mode`` and ``name`` (the local workspace path, the hub
    workspace name, or the mlflow tracking URI).
    """
    return f"{mode}:{workspace}:{name}"


class SyncRegistry:
    """Persist sync mappings under ``workspace/sync/``."""

    def __init__(self, workspace: Path) -> None:
        sync_dir = workspace / "sync"
        sync_dir.mkdir(parents=True, exist_ok=True)
        self._storage = DiskCacheStorage(sync_dir)

    @staticmethod
    def _composite(report_id: str, key: str) -> str:
        return f"{report_id}@{key}"

    def link(
        self,
        *,
        report_id: str,
        key: str,
        remote_uri: str,
        remote_id: str,
    ) -> None:
        """Record that ``(report_id, key)`` maps to ``remote_id`` on ``remote_uri``."""
        composite = self._composite(report_id, key)

        if composite in self._storage:
            existing = self._storage[composite]
            remotes = dict(existing["remotes"])
        else:
            remotes = {}

        remotes[remote_uri] = RemoteLink(
            remote_id=remote_id,
            synced_at=datetime.now(UTC).isoformat(),
        )
        self._storage[composite] = SyncRecord(
            report_id=report_id,
            key=key,
            remotes=remotes,
        )

    def linked(
        self,
        *,
        report_id: str,
        key: str,
        remote_uri: str,
        remote_id: str,
    ) -> bool:
        """Return whether ``(report_id, key)`` is linked to ``remote_id``."""
        composite = self._composite(report_id, key)
        if composite not in self._storage:
            return False
        link = self._storage[composite]["remotes"].get(remote_uri)
        return link is not None and link["remote_id"] == remote_id
