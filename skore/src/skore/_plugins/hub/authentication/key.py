from __future__ import annotations

from collections.abc import Generator
from typing import Literal

from skore._plugins.hub.authentication import registry
from skore._plugins.hub.authentication.uri import URI, normalize


class KeyExistsError(Exception):
    pass


def keys() -> Generator[tuple[int, str, str]]:
    """Yield ``(id, host, workspace)`` keys stored in the registry."""
    yield from registry.local.keys()


def generate(
    *,
    host: str | None = None,
    workspace: str,
    name: str | None = None,
    expires: Literal["1", "3", "6", "never"] = "never",
    timeout: int = 600,
    force: bool = False,
) -> None:
    host = normalize(host or URI())

    if old := registry.local.get(host=host, workspace=workspace):
        if not force:
            raise KeyExistsError(
                f"An API key for host:{host!r} and workspace:{workspace!r} already "
                "exists; use force option to overwrite it."
            )

        registry.local.delete(host=host, workspace=workspace)
        registry.distant.delete(host=host, id=old.id)

    new = registry.distant.generate(
        host=host,
        workspace=workspace,
        name=name,
        expires=expires,
        timeout=timeout,
    )

    registry.local.set(id=new.id, host=host, workspace=workspace, key=new.key)


def get(*, host: str | None = None, workspace: str) -> str | None:
    host = normalize(host or URI())

    return (
        (key := registry.local.get(host=host, workspace=workspace)) and key.key or None
    )


def delete(*, host: str | None = None, workspace: str) -> None:
    host = normalize(host or URI())

    if key := registry.local.get(host=host, workspace=workspace):
        registry.local.delete(host=host, workspace=workspace)
        registry.distant.delete(host=host, id=key.id)
