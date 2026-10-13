"""Workspace API keys stored by the skore hub credential registry."""

from __future__ import annotations

from collections.abc import Iterator

from skore._cli.skore import apply_hub_uri


class HubError(Exception):
    """Credential failure reported through the CLI."""


def _hub_key():
    from skore._plugins.hub.authentication import key as hub_key

    return hub_key


def mint_api_key(
    *,
    host: str | None,
    workspace: str,
    name: str | None,
    expires: str,
    timeout: int,
    force: bool,
) -> None:
    """Mint a workspace API key and store it in the local registry.

    ``host``, when given, is recorded as ``SKORE_HUB_URI``. Callers that need
    frontend discovery resolve ``host`` before calling this function.
    """
    hub_key = _hub_key()
    if host:
        host = apply_hub_uri(host)
    try:
        hub_key.generate(
            host=host,
            workspace=workspace,
            name=name,
            expires=expires,
            timeout=timeout,
            force=force,
        )
    except hub_key.KeyExistsError as error:
        raise HubError(
            f"an API key for workspace '{workspace}' is already stored; "
            "pass --force to replace it."
        ) from error
    except PermissionError as error:
        raise HubError(str(error)) from error


def stored_api_key(*, host: str | None, workspace: str) -> str | None:
    """Return the stored API key for ``host`` and ``workspace``."""
    return _hub_key().get(host=host, workspace=workspace)


def require_workspace_api_key(
    *, host: str, workspace: str, login_timeout: int
) -> tuple[str, bool]:
    """Return the workspace API key, minting one when none is stored.

    Returns
    -------
    key : str
        Stored or newly minted API key.
    minted : bool
        True when this call minted the key.
    """
    existing = stored_api_key(host=host, workspace=workspace)
    if existing:
        return existing, False

    mint_api_key(
        host=host,
        workspace=workspace,
        name=None,
        expires="never",
        timeout=login_timeout,
        force=False,
    )
    minted = stored_api_key(host=host, workspace=workspace)
    if not minted:
        raise HubError(f"could not read an API key for workspace '{workspace}'.")
    return minted, True


def revoke_api_key(*, host: str | None, workspace: str, timeout: int) -> bool:
    """Revoke the stored key and return whether one was present."""
    hub_key = _hub_key()
    if host:
        host = apply_hub_uri(host)
    before = list(hub_key.keys())
    hub_key.revoke(host=host, workspace=workspace, timeout=timeout)
    return list(hub_key.keys()) != before


def iter_api_keys() -> Iterator[tuple[int, str, str]]:
    """Yield ``(id, host, workspace)`` for stored Hub API keys."""
    yield from _hub_key().keys()
