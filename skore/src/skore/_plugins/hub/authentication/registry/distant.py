"""Create and revoke API keys on ``skore hub``."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urljoin
from uuid import uuid4

from skore._plugins.hub.authentication.login import Token, login
from skore._plugins.hub.authentication.uri import normalize
from skore._plugins.hub.client.client import Client

PERMISSIONS = (
    "create:project",
    "read:project",
    "update:project",
    "delete:project",
)


@dataclass(frozen=True)
class Workspace:
    """Workspace membership returned by the hub."""

    id: int
    public_id: str
    permissions: frozenset[str]


@dataclass(frozen=True)
class Identity:
    """Authenticated user and their workspace memberships."""

    id: str
    workspaces: list[Workspace]


@dataclass(frozen=True)
class Key:
    """API key issued by the hub."""

    id: int
    key: str


def identity(*, host: str, token: Token) -> Identity:
    """
    Return the authenticated user and their workspace memberships.

    Parameters
    ----------
    host : str
        Hub backend URI.
    token : Token
        OAuth token used to call the hub.

    Returns
    -------
    Identity
        The current user.
    """
    url = urljoin(host, "/identity/users/me")
    headers = {"Authorization": f"Bearer {token.access}"}

    with Client() as client:
        response = client.request(method="GET", url=url, headers=headers).json()

    return Identity(
        id=response["id"],
        workspaces=[
            Workspace(
                id=int(workspace["workspace_id"]),
                public_id=workspace["public_id"],
                permissions=frozenset(workspace["permissions"]),
            )
            for workspace in response["workspace_memberships"]
        ],
    )


def expires_at_from_expires(expires: Literal["1", "3", "6"], /) -> str:
    """
    Return an absolute expiration timestamp for a relative lifetime.

    Parameters
    ----------
    expires : {"1", "3", "6"}
        Lifetime in months.

    Returns
    -------
    str or None
        Expiration as an ISO 8601 string, or ``None`` when the key does not expire.
    """
    now = datetime.now(tz=UTC)
    shifted_month = now.month - 1 + int(expires)

    new_year = now.year + shifted_month // 12
    new_month = shifted_month % 12 + 1
    new_day = min(now.day, monthrange(new_year, new_month)[1])

    expiration = now.replace(year=new_year, month=new_month, day=new_day)

    return expiration.isoformat(timespec="seconds")


def generate(
    *,
    host: str,
    workspace: str,
    name: str | None = None,
    expires: Literal["1", "3", "6", "never"] = "never",
    timeout: int = 600,
) -> Key:
    """
    Create an API key on the hub.

    Parameters
    ----------
    host : str
        Hub backend URI.
    workspace : str
        Public identifier of the workspace the key is scoped to.
    name : str or None, default=None
        Name of the API key. A random name is used when omitted.
    expires : {"1", "3", "6", "never"}, default="never"
        Lifetime of the API key, in months, or ``"never"``.
    timeout : int, default=600
        Seconds to wait for interactive authentication.

    Returns
    -------
    Key
        The issued API key.

    Raises
    ------
    PermissionError
        The user is not a member of ``workspace``, or lacks the required permissions.
    """
    host = normalize(host)
    token = login(host=host, timeout=timeout)
    user = identity(host=host, token=token)
    membership = next((w for w in user.workspaces if w.public_id == workspace), None)

    if membership is None:
        raise PermissionError(f"You are not member of {workspace!r}")

    if not membership.permissions.issuperset(set(PERMISSIONS)):
        raise PermissionError(
            f"Insuffisiant permissions, "
            f"missing {set(PERMISSIONS) - membership.permissions}"
        )

    url = urljoin(host, f"/identity/users/{user.id}/api-keys")
    headers = {"Authorization": f"Bearer {token.access}"}
    json = {
        "name": (name or uuid4().hex),
        "permissions": PERMISSIONS,
        "workspace_id": membership.id,
    }

    if expires != "never":
        json["expires_at"] = expires_at_from_expires(expires)

    with Client() as client:
        response = client.request(method="POST", url=url, headers=headers, json=json)
        response = response.json()

    return Key(id=response["api_key_id"], key=response["api_key"])


def revoke(*, host: str, id: int, timeout: int = 600) -> None:
    """
    Revoke an API key.

    Parameters
    ----------
    host : str
        Hub backend URI.
    id : int
        Identifier of the API key to revoke.
    timeout : int, default=600
        Seconds to wait for interactive authentication.
    """
    host = normalize(host)
    token = login(host=host, timeout=timeout)
    user = identity(host=host, token=token)

    with Client() as client:
        client.request(
            method="DELETE",
            url=urljoin(host, f"/identity/users/{user.id}/api-keys/{id}"),
            headers={"Authorization": f"Bearer {token.access}"},
        )
