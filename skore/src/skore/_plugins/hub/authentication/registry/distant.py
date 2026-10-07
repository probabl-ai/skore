from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import UTC, datetime
from os import environ
from typing import Literal
from urllib.parse import urljoin
from uuid import uuid4

from skore._plugins.hub.authentication.login import Token, login
from skore._plugins.hub.authentication.uri import URI, normalize
from skore._plugins.hub.client.client import Client

PERMISSIONS = (
    "create:project",
    "read:project",
    "update:project",
    "delete:project",
)


@dataclass(frozen=True)
class Workspace:
    id: int
    public_id: str
    permissions: frozenset[str]


@dataclass(frozen=True)
class Identity:
    id: str
    workspaces: list[Workspace]


@dataclass(frozen=True)
class Key:
    id: int
    key: str


def identity(*, host: str, token: Token) -> Identity:
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


def expires_at_from_expires(expires: Literal["1", "3", "6", "never"], /) -> str | None:
    if expires == "never":
        return None

    now = datetime.now(tz=UTC)
    shifted_month = now.month - 1 + int(expires)

    new_year = now.year + shifted_month // 12
    new_month = shifted_month % 12 + 1
    new_day = min(now.day, monthrange(new_year, new_month)[1])

    expiration = now.replace(year=new_year, month=new_month, day=new_day)

    return expiration.isoformat(timespec="seconds")  # .replace("+00:00", "Z")


def generate(
    *,
    host: str | None = None,
    workspace: str,
    name: str | None = None,
    expires: Literal["1", "3", "6", "never"] = "never",
    timeout: int = 600,
) -> Key:
    host = normalize(host or URI())
    environ["SKORE_HUB_URI"] = host

    token = login(timeout=timeout)
    user = identity(host=host, token=token)

    if not (
        membership := next(
            (w for w in user.workspaces if w.public_id == workspace),
            None,
        )
    ):
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

    if expires is not None:
        json["expires_at"] = expires_at_from_expires(expires)

    with Client() as client:
        response = client.request(method="POST", url=url, headers=headers, json=json)
        response = response.json()

    return Key(id=response["api_key_id"], key=response["api_key"])


def delete(*, host: str | None = None, id: int, timeout: int = 600) -> None:
    host = normalize(host or URI())
    environ["SKORE_HUB_URI"] = host

    token = login(timeout=timeout)
    user = identity(host=host, token=token)

    with Client() as client:
        client.request(
            method="DELETE",
            url=urljoin(host, f"/identity/users/{user.id}/api-keys/{id}"),
            headers={"Authorization": f"Bearer {token.access}"},
        )
