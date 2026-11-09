from datetime import UTC, datetime
from json import loads
from types import SimpleNamespace
from urllib.parse import urljoin

from httpx import Response
from pytest import mark, raises

from skore._plugins.hub.authentication.registry import distant
from skore._plugins.hub.authentication.registry.distant import (
    PERMISSIONS,
    Identity,
    Workspace,
    expires_at_from_expires,
)

HOST = "https://hub.example"
USER_URL = urljoin(HOST, "/identity/users/me")
KEYS_URL = urljoin(HOST, "/identity/users/user-1/api-keys")


def freeze(monkeypatch, moment):
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    monkeypatch.setattr(distant, "datetime", Frozen)


def membership(*, public_id="workspace", permissions=PERMISSIONS):
    return {
        "workspace_id": 9,
        "public_id": public_id,
        "permissions": list(permissions),
    }


def mock_identity(respx_mock, workspaces):
    respx_mock.get(USER_URL).mock(
        Response(
            200,
            json={"id": "user-1", "workspace_memberships": workspaces},
        )
    )


def install_login(monkeypatch):
    logins = []

    def login(*, host, timeout=600):
        logins.append((host, timeout))
        return SimpleNamespace(access="access-token")

    monkeypatch.setattr(distant, "login", login)
    return logins


def test_identity(respx_mock):
    mock_identity(respx_mock, [membership()])

    user = distant.identity(host=HOST, token=SimpleNamespace(access="access-token"))

    assert user == Identity(
        id="user-1",
        workspaces=[
            Workspace(
                id=9,
                public_id="workspace",
                permissions=frozenset(PERMISSIONS),
            )
        ],
    )
    assert respx_mock.calls.last.request.headers["Authorization"] == (
        "Bearer access-token"
    )


@mark.parametrize(
    ("expires", "expected"),
    [
        ("1", "2026-02-28T15:04:05+00:00"),
        ("3", "2026-04-30T15:04:05+00:00"),
        ("6", "2026-07-31T15:04:05+00:00"),
    ],
)
def test_expires_at_from_expires(monkeypatch, expires, expected):
    freeze(monkeypatch, datetime(2026, 1, 31, 15, 4, 5, 123456, tzinfo=UTC))

    assert expires_at_from_expires(expires) == expected


def test_expires_at_from_expires_clamps_to_leap_day(monkeypatch):
    freeze(monkeypatch, datetime(2024, 1, 31, 8, 0, 0, tzinfo=UTC))

    assert expires_at_from_expires("1") == "2024-02-29T08:00:00+00:00"


def test_expires_at_from_expires_crosses_year(monkeypatch):
    freeze(monkeypatch, datetime(2026, 11, 15, 8, 0, 0, tzinfo=UTC))

    assert expires_at_from_expires("3") == "2027-02-15T08:00:00+00:00"


@mark.respx()
def test_generate(monkeypatch, respx_mock):
    freeze(monkeypatch, datetime(2026, 1, 31, 15, 4, 5, tzinfo=UTC))
    logins = install_login(monkeypatch)
    mock_identity(
        respx_mock,
        [membership(permissions=[*PERMISSIONS, "extra:perm"])],
    )
    respx_mock.post(KEYS_URL).mock(
        Response(200, json={"api_key_id": 42, "api_key": "secret"})
    )

    key = distant.generate(
        host="HTTPS://Hub.Example/",
        workspace="workspace",
        name="laptop",
        expires="1",
        timeout=15,
    )

    assert key == distant.Key(id=42, key="secret")
    assert logins == [("https://hub.example", 15)]
    assert loads(respx_mock.calls.last.request.content) == {
        "name": "laptop",
        "permissions": list(PERMISSIONS),
        "workspace_id": 9,
        "expires_at": "2026-02-28T15:04:05+00:00",
    }
    assert respx_mock.calls.last.request.headers["Authorization"] == (
        "Bearer access-token"
    )


@mark.respx()
def test_generate_defaults(monkeypatch, respx_mock):
    install_login(monkeypatch)
    monkeypatch.setattr(distant, "uuid4", lambda: SimpleNamespace(hex="abc123"))
    mock_identity(respx_mock, [membership()])
    respx_mock.post(KEYS_URL).mock(
        Response(200, json={"api_key_id": 7, "api_key": "secret"})
    )

    distant.generate(host=HOST, workspace="workspace")

    assert loads(respx_mock.calls.last.request.content) == {
        "name": "abc123",
        "permissions": list(PERMISSIONS),
        "workspace_id": 9,
    }


@mark.respx()
def test_generate_requires_membership(monkeypatch, respx_mock):
    install_login(monkeypatch)
    mock_identity(respx_mock, [membership(public_id="other")])

    with raises(PermissionError, match="not member of 'workspace'"):
        distant.generate(host=HOST, workspace="workspace")

    assert len(respx_mock.calls) == 1


@mark.respx()
def test_generate_requires_permissions(monkeypatch, respx_mock):
    install_login(monkeypatch)
    mock_identity(
        respx_mock,
        [membership(permissions=PERMISSIONS[:-1])],
    )

    with raises(PermissionError, match="missing"):
        distant.generate(host=HOST, workspace="workspace")

    assert len(respx_mock.calls) == 1


@mark.respx()
def test_revoke(monkeypatch, respx_mock):
    logins = install_login(monkeypatch)
    mock_identity(respx_mock, [membership()])
    route = respx_mock.delete(urljoin(HOST, "/identity/users/user-1/api-keys/42")).mock(
        Response(204)
    )

    distant.revoke(host="HTTPS://Hub.Example/", id=42, timeout=15)

    assert route.called
    assert logins == [("https://hub.example", 15)]
    assert route.calls.last.request.headers["Authorization"] == "Bearer access-token"
