from datetime import datetime, timedelta, timezone
from urllib.parse import urljoin

from pytest import mark, raises
from httpx import Response
from skore_hub_project.authentication.login import login
from skore_hub_project.client.api import URI
from skore_hub_project.client.client import AuthenticationError
from skore_hub_project.authentication.token import Token

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

REFRESH_URL = urljoin(URI, "identity/oauth/token/refresh")
LOGIN_URL = urljoin(URI, "identity/oauth/device/login")
PROBE_URL = urljoin(URI, "identity/oauth/device/code-probe")
CALLBACK_URL = urljoin(URI, "identity/oauth/device/callback")
TOKEN_URL = urljoin(URI, "identity/oauth/device/token")


@mark.respx(assert_all_mocked=False)
def test_login_with_token(respx_mock):
    assert not Token.exists()

    Token.save("A", "B", DATETIME_MAX)

    assert Token.exists()
    assert Token.access() == "A"

    login()

    assert not respx_mock.calls
    assert Token.exists()
    assert Token.access() == "A"


def test_login_with_expired_token(respx_mock):
    respx_mock.post(REFRESH_URL).mock(
        Response(
            200,
            json={
                "access_token": "D",
                "refresh_token": "E",
                "expires_at": DATETIME_MAX,
            },
        )
    )

    assert not Token.exists()

    Token.save("A", "B", DATETIME_MIN)

    assert Token.exists()
    assert Token.access(refresh=False) == "A"

    login()

    assert Token.exists()
    assert Token.access() == "D"


def test_login(monkeypatch, respx_mock):
    def open_webbrowser(url):
        open_webbrowser.url = url
        return True

    monkeypatch.setattr(
        "skore_hub_project.authentication.login.open_webbrowser", open_webbrowser
    )

    respx_mock.post(REFRESH_URL).mock(Response(404))
    respx_mock.get(LOGIN_URL).mock(
        Response(
            200,
            json={
                "authorization_url": "<url>",
                "device_code": "<device>",
                "user_code": "<user>",
            },
        )
    )
    respx_mock.get(PROBE_URL).mock(Response(200))
    respx_mock.get(CALLBACK_URL).mock(Response(200))
    respx_mock.get(TOKEN_URL).mock(
        Response(
            200,
            json={
                "token": {
                    "access_token": "D",
                    "refresh_token": "E",
                    "expires_at": DATETIME_MAX,
                }
            },
        )
    )

    assert not Token.exists()

    Token.save("A", "B", DATETIME_MIN)

    assert Token.exists()
    assert Token.access(refresh=False) == "A"

    login()

    # assert routes parameters
    # faire que probe retourne 1 erreur puis 1 succès ?

    assert open_webbrowser.url == "<url>"
    assert Token.exists()
    assert Token.access(refresh=False) == "D"


def test_login_timeout(): ...
