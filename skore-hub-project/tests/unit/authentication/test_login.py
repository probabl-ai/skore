from datetime import datetime, timezone
from itertools import repeat

from httpx import Response, TimeoutException
from pytest import mark, raises
from skore_hub_project.authentication import token as Token
from skore_hub_project.authentication.login import login

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

REFRESH_URL = "identity/oauth/token/refresh"
LOGIN_URL = "identity/oauth/device/login"
PROBE_URL = "identity/oauth/device/code-probe"
CALLBACK_URL = "identity/oauth/device/callback"
TOKEN_URL = "identity/oauth/device/token"


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
    respx_mock.post(CALLBACK_URL).mock(Response(200))
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

    assert open_webbrowser.url == "<url>"
    assert Token.exists()
    assert Token.access(refresh=False) == "D"


def test_login_timeout(monkeypatch, respx_mock):
    monkeypatch.setattr(
        "skore_hub_project.authentication.login.open_webbrowser",
        lambda _: True,
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
    respx_mock.get(PROBE_URL).mock(side_effect=repeat(Response(400)))
    respx_mock.post(CALLBACK_URL).mock(Response(200))
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

    with raises(TimeoutException):
        login(timeout=0)

    assert Token.exists()
    assert Token.access(refresh=False) == "A"
