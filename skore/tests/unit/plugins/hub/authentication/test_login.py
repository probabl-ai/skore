from datetime import UTC, datetime

from httpx import Response, TimeoutException
from pytest import mark, raises

from skore._plugins.hub.authentication import login as login_module
from skore._plugins.hub.authentication.uri import URI

DATETIME_MIN = datetime.min.replace(tzinfo=UTC).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=UTC).isoformat()

REFRESH_URL = "identity/oauth/token/refresh"
LOGIN_URL = "identity/oauth/device/login"
PROBE_URL = "identity/oauth/device/code-probe"
CALLBACK_URL = "identity/oauth/device/callback"
TOKEN_URL = "identity/oauth/device/token"


@mark.respx()
def test_login_with_token(monkeypatch, respx_mock):
    login_module.login.cache_clear()
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.login.open_webbrowser",
        lambda _: True,
    )
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

    token = login_module.login(host=URI())

    assert token.access == "D"


@mark.respx()
def test_login_with_expired_token(monkeypatch, respx_mock):
    login_module.login.cache_clear()
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.login.open_webbrowser",
        lambda _: True,
    )
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
                    "expires_at": DATETIME_MIN,
                }
            },
        )
    )
    respx_mock.post(REFRESH_URL).mock(
        Response(
            200,
            json={
                "access_token": "F",
                "refresh_token": "G",
                "expires_at": DATETIME_MAX,
            },
        )
    )

    token = login_module.login(host=URI())

    assert token.access == "F"


@mark.respx()
def test_login_with_token_timeout(monkeypatch, respx_mock):
    login_module.login.cache_clear()
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.login.open_webbrowser",
        lambda _: True,
    )

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
    respx_mock.get(PROBE_URL).mock(Response(400))

    # Simulate a user who does not complete the authentication process:
    # - the token can't be acknowledged by the hub until the user is logged in; 400
    # - the token can't be created; timeout
    with raises(TimeoutException):
        login_module.login(host=URI(), timeout=0)
