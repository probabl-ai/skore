from datetime import datetime, timezone
from itertools import repeat
from json import dumps, loads
from urllib.parse import urljoin

from httpx import HTTPError, Response, TimeoutException
from pytest import mark, raises

from skore_hub_project.authentication.token import (
    Token,
    get_oauth_device_code_probe,
    get_oauth_device_login,
    get_oauth_device_token,
    post_oauth_device_callback,
    post_oauth_refresh_token,
)
from skore_hub_project.authentication.uri import URI

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

REFRESH_URL = "identity/oauth/token/refresh"
LOGIN_URL = "identity/oauth/device/login"
PROBE_URL = "identity/oauth/device/code-probe"
CALLBACK_URL = "identity/oauth/device/callback"
TOKEN_URL = "identity/oauth/device/token"


@mark.parametrize("success_uri", [None, "toto"])
@mark.respx()
def test_get_oauth_device_login(respx_mock, success_uri):
    respx_mock.get(urljoin(URI, LOGIN_URL)).mock(
        Response(
            200,
            json={
                "authorization_url": "A",
                "device_code": "B",
                "user_code": "C",
            },
        )
    )

    params = {} if success_uri is None else {"success_uri": "toto"}
    authorization_url, device_code, user_code = get_oauth_device_login(
        success_uri=success_uri
    )

    assert dict(respx_mock.calls.last.request.url.params) == params
    assert authorization_url == "A"
    assert device_code == "B"
    assert user_code == "C"


@mark.respx()
def test_post_oauth_device_callback(respx_mock):
    route = respx_mock.post(urljoin(URI, CALLBACK_URL)).mock(Response(201, json={}))

    post_oauth_device_callback("my_state", "my_user_code")

    assert route.called
    assert route.calls.last.request.read() == b"state=my_state&user_code=my_user_code"


@mark.respx()
def test_get_oauth_device_token(respx_mock):
    respx_mock.get(urljoin(URI, TOKEN_URL)).mock(
        Response(
            200,
            json={
                "token": {
                    "access_token": "A",
                    "refresh_token": "B",
                    "expires_at": "C",
                }
            },
        )
    )

    access_token, refresh_token, expires_at = get_oauth_device_token("<device_code>")

    assert access_token == "A"
    assert refresh_token == "B"
    assert expires_at == "C"
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>"
    }


@mark.respx()
def test_get_oauth_device_code_probe(monkeypatch, respx_mock):
    monkeypatch.setattr("skore_hub_project.authentication.token.sleep", lambda _: None)
    respx_mock.get(urljoin(URI, PROBE_URL)).mock(
        side_effect=[
            Response(400),
            Response(400),
            Response(200),
        ]
    )

    get_oauth_device_code_probe("<device_code>")

    assert len(respx_mock.calls) == 3
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>",
    }


@mark.respx()
def test_get_oauth_device_code_probe_exception(respx_mock):
    respx_mock.get(urljoin(URI, PROBE_URL)).mock(
        side_effect=[
            Response(404),
            Response(400),
            Response(200),
        ]
    )

    with raises(HTTPError) as excinfo:
        get_oauth_device_code_probe("<device_code>")

    assert excinfo.value.response.status_code == 404
    assert len(respx_mock.calls) == 1
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>",
    }


@mark.respx()
def test_get_oauth_device_code_probe_timeout(respx_mock):
    respx_mock.get(urljoin(URI, PROBE_URL)).mock(
        side_effect=[
            Response(400),
            Response(400),
            Response(200),
        ]
    )

    with raises(TimeoutException):
        get_oauth_device_code_probe("<device_code>", timeout=0)

    assert len(respx_mock.calls) == 1
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>",
    }


@mark.respx()
def test_post_oauth_refresh_token(respx_mock):
    route = respx_mock.post(urljoin(URI, REFRESH_URL)).mock(
        Response(
            200,
            json={"access_token": "A", "refresh_token": "B", "expires_at": "C"},
        )
    )

    access_token, refresh_token, expires_at = post_oauth_refresh_token("token")

    assert route.calls.last.request.read() == b'{"refresh_token":"token"}'
    assert access_token == "A"
    assert refresh_token == "B"
    assert expires_at == "C"


class TestToken:
    @mark.respx()
    def test_init(self, monkeypatch, respx_mock):
        def open_webbrowser(url):
            open_webbrowser.url = url
            return True

        monkeypatch.setattr(
            "skore_hub_project.authentication.token.open_webbrowser", open_webbrowser
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

        token = Token()

        assert token._Token__access == "D"
        assert token._Token__refreshment == "E"
        assert token._Token__expiration == datetime.fromisoformat(DATETIME_MAX)
        assert open_webbrowser.url == "<url>"

    @mark.respx()
    def test_init_timeout(self, monkeypatch, respx_mock):
        monkeypatch.setattr(
            "skore_hub_project.authentication.token.open_webbrowser",
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
            Token(timeout=0)

    @mark.respx()
    def test_call(self, monkeypatch, respx_mock):
        monkeypatch.setattr(
            "skore_hub_project.authentication.token.open_webbrowser",
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

        assert Token()() == {"Authorization": "Bearer D"}

    @mark.respx()
    def test_call_with_expired_token(self, monkeypatch, respx_mock):
        monkeypatch.setattr(
            "skore_hub_project.authentication.token.open_webbrowser",
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

        token = Token()

        assert token._Token__access == "D"
        assert token._Token__refreshment == "E"
        assert token._Token__expiration == datetime.fromisoformat(DATETIME_MIN)

        assert token() == {"Authorization": "Bearer F"}

        assert token._Token__access == "F"
        assert token._Token__refreshment == "G"
        assert token._Token__expiration == datetime.fromisoformat(DATETIME_MAX)
