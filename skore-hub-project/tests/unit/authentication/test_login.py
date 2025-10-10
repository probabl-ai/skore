from datetime import datetime, timezone
from itertools import repeat
from urllib.parse import urljoin

from httpx import HTTPError, Response, TimeoutException
from pytest import mark, raises
from skore_hub_project.authentication import token, uri
from skore_hub_project.authentication.login import (
    get_oauth_device_code_probe,
    get_oauth_device_login,
    get_oauth_device_token,
    login,
    post_oauth_device_callback,
)


@mark.parametrize("success_uri", [None, "toto"])
def test_get_oauth_device_login(respx_mock, success_uri):
    respx_mock.get(urljoin(uri.DEFAULT, "identity/oauth/device/login")).mock(
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


def test_post_oauth_device_callback(respx_mock):
    route = respx_mock.post(
        urljoin(uri.DEFAULT, "identity/oauth/device/callback")
    ).mock(Response(201, json={}))

    post_oauth_device_callback("my_state", "my_user_code")

    assert route.called
    assert route.calls.last.request.read() == b"state=my_state&user_code=my_user_code"


def test_get_oauth_device_token(respx_mock):
    respx_mock.get(urljoin(uri.DEFAULT, "identity/oauth/device/token")).mock(
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


def test_get_oauth_device_code_probe(monkeypatch, respx_mock):
    monkeypatch.setattr("skore_hub_project.authentication.login.sleep", lambda _: None)
    respx_mock.get(urljoin(uri.DEFAULT, "identity/oauth/device/code-probe")).mock(
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


def test_get_oauth_device_code_probe_exception(respx_mock):
    respx_mock.get(urljoin(uri.DEFAULT, "identity/oauth/device/code-probe")).mock(
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


def test_get_oauth_device_code_probe_timeout(respx_mock):
    respx_mock.get(urljoin(uri.DEFAULT, "identity/oauth/device/code-probe")).mock(
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


DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

REFRESH_URL = "identity/oauth/token/refresh"
LOGIN_URL = "identity/oauth/device/login"
PROBE_URL = "identity/oauth/device/code-probe"
CALLBACK_URL = "identity/oauth/device/callback"
TOKEN_URL = "identity/oauth/device/token"


@mark.respx(assert_all_mocked=False)
def test_login_with_token(respx_mock):
    assert not token.Filepath().exists()

    token.persist("A", "B", DATETIME_MAX)

    assert token.Filepath().exists()
    assert token.access(refresh=False) == "A"

    login()

    assert not respx_mock.calls
    assert token.Filepath().exists()
    assert token.access(refresh=False) == "A"


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

    assert not token.Filepath().exists()

    token.persist("A", "B", DATETIME_MIN)

    assert token.Filepath().exists()
    assert token.access(refresh=False) == "A"

    login()

    assert token.Filepath().exists()
    assert token.access(refresh=False) == "D"


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

    assert not token.Filepath().exists()

    token.persist("A", "B", DATETIME_MIN)

    assert token.Filepath().exists()

    assert token.access(refresh=False) == "A"
    assert uri.URI() == uri.DEFAULT

    login()

    assert open_webbrowser.url == "<url>"
    assert token.Filepath().exists()

    assert token.access(refresh=False) == "D"
    assert uri.URI() == uri.DEFAULT


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

    assert not token.Filepath().exists()

    token.persist("A", "B", DATETIME_MIN)

    assert token.Filepath().exists()
    assert token.access(refresh=False) == "A"

    # The token can't be refreshed: 404
    # The token is dropped
    # The token can't be recreated: timeout
    with raises(TimeoutException):
        login(timeout=0)

    assert not token.Filepath().exists()


def test_login_successively_on_different_uri(monkeypatch, respx_mock):
    def open_webbrowser(url):
        open_webbrowser.url = url
        return True

    monkeypatch.setattr(
        "skore_hub_project.authentication.login.open_webbrowser", open_webbrowser
    )

    for u in ("https://my-1-uri", "https://my-2-uri"):
        respx_mock.post(urljoin(u, REFRESH_URL)).mock(Response(404))
        respx_mock.get(urljoin(u, LOGIN_URL)).mock(
            Response(
                200,
                json={
                    "authorization_url": "<url>",
                    "device_code": "<device>",
                    "user_code": "<user>",
                },
            )
        )
        respx_mock.get(urljoin(u, PROBE_URL)).mock(Response(200))
        respx_mock.post(urljoin(u, CALLBACK_URL)).mock(Response(200))
        respx_mock.get(urljoin(u, TOKEN_URL)).mock(
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

    # Login into a first custom uri using ENVAR
    monkeypatch.setenv(uri.ENVARNAME, "https://my-1-uri")

    login()

    assert uri.URI() == "https://my-1-uri"

    # Login into a second custom uri using ENVAR
    monkeypatch.setenv(uri.ENVARNAME, "https://my-2-uri")

    login()

    assert uri.URI() == "https://my-2-uri"
