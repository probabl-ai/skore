from datetime import UTC, datetime, timedelta
from urllib.parse import urljoin

from httpx import HTTPError, Response, TimeoutException
from pytest import fixture, mark, raises

from skore._plugins.hub.authentication import store
from skore._plugins.hub.authentication.token import (
    Token,
    _token_expired,
    fresh_token,
    get_oauth_device_code_probe,
    get_oauth_device_login,
    get_oauth_device_token,
    interactive_device_login,
    post_oauth_device_callback,
    post_oauth_logout,
    post_oauth_refresh_token,
)
from skore._plugins.hub.authentication.uri import URI

DATETIME_MIN = datetime.min.replace(tzinfo=UTC).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=UTC).isoformat()

REFRESH_URL = "identity/oauth/token/refresh"
LOGIN_URL = "identity/oauth/device/login"
PROBE_URL = "identity/oauth/device/code-probe"
CALLBACK_URL = "identity/oauth/device/callback"
TOKEN_URL = "identity/oauth/device/token"
LOGOUT_URL = "identity/oauth/logout"


@mark.parametrize("success_uri", [None, "toto"])
@mark.respx()
def test_get_oauth_device_login(respx_mock, success_uri):
    respx_mock.get(urljoin(URI(), LOGIN_URL)).mock(
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
    route = respx_mock.post(urljoin(URI(), CALLBACK_URL)).mock(Response(201, json={}))

    post_oauth_device_callback("my_state", "my_user_code")

    assert route.called
    assert route.calls.last.request.read() == b"state=my_state&user_code=my_user_code"


@mark.respx()
def test_get_oauth_device_token(respx_mock):
    respx_mock.get(urljoin(URI(), TOKEN_URL)).mock(
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
    monkeypatch.setattr("skore._plugins.hub.authentication.token.sleep", lambda _: None)
    respx_mock.get(urljoin(URI(), PROBE_URL)).mock(
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
    respx_mock.get(urljoin(URI(), PROBE_URL)).mock(
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
    respx_mock.get(urljoin(URI(), PROBE_URL)).mock(
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
    route = respx_mock.post(urljoin(URI(), REFRESH_URL)).mock(
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


def _mock_device_flow(respx_mock, *, expires_at=DATETIME_MAX):
    """Mock the full device-login flow (login -> probe -> callback -> token)."""
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
                    "expires_at": expires_at,
                }
            },
        )
    )


@mark.respx()
def test_interactive_device_login(monkeypatch, respx_mock):
    opened = {}

    def open_webbrowser(url):
        opened["url"] = url
        return True

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.token.open_webbrowser", open_webbrowser
    )
    _mock_device_flow(respx_mock)

    access_token, refresh_token, expires_at = interactive_device_login()

    assert access_token == "D"
    assert refresh_token == "E"
    assert expires_at == DATETIME_MAX
    assert opened["url"] == "<url>"


@mark.respx()
def test_interactive_device_login_without_browser(monkeypatch, respx_mock):
    opened = {}

    def open_webbrowser(url):
        opened["url"] = url
        return True

    monkeypatch.setattr(
        "skore._plugins.hub.authentication.token.open_webbrowser", open_webbrowser
    )
    _mock_device_flow(respx_mock)

    access_token, refresh_token, expires_at = interactive_device_login(
        open_browser=False
    )

    assert access_token == "D"
    assert "url" not in opened


@mark.respx()
def test_post_oauth_logout_with_refresh_token(respx_mock):
    route = respx_mock.post(urljoin(URI(), LOGOUT_URL)).mock(Response(200))

    post_oauth_logout("<access>", "<refresh>")

    assert route.called
    cookie = route.calls.last.request.headers["cookie"]
    assert "access_token=<access>" in cookie
    assert "refresh_token=<refresh>" in cookie


@mark.respx()
def test_post_oauth_logout_without_refresh_token(respx_mock):
    route = respx_mock.post(urljoin(URI(), LOGOUT_URL)).mock(Response(200))

    post_oauth_logout("<access>")

    assert route.called
    cookie = route.calls.last.request.headers["cookie"]
    assert "access_token=<access>" in cookie
    assert "refresh_token" not in cookie


def test_token_expired_none():
    assert _token_expired(None) is True


def test_token_expired_unparsable():
    assert _token_expired("not-a-datetime") is True


def test_token_expired_past():
    assert _token_expired(DATETIME_MIN) is True


def test_token_expired_future():
    assert _token_expired(DATETIME_MAX) is False


def test_token_expired_naive_treated_as_utc():
    # A far-future naive timestamp is not expired once interpreted as UTC.
    naive = datetime.max.replace(microsecond=0).isoformat()
    assert _token_expired(naive) is False


def test_token_expired_within_margin():
    soon = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()
    # Default 60s margin: a token expiring in 30s is treated as expired...
    assert _token_expired(soon) is True
    # ...but with no margin it is still valid.
    assert _token_expired(soon, margin_seconds=0) is False


@fixture
def credentials(monkeypatch, tmp_path):
    """Point the persisted token cache at an isolated file in ``tmp_path``."""
    file = tmp_path / "hub.json"
    monkeypatch.setenv("SKORE_HUB_CREDENTIALS", str(file))
    return file


@mark.respx()
def test_fresh_token_no_stored_token(credentials, respx_mock):
    assert fresh_token() is None
    assert len(respx_mock.calls) == 0


@mark.respx()
def test_fresh_token_missing_access_token(credentials, respx_mock):
    store.save({"refresh_token": "B", "expires_at": DATETIME_MAX})

    assert fresh_token() is None
    assert len(respx_mock.calls) == 0


@mark.respx()
def test_fresh_token_valid_returned_unchanged(credentials, respx_mock):
    token = {
        "uri": "https://hub",
        "access_token": "A",
        "refresh_token": "B",
        "expires_at": DATETIME_MAX,
    }
    store.save(token)

    assert fresh_token() == token
    assert len(respx_mock.calls) == 0


@mark.respx()
def test_fresh_token_refreshes_when_expired(credentials, respx_mock):
    store.save(
        {
            "uri": "https://hub",
            "access_token": "A",
            "refresh_token": "B",
            "expires_at": DATETIME_MIN,
        }
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

    token = fresh_token()

    assert token["access_token"] == "F"
    assert token["refresh_token"] == "G"
    assert token["expires_at"] == DATETIME_MAX
    # The refreshed token is persisted for the next process.
    assert store.load() == token


@mark.respx()
def test_fresh_token_relogin_when_refresh_fails(monkeypatch, credentials, respx_mock):
    monkeypatch.setattr(
        "skore._plugins.hub.authentication.token.open_webbrowser",
        lambda _: True,
    )
    store.save(
        {
            "uri": "https://hub",
            "access_token": "A",
            "refresh_token": "B",
            "expires_at": DATETIME_MIN,
        }
    )
    respx_mock.post(REFRESH_URL).mock(Response(400))
    _mock_device_flow(respx_mock)

    token = fresh_token()

    assert token["access_token"] == "D"
    assert token["refresh_token"] == "E"
    assert token["expires_at"] == DATETIME_MAX
    assert store.load() == token


@mark.respx()
def test_fresh_token_no_refresh_no_relogin_returns_none(credentials, respx_mock):
    store.save(
        {
            "uri": "https://hub",
            "access_token": "A",
            "expires_at": DATETIME_MIN,
        }
    )

    assert fresh_token(relogin=False) is None
    assert len(respx_mock.calls) == 0


@mark.respx()
def test_fresh_token_refresh_fails_no_relogin_returns_none(credentials, respx_mock):
    store.save(
        {
            "uri": "https://hub",
            "access_token": "A",
            "refresh_token": "B",
            "expires_at": DATETIME_MIN,
        }
    )
    respx_mock.post(REFRESH_URL).mock(Response(400))

    assert fresh_token(relogin=False) is None


class TestToken:
    @mark.respx()
    def test_init(self, monkeypatch, respx_mock):
        def open_webbrowser(url):
            open_webbrowser.url = url
            return True

        monkeypatch.setattr(
            "skore._plugins.hub.authentication.token.open_webbrowser", open_webbrowser
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
            "skore._plugins.hub.authentication.token.open_webbrowser",
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
            "skore._plugins.hub.authentication.token.open_webbrowser",
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
            "skore._plugins.hub.authentication.token.open_webbrowser",
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
