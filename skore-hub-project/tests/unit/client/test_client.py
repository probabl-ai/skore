from datetime import datetime, timezone
from urllib.parse import urljoin

from httpx import (
    HTTPStatusError,
    NetworkError,
    RemoteProtocolError,
    Response,
    TimeoutException,
)
from pytest import mark, raises

from skore_hub_project.authentication.login import login
from skore_hub_project.authentication.uri import URI
from skore_hub_project.client.client import Client, HUBClient, __semver

DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

LOGIN_URL = "identity/oauth/device/login"
PROBE_URL = "identity/oauth/device/code-probe"
CALLBACK_URL = "identity/oauth/device/callback"
TOKEN_URL = "identity/oauth/device/token"


class TestClient:
    @mark.respx()
    def test_request_without_retry(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore_hub_project.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            Response(408),
        ]

        with raises(TimeoutException), Client(retry=False) as client:
            client.get("http://localhost/foo")

        assert timeouts == []

        with raises(HTTPStatusError) as excinfo, Client(retry=False) as client:
            client.get("http://localhost/foo")

        assert timeouts == []
        assert excinfo.value.response.status_code == 408

    @mark.respx()
    def test_request_with_retry(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore_hub_project.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            NetworkError(""),
            RemoteProtocolError(""),
            Response(408),
            Response(425),
            Response(429),
            Response(502),
            Response(503),
            Response(504),
            Response(200),
        ]

        with Client() as client:
            client.get("http://localhost/foo")

        assert timeouts == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

    @mark.respx()
    def test_request_with_retry_and_retry_total(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore_hub_project.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            NetworkError(""),
            RemoteProtocolError(""),
            Response(408),
            Response(425),
            Response(429),
            Response(502),
            Response(503),
            Response(504),
            Response(200),
        ]

        with raises(HTTPStatusError) as excinfo, Client(retry_total=5) as client:
            client.get("http://localhost/foo")

        assert timeouts == [0.25, 0.5, 1.0, 2.0, 4.0]
        assert excinfo.value.response.status_code == 429

    @mark.respx()
    def test_request_with_retry_and_unretryable_status(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore_hub_project.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            NetworkError(""),
            RemoteProtocolError(""),
            Response(400),
        ]

        with raises(HTTPStatusError) as excinfo, Client() as client:
            client.get("http://localhost/foo")

        assert timeouts == [0.25, 0.5, 1.0]
        assert excinfo.value.response.status_code == 400


def test___semver():
    assert __semver("0.0.0+unknown") is None
    assert __semver("0.1.2") == "0.1.2"
    assert __semver("0.1.2rc10") == "0.1.2-rc.10"


class TestHUBClient:
    @mark.respx()
    def test_request_with_api_key(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        login()

        with HUBClient() as client:
            client.get("foo")

        assert "authorization" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

    @mark.respx()
    def test_request_with_token(self, monkeypatch, respx_mock):
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
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        login()

        with HUBClient() as client:
            client.get("foo")

        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer D"

    @mark.respx()
    def test_request_without_credentials(self):
        with raises(RuntimeError, match="not logged in"), HUBClient() as client:
            client.get("foo")

    @mark.respx()
    def test_request_raises(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI, "foo")).mock(Response(404))
        login()

        with raises(HTTPStatusError), HUBClient() as client:
            client.get("foo")

    @mark.respx()
    def test_request_without_package_semver(self, monkeypatch, respx_mock):
        from importlib.metadata import version

        from skore_hub_project.client.client import PACKAGE_SEMVER

        assert version("skore-hub-project") == "0.0.0+unknown"
        assert PACKAGE_SEMVER is None

        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        login()

        with HUBClient() as client:
            client.get("foo")

        assert "X-Skore-Client" not in respx_mock.calls.last.request.headers

    @mark.respx()
    def test_request_with_package_semver(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        monkeypatch.setattr("skore_hub_project.client.client.PACKAGE_SEMVER", "1.0.0")
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        login()

        with HUBClient() as client:
            client.get("foo")

        assert (
            respx_mock.calls.last.request.headers["X-Skore-Client"]
            == "skore-hub-project/1.0.0"
        )
