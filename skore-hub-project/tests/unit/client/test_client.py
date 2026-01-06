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

from skore_hub_project.authentication import token, uri
from skore_hub_project.client.client import Client, HUBClient, __semver

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

REFRESH_URL = "identity/oauth/token/refresh"


class TestClient:
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
    def test_request_with_api_key(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))

        with HUBClient() as client:
            client.get("foo")

        assert "authorization" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

    def test_request_with_token(self, respx_mock):
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))

        assert not token.Filepath().exists()

        token.persist("A", "B", DATETIME_MAX)

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"

        with HUBClient() as client:
            client.get("foo")

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"
        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer A"

    def test_request_with_token_and_uri(self, respx_mock):
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))

        assert not token.Filepath().exists()

        token.persist("A", "B", DATETIME_MAX)
        uri.persist(uri.DEFAULT)

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"

        with HUBClient() as client:
            client.get("foo")

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"
        assert uri.URI() == uri.DEFAULT
        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer A"

    def test_request_with_api_key_and_token_and_uri(self, monkeypatch, respx_mock):
        TOKEN_URI = "https://token.com"
        API_KEY_URI = "https://apikey.com"

        assert not uri.Filepath().exists()

        # simulate user with token
        uri.persist(TOKEN_URI)

        assert uri.URI() == TOKEN_URI

        # simulate user with token and API key
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))

        assert uri.URI() == uri.DEFAULT

        with HUBClient() as client:
            client.get("foo")

        assert respx_mock.calls.last.request.url == urljoin(uri.DEFAULT, "foo")
        assert "authorization" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

        # simulate user with token, API key and URI in environment
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        monkeypatch.setenv("SKORE_HUB_URI", API_KEY_URI)
        respx_mock.get(urljoin(API_KEY_URI, "foo")).mock(Response(200))

        assert uri.URI() == API_KEY_URI

        with HUBClient() as client:
            client.get("foo")

        assert respx_mock.calls.last.request.url == urljoin(API_KEY_URI, "foo")
        assert "authorization" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

    @mark.respx(assert_all_mocked=False)
    def test_request_with_invalid_token_raises(self, respx_mock):
        with raises(token.TokenError, match="not logged in"), HUBClient() as client:
            client.get("foo")

        assert not respx_mock.calls

    @mark.respx(assert_all_mocked=False)
    def test_request_with_conflicting_uri_raises(self, respx_mock, monkeypatch):
        token.persist("A", "B", DATETIME_MAX)
        uri.persist(uri.DEFAULT)
        monkeypatch.setenv(uri.ENVARNAME, "https://my-conflicting-uri")

        with (
            raises(uri.URIError, match="the persisted URI is conflicting"),
            HUBClient() as client,
        ):
            client.get("foo")

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"
        assert not respx_mock.calls

    def test_request_with_expired_token(self, tmp_path, respx_mock):
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))
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

        with HUBClient() as client:
            client.get("foo")

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "D"
        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer D"

    def test_request_raises(self, tmp_path, respx_mock):
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(404))

        assert not token.Filepath().exists()

        token.persist("A", "B", DATETIME_MAX)

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"

        with raises(HTTPStatusError), HUBClient() as client:
            client.get("foo")

        assert token.Filepath().exists()
        assert token.access(refresh=False) == "A"

    def test_request_without_package_version(self, respx_mock):
        from importlib.metadata import version

        from skore_hub_project.client.client import PACKAGE_VERSION

        assert version("skore-hub-project") == "0.0.0+unknown"
        assert PACKAGE_VERSION is None

        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))

        with HUBClient(authenticated=False) as client:
            client.get("foo")

        assert "X-Skore-Client" not in respx_mock.calls.last.request.headers

    def test_request_with_package_version(self, monkeypatch, respx_mock):
        monkeypatch.setattr("skore_hub_project.client.client.PACKAGE_VERSION", "1.0.0")
        respx_mock.get(urljoin(uri.DEFAULT, "foo")).mock(Response(200))

        with HUBClient(authenticated=False) as client:
            client.get("foo")

        assert (
            respx_mock.calls.last.request.headers["X-Skore-Client"]
            == "skore-hub-project/1.0.0"
        )
