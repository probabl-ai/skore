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
from skore_hub_project.authentication import token as Token
from skore_hub_project.client.client import AuthenticationError, Client, HUBClient

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

URI = HUBClient.URI
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


class TestHUBClient:
    def test_request_with_api_key(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))

        with HUBClient() as client:
            client.get("foo")

        assert "authorization" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

    def test_request_with_token(self, tmp_path, respx_mock):
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))

        assert not Token.exists()

        Token.save("A", "B", DATETIME_MAX)

        assert Token.exists()
        assert Token.access() == "A"

        with HUBClient() as client:
            client.get("foo")

        assert Token.exists()
        assert Token.access() == "A"
        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer A"

    @mark.respx(assert_all_mocked=False)
    def test_request_with_invalid_token_raises(self, respx_mock):
        with (
            raises(AuthenticationError, match="not logged in"),
            HUBClient() as client,
        ):
            client.get("foo")

        assert not respx_mock.calls

    def test_request_with_expired_token(self, tmp_path, respx_mock):
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
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

        with HUBClient() as client:
            client.get("foo")

        assert Token.exists()
        assert Token.access() == "D"
        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer D"

    def test_request_raises(self, tmp_path, respx_mock):
        respx_mock.get(urljoin(URI, "foo")).mock(Response(404))

        assert not Token.exists()

        Token.save("A", "B", DATETIME_MAX)

        assert Token.exists()
        assert Token.access() == "A"

        with raises(HTTPStatusError), HUBClient() as client:
            client.get("foo")

        assert Token.exists()
        assert Token.access() == "A"
