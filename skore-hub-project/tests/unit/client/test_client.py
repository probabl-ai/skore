from datetime import datetime, timezone
from urllib.parse import urljoin

from pytest import mark, raises
from httpx import HTTPStatusError, Response
from skore_hub_project.client.api import URI
from skore_hub_project.client.client import AuthenticatedClient, AuthenticationError
from skore_hub_project.authentication.token import Token

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()

REFRESH_URL = urljoin(URI, "identity/oauth/token/refresh")


class TestAuthenticatedClient:
    def test_request_with_api_key(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))

        with AuthenticatedClient() as client:
            client.get("foo")

        assert "authorization" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

    def test_request_with_token(self, tmp_path, respx_mock):
        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))

        assert not Token.exists()

        Token.save("A", "B", DATETIME_MAX)

        assert Token.exists()
        assert Token.access() == "A"

        with AuthenticatedClient() as client:
            client.get("foo")

        assert Token.exists()
        assert Token.access() == "A"
        assert "X-API-Key" not in respx_mock.calls.last.request.headers
        assert respx_mock.calls.last.request.headers["authorization"] == "Bearer A"

    @mark.respx(assert_all_mocked=False)
    def test_request_with_invalid_token_raises(self, respx_mock):
        with (
            raises(AuthenticationError, match="not logged in"),
            AuthenticatedClient() as client,
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

        with AuthenticatedClient() as client:
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

        with raises(HTTPStatusError), AuthenticatedClient(raises=True) as client:
            client.get("foo")

        assert Token.exists()
        assert Token.access() == "A"
