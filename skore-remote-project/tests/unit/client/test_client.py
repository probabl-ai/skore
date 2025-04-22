from datetime import datetime, timezone
from urllib.parse import urljoin

import pytest
from httpx import HTTPStatusError, Response
from skore_remote_project.client.api import URI
from skore_remote_project.client.client import AuthenticatedClient, AuthenticationError

REFRESH_URL = urljoin(URI, "identity/oauth/token/refresh")
DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc)
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc)


class TestAuthenticatedClient:
    @pytest.mark.respx(assert_all_called=False)
    def test_request_with_invalid_token_raises(self, respx_mock):
        foo_route = respx_mock.get("foo").mock(Response(200))

        with (
            pytest.raises(AuthenticationError, match="not logged in"),
            AuthenticatedClient() as client,
        ):
            client.get("foo")

        assert not foo_route.called

    @pytest.mark.respx(assert_all_called=True)
    def test_request_with_token(self, tmp_path, respx_mock):
        (tmp_path / "skore.token").write_text(
            f'["A", "B", "{DATETIME_MAX.isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))

        with AuthenticatedClient() as client:
            client.get("foo")

            assert client.token.access == "A"
            assert client.token.refreshment == "B"
            assert client.token.expires_at == DATETIME_MAX

    @pytest.mark.respx(assert_all_called=True)
    def test_request_with_expired_token(self, tmp_path, respx_mock):
        (tmp_path / "skore.token").write_text(
            f'["A", "B", "{DATETIME_MIN.isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        respx_mock.post(REFRESH_URL).mock(
            Response(
                200,
                json={
                    "access_token": "D",
                    "refresh_token": "E",
                    "expires_at": DATETIME_MAX.isoformat(),
                },
            )
        )

        with AuthenticatedClient() as client:
            client.get("foo")

            assert client.token.access == "D"
            assert client.token.refreshment == "E"
            assert client.token.expires_at == DATETIME_MAX

    @pytest.mark.respx(assert_all_called=True)
    def test_request_raises(self, tmp_path, respx_mock):
        (tmp_path / "skore.token").write_text(
            f'["A", "B", "{DATETIME_MAX.isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(404))

        with pytest.raises(HTTPStatusError), AuthenticatedClient(raises=True) as client:
            client.get("foo")
