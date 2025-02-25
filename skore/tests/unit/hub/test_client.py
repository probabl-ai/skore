from datetime import datetime
from urllib.parse import urljoin

import pytest
from httpx import Response
from skore.hub.api import URI
from skore.hub.client import AuthenticatedClient, AuthenticationError

REFRESH_URL = urljoin(URI, "identity/oauth/token/refresh")


class TestAuthenticatedClient:
    def test_token_with_file(self, tmp_path, mock_now, mock_nowstr):
        (tmp_path / "skore.token").write_text(f'["A", "B", "{mock_nowstr}"]')
        token = AuthenticatedClient().token

        assert token.access == "A"
        assert token.refreshment == "B"
        assert token.expires_at == mock_now

    def test_token_without_file(self):
        token = AuthenticatedClient().token

        assert token.access is None
        assert token.refreshment is None
        assert token.expires_at is None

    @pytest.mark.respx(assert_all_called=False)
    def test_request_with_invalid_token_raises(self, respx_mock):
        foo_route = respx_mock.get("foo").mock(Response(200))

        with pytest.raises(AuthenticationError):
            AuthenticatedClient().get("foo")

        assert not foo_route.called

    @pytest.mark.respx(assert_all_called=True)
    def test_request_with_expired_token(
        self, tmp_path, respx_mock, mock_now, mock_nowstr
    ):
        (tmp_path / "skore.token").write_text(
            f'["A", "B", "{datetime(2000, 1, 1, 0, 0, 0).isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        respx_mock.post(REFRESH_URL).mock(
            Response(
                200,
                json={
                    "token": {
                        "access_token": "D",
                        "refresh_token": "E",
                        "expires_at": mock_nowstr,
                    }
                },
            )
        )

        client = AuthenticatedClient()
        client.get("foo")

        assert client.token.access == "D"
        assert client.token.refreshment == "E"
        assert client.token.expires_at == mock_now
