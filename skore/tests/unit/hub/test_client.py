from datetime import datetime
from urllib.parse import urljoin

import pytest
from httpx import Response
from skore.hub.api import URI
from skore.hub.client import AuthenticatedClient, AuthenticationError

REFRESH_URL = urljoin(URI, "identity/oauth/token/refresh")


class TestAuthenticatedClient:
    @pytest.mark.respx(assert_all_called=False)
    def test_request_with_invalid_token_raises(self, respx_mock):
        foo_route = respx_mock.get("foo").mock(Response(200))

        with pytest.raises(AuthenticationError):
            AuthenticatedClient().get("foo")

        assert not foo_route.called

    @pytest.mark.respx(assert_all_called=True)
    def test_request_with_token(self, tmp_path, respx_mock):
        (tmp_path / "skore.token").write_text(
            f'["A", "B", "{datetime.max.isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))

        client = AuthenticatedClient()
        client.get("foo")

        assert client.token.access == "A"
        assert client.token.refreshment == "B"
        assert client.token.expires_at == datetime.max

    @pytest.mark.respx(assert_all_called=True)
    def test_request_with_expired_token(self, tmp_path, respx_mock):
        (tmp_path / "skore.token").write_text(
            f'["A", "B", "{datetime.min.isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        respx_mock.post(REFRESH_URL).mock(
            Response(
                200,
                json={
                    "token": {
                        "access_token": "D",
                        "refresh_token": "E",
                        "expires_at": datetime.max.isoformat(),
                    }
                },
            )
        )

        client = AuthenticatedClient()
        client.get("foo")

        assert client.token.access == "D"
        assert client.token.refreshment == "E"
        assert client.token.expires_at == datetime.max
