import json
from datetime import datetime
from urllib.parse import urljoin

import pytest
from httpx import Response
from skore.hub.api import URI
from skore.hub.token import AuthenticationToken


class TestAuthenticationToken:
    def test_init_with_parameters(self, mock_now, mock_nowstr):
        token = AuthenticationToken("A", "B", mock_nowstr)

        assert token.access == "A"
        assert token.refreshment == "B"
        assert token.expires_at == mock_now

    def test_init_without_parameters_with_file(self, tmp_path, mock_now, mock_nowstr):
        (tmp_path / "skore.token").write_text(f'["A", "B", "{mock_nowstr}"]')
        token = AuthenticationToken()

        assert token.access == "A"
        assert token.refreshment == "B"
        assert token.expires_at == mock_now

    def test_init_without_parameters_without_file(self):
        token = AuthenticationToken()

        assert token.access is None
        assert token.refreshment is None
        assert token.expires_at is None

    def test_save(self, tmp_path, mock_nowstr):
        token = AuthenticationToken("A", "B", mock_nowstr)
        token.save()

        assert json.loads((tmp_path / "skore.token").read_text()) == [
            "A",
            "B",
            mock_nowstr,
        ]

    @pytest.mark.respx
    def test_refresh(self, tmp_path, respx_mock, mock_now, mock_nowstr):
        respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(
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

        token = AuthenticationToken("A", "B", datetime(2000, 1, 1).isoformat())
        token.refresh()

        assert token.access == "D"
        assert token.refreshment == "E"
        assert token.expires_at == mock_now
        assert json.loads((tmp_path / "skore.token").read_text()) == [
            "D",
            "E",
            mock_nowstr,
        ]

    def test_is_valid_true(self, mock_nowstr):
        assert AuthenticationToken("A", "B", mock_nowstr).is_valid()

    def test_is_valid_false(self):
        assert not AuthenticationToken().is_valid()

    def test_repr(self, mock_nowstr):
        token = AuthenticationToken("A" * 1000, "B" * 1000, mock_nowstr)

        assert repr(token) == f"AuthenticationToken('{'A' * 10}[...]')"
