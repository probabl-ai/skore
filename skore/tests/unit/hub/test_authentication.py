import json
from datetime import datetime
from importlib import reload
from urllib.parse import urljoin

import pytest
from httpx import Response
from skore.hub.authentication import (
    URI,
    AuthenticatedClient,
    AuthenticationError,
    AuthenticationToken,
    login,
)


@pytest.fixture(autouse=True)
def monkeypatch_home(monkeypatch, tmp_path):
    import tempfile

    with monkeypatch.context() as mp:
        mp.setenv("TMPDIR", str(tmp_path))

        # The first call of `tempfile.gettempdir` being cached, reload the module.
        # https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
        reload(tempfile)

        yield


class TestAuthenticationToken:
    def test_init_with_parameters(self):
        token = AuthenticationToken("A", "B", "C")

        assert token.access == "A"
        assert token.refreshment == "B"
        assert token.expires_at == "C"

    def test_init_without_parameters_with_file(self, tmp_path):
        filepath = tmp_path / "skore.token"
        filepath.write_text('["A", "B", "C"]')
        token = AuthenticationToken()

        assert token.access == "A"
        assert token.refreshment == "B"
        assert token.expires_at == "C"

    def test_init_without_parameters_without_file(self):
        token = AuthenticationToken()

        assert token.access is None
        assert token.refreshment is None
        assert token.expires_at is None

    def test_save(self, tmp_path):
        filepath = tmp_path / "skore.token"
        token = AuthenticationToken("A", "B", "C")
        token.save()

        assert json.loads(filepath.read_text()) == ["A", "B", "C"]

    @pytest.mark.respx
    def test_refresh(self, tmp_path, respx_mock):
        filepath = tmp_path / "skore.token"
        refresh_route = respx_mock.post(
            urljoin(URI, "identity/oauth/token/refresh")
        ).mock(
            Response(
                200,
                json={
                    "token": {
                        "access_token": "D",
                        "refresh_token": "E",
                        "expires_at": "F",
                    }
                },
            )
        )

        token = AuthenticationToken("A", "B", "C")
        token.refresh()

        assert refresh_route.called
        assert token.access == "D"
        assert token.refreshment == "E"
        assert token.expires_at == "F"
        assert json.loads(filepath.read_text()) == ["D", "E", "F"]

    def test_is_valid_true(self):
        assert AuthenticationToken("A", "B", "C").is_valid()

    def test_is_valid_false(self):
        assert not AuthenticationToken().is_valid()

    def test_repr(self):
        token = AuthenticationToken("A" * 1000, "B" * 1000, "C" * 1000)

        assert repr(token) == f"AuthenticationToken('{'A' * 10}[...]')"


class TestAuthenticatedClient:
    def test_token_with_file(self, tmp_path):
        filepath = tmp_path / "skore.token"
        filepath.write_text('["A", "B", "C"]')
        token = AuthenticatedClient().token

        assert token.access == "A"
        assert token.refreshment == "B"
        assert token.expires_at == "C"

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
    def test_request_with_expired_token(self, tmp_path, respx_mock):
        filepath = tmp_path / "skore.token"
        filepath.write_text(
            f'["A", "B", "{datetime(2000, 1, 1, 0, 0, 0).isoformat()}"]'
        )

        respx_mock.get(urljoin(URI, "foo")).mock(Response(200))
        respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(
            Response(
                200,
                json={
                    "token": {
                        "access_token": "D",
                        "refresh_token": "E",
                        "expires_at": "F",
                    }
                },
            )
        )

        client = AuthenticatedClient()
        client.get("foo")

        # breakpoint()
        assert client.token.access == "D"
        assert client.token.refreshment == "E"
        assert client.token.expires_at == "F"


@pytest.mark.respx(assert_all_called=True)
def test_login(monkeypatch, respx_mock):
    respx_mock.get(urljoin(URI, "identity/oauth/device/login")).mock(
        Response(
            200,
            json={
                "authorization_url": "https://idp.com",
                "device_code": "device-123",
                "user_code": "123456",
            },
        )
    )
    respx_mock.get(
        urljoin(URI, "identity/oauth/device/token?device_code=device-123")
    ).mock(
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

    called_url = None

    def browser_open(url):
        nonlocal called_url
        called_url = url

    monkeypatch.setattr("webbrowser.open", browser_open)

    token = login()

    assert called_url == "https://idp.com"
    assert token.access == "A"
    assert token.refreshment == "B"
    assert token.expires_at == "C"
