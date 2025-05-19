from datetime import datetime, timedelta, timezone
from threading import Thread
from urllib.parse import urljoin

import httpx
import pytest
from httpx import Response
from skore_remote_project.authentication.login import login
from skore_remote_project.client.api import URI
from skore_remote_project.client.client import AuthenticationError

CALLBACK_URL = urljoin(URI, "identity/oauth/device/callback")
LOGIN_URL = urljoin(URI, "identity/oauth/device/login")
TOKEN_URL = urljoin(URI, "identity/oauth/device/token")


@pytest.mark.respx(assert_all_called=True)
def test_manual_login(monkeypatch, respx_mock, now, nowstr):
    def webbrowser_open(url):
        webbrowser_open.url = url

    monkeypatch.setattr("webbrowser.open", webbrowser_open)

    respx_mock.get(LOGIN_URL).mock(
        Response(
            200,
            json={
                "authorization_url": "url",
                "device_code": "device",
                "user_code": "user",
            },
        )
    )
    respx_mock.get(TOKEN_URL).mock(
        Response(
            200,
            json={
                "token": {
                    "access_token": "A",
                    "refresh_token": "B",
                    "expires_at": nowstr,
                }
            },
        )
    )

    token = login(auto_otp=False)

    assert webbrowser_open.url == "url"
    assert token.access_token == "A"
    assert token.refresh_token == "B"
    assert token.expires_at == now


@pytest.mark.respx(assert_all_called=True)
def test_manual_login_timeout(monkeypatch, respx_mock):
    monkeypatch.setattr("webbrowser.open", lambda _: None)

    respx_mock.get(TOKEN_URL).mock(Response(500))
    respx_mock.get(LOGIN_URL).mock(
        Response(
            200,
            json={
                "authorization_url": "https://idp.com",
                "device_code": "device-123",
                "user_code": "123",
            },
        )
    )

    with pytest.raises(AuthenticationError, match="Timeout"):
        login(timeout=0, auto_otp=False)


@pytest.mark.respx(assert_all_called=True)
def test_auto_otp_login(monkeypatch, respx_mock, now, nowstr):
    def webbrowser_open(url):
        webbrowser_open.url = url

    monkeypatch.setattr("webbrowser.open", webbrowser_open)
    respx_mock.route(host="localhost").pass_through()

    login_route = respx_mock.get(LOGIN_URL).mock(
        Response(
            200,
            json={
                "authorization_url": "url",
                "device_code": "device",
                "user_code": "user",
            },
        )
    )
    respx_mock.post(CALLBACK_URL).mock(Response(201))
    respx_mock.get(TOKEN_URL).mock(
        Response(
            200,
            json={
                "token": {
                    "access_token": "A",
                    "refresh_token": "B",
                    "expires_at": nowstr,
                }
            },
        )
    )

    def call_success():
        with httpx.Client() as client:
            while not login_route.called:
                pass

            client.get("http://localhost:65535")

    call_success_thread = Thread(target=call_success)
    call_success_thread.start()

    token = login(auto_otp=True, port=65535)

    assert webbrowser_open.url == "url"
    assert token.access_token == "A"
    assert token.refresh_token == "B"
    assert token.expires_at == now


@pytest.mark.respx(assert_all_called=True)
def test_auto_otp_login_timeout(monkeypatch, respx_mock):
    monkeypatch.setattr("webbrowser.open", lambda _: None)

    respx_mock.get(LOGIN_URL).mock(
        Response(
            200,
            json={
                "authorization_url": "url",
                "device_code": "device",
                "user_code": "user",
            },
        )
    )

    with pytest.raises(AuthenticationError, match="Timeout"):
        login(timeout=0, auto_otp=True)


@pytest.mark.respx()
def test_login_with_refresh(tmp_path, respx_mock):
    first = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()
    second = datetime(2000, 1, 2, tzinfo=timezone.utc)

    (tmp_path / "skore.token").write_text(f'["A", "B", "{first}"]')
    respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(
        Response(
            200,
            json={
                "access_token": "D",
                "refresh_token": "E",
                "expires_at": second.isoformat(),
            },
        )
    )

    token = login(auto_otp=True)

    assert token.access_token == "D"
    assert token.refresh_token == "E"
    assert token.expires_at == second


@pytest.mark.respx()
def test_login_with_falling_refresh(monkeypatch, respx_mock, tmp_path):
    """Attempt to refresh an expired token and fail.

    Then check that login process is launched."""
    old = datetime(2000, 1, 1, tzinfo=timezone.utc).isoformat()

    (tmp_path / "skore.token").write_text(f'["A", "B", "{old}"]')

    def patched_webserver_open(url):
        patched_webserver_open.url = url

    monkeypatch.setattr("webbrowser.open", patched_webserver_open)

    respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(Response(500))
    respx_mock.get(LOGIN_URL).mock(
        Response(
            200,
            json={
                "authorization_url": "url",
                "device_code": "device",
                "user_code": "user",
            },
        )
    )

    with pytest.raises(AuthenticationError, match="Timeout"):
        login(timeout=0, auto_otp=True)
    assert patched_webserver_open.url == "url"


def test_login_with_existing_token(now, tmp_path):
    tomorrow = now + timedelta(days=1)
    (tmp_path / "skore.token").write_text(f'["A", "B", "{tomorrow}"]')

    token = login(auto_otp=True)

    assert token.access_token == "A"
    assert token.refresh_token == "B"
    assert token.expires_at == tomorrow
