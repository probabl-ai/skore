import time
from threading import Thread
from urllib.parse import urljoin

import httpx
import pytest
from httpx import Response
from skore.hub.api import URI
from skore.hub.client import AuthenticationError
from skore.hub.login import login

CALLBACK_URL = urljoin(URI, "identity/oauth/device/callback")
LOGIN_URL = urljoin(URI, "identity/oauth/device/login")
TOKEN_URL = urljoin(URI, "identity/oauth/device/token")


@pytest.mark.respx(assert_all_called=True)
def test_manual_login(monkeypatch, respx_mock, mock_now, mock_nowstr):
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
                    "expires_at": mock_nowstr,
                }
            },
        )
    )

    token = login(auto_otp=False)

    assert webbrowser_open.url == "url"
    assert token.access == "A"
    assert token.refreshment == "B"
    assert token.expires_at == mock_now


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
def test_auto_otp_login(monkeypatch, respx_mock, mock_now, mock_nowstr):
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
                    "expires_at": mock_nowstr,
                }
            },
        )
    )

    port = -1

    def mock_create_server(callback):
        from skore.hub.callback_server import launch_callback_server

        nonlocal port
        port = launch_callback_server(callback)
        return port

    monkeypatch.setattr("skore.hub.login.launch_callback_server", mock_create_server)

    def call_success():
        with httpx.Client() as client:
            while not login_route.called:
                time.sleep(0.5)

            client.get(f"http://localhost:{port}")

    success_thread = Thread(target=call_success)
    success_thread.start()

    token = login(auto_otp=True)

    assert webbrowser_open.url == "url"
    assert token.access == "A"
    assert token.refreshment == "B"
    assert token.expires_at == mock_now
