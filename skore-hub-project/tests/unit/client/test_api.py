from urllib.parse import urljoin

import pytest
from httpx import Response
from skore_hub_project.client import api
from skore_hub_project.client.api import URI


@pytest.mark.parametrize("success_uri", [None, "toto"])
@pytest.mark.respx
def test_get_oauth_device_login(respx_mock, success_uri):
    route = respx_mock.get(urljoin(URI, "identity/oauth/device/login")).mock(
        Response(
            200,
            json={
                "authorization_url": "A",
                "device_code": "B",
                "user_code": "C",
            },
        )
    )
    authorization_url, device_code, user_code = api.get_oauth_device_login(
        success_uri=success_uri
    )

    params = list(route.calls.last.request.url.params.items())

    if success_uri is None:
        assert params == []
    else:
        assert params == [("success_uri", "toto")]

    assert authorization_url == "A"
    assert device_code == "B"
    assert user_code == "C"


@pytest.mark.respx
def test_post_oauth_device_callback(respx_mock):
    route = respx_mock.post(urljoin(URI, "identity/oauth/device/callback")).mock(
        Response(201, json={})
    )

    api.post_oauth_device_callback("my_state", "my_user_code")

    assert route.called
    assert route.calls.last.request.read() == b"state=my_state&user_code=my_user_code"


@pytest.mark.respx
def test_get_oauth_device_token(respx_mock):
    route = respx_mock.get(urljoin(URI, "identity/oauth/device/token")).mock(
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

    access_token, refresh_token, expires_at = api.get_oauth_device_token("code")

    params = list(route.calls.last.request.url.params.items())

    assert params == [("device_code", "code")]
    assert access_token == "A"
    assert refresh_token == "B"
    assert expires_at == "C"


@pytest.mark.respx
def test_post_oauth_refresh_token(respx_mock):
    route = respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(
        Response(
            200,
            json={"access_token": "A", "refresh_token": "B", "expires_at": "C"},
        )
    )

    access_token, refresh_token, expires_at = api.post_oauth_refresh_token("token")

    assert route.calls.last.request.read() == b'{"refresh_token":"token"}'
    assert access_token == "A"
    assert refresh_token == "B"
    assert expires_at == "C"
