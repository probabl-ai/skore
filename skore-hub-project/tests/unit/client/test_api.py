from urllib.parse import urljoin

from pytest import mark, raises
from httpx import Response, HTTPError, TimeoutException
from skore_hub_project.client import api
from skore_hub_project.client.api import URI


@mark.parametrize("success_uri", [None, "toto"])
def test_get_oauth_device_login(respx_mock, success_uri):
    respx_mock.get(urljoin(URI, "identity/oauth/device/login")).mock(
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

    params = list()

    if success_uri is None:
        params = {}
    else:
        params = {"success_uri": "toto"}

    assert dict(respx_mock.calls.last.request.url.params) == params
    assert authorization_url == "A"
    assert device_code == "B"
    assert user_code == "C"


def test_post_oauth_device_callback(respx_mock):
    route = respx_mock.post(urljoin(URI, "identity/oauth/device/callback")).mock(
        Response(201, json={})
    )

    api.post_oauth_device_callback("my_state", "my_user_code")

    assert route.called
    assert route.calls.last.request.read() == b"state=my_state&user_code=my_user_code"


def test_get_oauth_device_token(respx_mock):
    respx_mock.get(urljoin(URI, "identity/oauth/device/token")).mock(
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

    access_token, refresh_token, expires_at = api.get_oauth_device_token(
        "<device_code>"
    )

    assert access_token == "A"
    assert refresh_token == "B"
    assert expires_at == "C"
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>"
    }


def test_get_oauth_device_code_probe(monkeypatch, respx_mock):
    monkeypatch.setattr("skore_hub_project.client.api.sleep", lambda _: None)
    respx_mock.get(urljoin(URI, "identity/oauth/device/code-probe")).mock(
        side_effect=[
            Response(400),
            Response(400),
            Response(200),
        ]
    )

    api.get_oauth_device_code_probe("<device_code>")

    assert len(respx_mock.calls) == 3
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>",
    }


def test_get_oauth_device_code_probe_exception(respx_mock):
    respx_mock.get(urljoin(URI, "identity/oauth/device/code-probe")).mock(
        side_effect=[
            Response(404),
            Response(400),
            Response(200),
        ]
    )

    with raises(HTTPError) as excinfo:
        api.get_oauth_device_code_probe("<device_code>")

    assert excinfo.value.response.status_code == 404
    assert len(respx_mock.calls) == 1
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>",
    }


def test_get_oauth_device_code_probe_timeout(respx_mock):
    respx_mock.get(urljoin(URI, "identity/oauth/device/code-probe")).mock(
        side_effect=[
            Response(400),
            Response(400),
            Response(200),
        ]
    )

    with raises(TimeoutException):
        api.get_oauth_device_code_probe("<device_code>", timeout=0)

    assert len(respx_mock.calls) == 1
    assert dict(respx_mock.calls.last.request.url.params) == {
        "device_code": "<device_code>",
    }


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
