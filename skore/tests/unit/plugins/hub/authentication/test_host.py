from httpx import ConnectError, Response
from pytest import mark

from skore._plugins.hub.authentication.host import resolve

API = "https://api.skore.probabl.ai"
FRONTEND = "https://skore.probabl.ai"
DISCOVERY = f"{FRONTEND}/.well-known/skore-hub.json"


def setup_function():
    resolve.cache_clear()


@mark.respx()
def test_resolve_returns_api_url(respx_mock):
    respx_mock.get(DISCOVERY).mock(Response(200, json={"api_url": API}))

    assert resolve(FRONTEND) == API


@mark.respx()
def test_resolve_returns_uri_when_api_url_is_missing(respx_mock):
    respx_mock.get(DISCOVERY).mock(Response(200, json={}))

    assert resolve(FRONTEND) == FRONTEND


@mark.respx()
def test_resolve_returns_uri_when_discovery_is_missing(respx_mock):
    respx_mock.get(DISCOVERY).mock(Response(404))

    assert resolve(FRONTEND) == FRONTEND


@mark.respx()
def test_resolve_returns_uri_when_discovery_fails(respx_mock):
    respx_mock.get(DISCOVERY).mock(side_effect=ConnectError("no route"))

    assert resolve(FRONTEND) == FRONTEND
