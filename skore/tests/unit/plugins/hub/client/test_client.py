from re import escape
from urllib.parse import urljoin

from httpx import (
    HTTPStatusError,
    NetworkError,
    RemoteProtocolError,
    Response,
    TimeoutException,
)
from pytest import mark, raises

from skore._plugins.hub.authentication.registry import local
from skore._plugins.hub.authentication.uri import URI
from skore._plugins.hub.client.client import Client, HUBClient, __semver


class TestClient:
    @mark.respx()
    def test_request_without_retry(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore._plugins.hub.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            Response(408),
        ]

        with raises(TimeoutException), Client(retry=False) as client:
            client.get("http://localhost/foo")

        assert timeouts == []

        with raises(HTTPStatusError) as excinfo, Client(retry=False) as client:
            client.get("http://localhost/foo")

        assert timeouts == []
        assert excinfo.value.response.status_code == 408

    @mark.respx()
    def test_request_with_retry(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore._plugins.hub.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            NetworkError(""),
            RemoteProtocolError(""),
            Response(408),
            Response(425),
            Response(429),
            Response(502),
            Response(503),
            Response(504),
            Response(200),
        ]

        with Client() as client:
            client.get("http://localhost/foo")

        assert timeouts == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

    @mark.respx()
    def test_request_with_retry_and_retry_total(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore._plugins.hub.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            NetworkError(""),
            RemoteProtocolError(""),
            Response(408),
            Response(425),
            Response(429),
            Response(502),
            Response(503),
            Response(504),
            Response(200),
        ]

        with raises(HTTPStatusError) as excinfo, Client(retry_total=5) as client:
            client.get("http://localhost/foo")

        assert timeouts == [0.25, 0.5, 1.0, 2.0, 4.0]
        assert excinfo.value.response.status_code == 429

    @mark.respx()
    def test_request_with_retry_and_unretryable_status(self, monkeypatch, respx_mock):
        timeouts = []

        def sleep(timeout):
            timeouts.append(timeout)

        monkeypatch.setattr("skore._plugins.hub.client.client.sleep", sleep)
        route = respx_mock.get("http://localhost/foo")
        route.side_effect = [
            TimeoutException(""),
            NetworkError(""),
            RemoteProtocolError(""),
            Response(400),
        ]

        with raises(HTTPStatusError) as excinfo, Client() as client:
            client.get("http://localhost/foo")

        assert timeouts == [0.25, 0.5, 1.0]
        assert excinfo.value.response.status_code == 400


def test___semver():
    assert __semver("0.0.0+unknown") is None
    assert __semver("0.1.2") == "0.1.2"
    assert __semver("0.1.2rc10") == "0.1.2-rc.10"


class TestHUBClient:
    @mark.respx()
    def test_request_with_api_key(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI(), "projects/workspace")).mock(Response(200))

        with HUBClient() as client:
            client.request("GET", "workspace")

        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<api-key>"

    @mark.respx()
    def test_request_with_registry_api_key(self, respx_mock):
        local.set(id=1, host=URI(), workspace="workspace", key="<registry-key>")
        respx_mock.get(urljoin(URI(), "projects/workspace")).mock(Response(200))

        with HUBClient() as client:
            client.request("GET", "workspace")

        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<registry-key>"

    @mark.respx()
    def test_request_prefers_environment_over_registry(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<env-key>")
        local.set(id=1, host=URI(), workspace="workspace", key="<registry-key>")
        respx_mock.get(urljoin(URI(), "projects/workspace")).mock(Response(200))

        with HUBClient() as client:
            client.request("GET", "workspace")

        assert respx_mock.calls.last.request.headers["X-API-Key"] == "<env-key>"

    @mark.respx()
    @mark.parametrize(
        ("cli_version", "message"),
        [
            (
                "0.5.0",
                (
                    "No API key found for this workspace; generate one with "
                    "`$ skore hub api-key generate --workspace <workspace>`, "
                    "or set the SKORE_HUB_API_KEY environment variable."
                ),
            ),
            (
                "0.4.1",
                (
                    "No API key found for this workspace; "
                    "install `skore-cli>=0.5` (`pip install 'skore-cli>=0.5'`) and "
                    "generate one with "
                    "`$ skore hub api-key generate --workspace <workspace>`, "
                    "or set the SKORE_HUB_API_KEY environment variable."
                ),
            ),
            (
                None,
                (
                    "No API key found for this workspace; "
                    "install `skore-cli>=0.5` (`pip install 'skore-cli>=0.5'`) and "
                    "generate one with "
                    "`$ skore hub api-key generate --workspace <workspace>`, "
                    "or set the SKORE_HUB_API_KEY environment variable."
                ),
            ),
        ],
    )
    def test_request_without_credentials(self, monkeypatch, cli_version, message):
        from importlib.metadata import PackageNotFoundError

        def fake_version(distribution):
            if distribution == "skore-cli":
                if cli_version is None:
                    raise PackageNotFoundError(distribution)

                return cli_version

            return "1.0.0"

        monkeypatch.setattr("skore._plugins.hub.client.client.version", fake_version)

        with raises(RuntimeError, match=f"^{escape(message)}$"), HUBClient() as client:
            client.request("GET", "workspace")

    @mark.respx()
    def test_request_raises(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI(), "projects/workspace")).mock(Response(404))

        with raises(HTTPStatusError), HUBClient() as client:
            client.request("GET", "workspace")

    @mark.respx()
    def test_request_without_package_semver(self, monkeypatch, respx_mock):
        from importlib.metadata import version

        from skore._plugins.hub.client.client import PACKAGE_SEMVER

        assert version("skore") == "0.0.0+unknown"
        assert PACKAGE_SEMVER is None

        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        respx_mock.get(urljoin(URI(), "projects/workspace")).mock(Response(200))

        with HUBClient() as client:
            client.request("GET", "workspace")

        assert "X-Skore-Client" not in respx_mock.calls.last.request.headers

    @mark.respx()
    def test_request_with_package_semver(self, monkeypatch, respx_mock):
        monkeypatch.setenv("SKORE_HUB_API_KEY", "<api-key>")
        monkeypatch.setattr("skore._plugins.hub.client.client.PACKAGE_SEMVER", "1.0.0")
        respx_mock.get(urljoin(URI(), "projects/workspace")).mock(Response(200))

        with HUBClient() as client:
            client.request("GET", "workspace")

        assert respx_mock.calls.last.request.headers["X-Skore-Client"] == "skore/1.0.0"

    def test_jupyterlite(self, monkeypatch):
        from sys import modules
        from unittest.mock import Mock

        monkeypatch.setattr("skore._plugins.hub.client.client.JUPYTERLITE", True)
        monkeypatch.setitem(modules, "js", Mock())
        monkeypatch.setitem(modules, "pyodide.ffi", Mock())
        monkeypatch.setitem(modules, "pyodide.http.pyxhr", Mock())

        with HUBClient() as client:
            assert client._transport.__class__.__name__ == "JupyterliteTransport"
