from json import dumps
from urllib.parse import urljoin

from httpx import Client, Response
from pytest import fixture, mark, raises
from skore_remote_project import Project


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


class TestProject:
    @fixture(autouse=True)
    def monkeypatch_client(self, monkeypatch):
        monkeypatch.setattr(
            "skore_remote_project.project.project.AuthenticatedClient",
            FakeClient,
        )

    def test_tenant(self):
        assert Project("<tenant>", "<name>").tenant == "<tenant>"

    def test_name(self):
        assert Project("<tenant>", "<name>").name == "<name>"

    @mark.respx(assert_all_called=True)
    def test_run_id(self, monkeypatch, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<id>"})
        )

        assert Project("<tenant>", "<name>").run_id == "<id>"

    def test_put_exception(self):
        with raises(TypeError, match="Key must be a string"):
            Project("<tenant>", "<name>").put(None, "<value>")

        with raises(TypeError, match="Note must be a string"):
            Project("<tenant>", "<name>").put("<key>", "<value>", note=0)

    def test_put(self, monkeypatch, respx_mock, now):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<id>"})
        )
        respx_mock.post("projects/<tenant>/<name>/items").mock(Response(200))

        Project("<tenant>", "<name>").put("<key>", "<value>", note="<note>")

        assert respx_mock.calls.last.request.content == str.encode(
            dumps(
                {
                    "representation": {
                        "media_type": "application/json",
                        "value": "<value>",
                    },
                    "parameters": {
                        "class": "JSONableItem",
                        "parameters": {"value": "<value>"},
                    },
                    "key": "<key>",
                    "run_id": "<id>",
                    "note": "<note>",
                },
                separators=(",", ":"),
            )
        )
