from datetime import datetime, timezone
from json import dumps
from urllib.parse import urljoin

from httpx import Client, Response
from pytest import mark, raises
from skore.hub import Project


class FakeClient(Client):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def request(self, method, url, **kwargs):
        response = super().request(method, urljoin("http://localhost", url), **kwargs)
        response.raise_for_status()

        return response


class TestProject:
    @mark.respx(assert_all_called=True)
    def test_id_existing_project(self, monkeypatch, respx_mock):
        monkeypatch.setattr("skore.hub.project.project.AuthenticatedClient", FakeClient)
        respx_mock.get("projects/?tenant_id=0").mock(
            Response(
                200,
                json=[{"name": "<name>", "id": "<id>"}],
            )
        )

        assert Project("<name>", 0).id == "<id>"

    @mark.respx(assert_all_called=True)
    def test_id_new_project(self, monkeypatch, respx_mock):
        monkeypatch.setattr("skore.hub.project.project.AuthenticatedClient", FakeClient)
        respx_mock.get("projects/?tenant_id=0").mock(Response(200, json=[]))
        respx_mock.post("projects/").mock(Response(200, json={"project_id": "<id>"}))

        assert Project("<name>", 0).id == "<id>"
        assert respx_mock.calls.last.request.content == str.encode(
            dumps({"name": "<name>", "tenant_id": 0}, separators=(",", ":"))
        )

    def test_put_exception(self):
        with raises(TypeError, match="Key must be a string"):
            Project("<name>", 0).put(None, "<value>")

        with raises(TypeError, match="Note must be a string"):
            Project("<name>", 0).put("<key>", "<value>", note=0)

    def test_put(self, monkeypatch, respx_mock, MockDatetime, mock_now):
        monkeypatch.setattr("skore.hub.project.project.datetime", MockDatetime)
        monkeypatch.setattr("skore.hub.project.project.AuthenticatedClient", FakeClient)
        respx_mock.get("projects/?tenant_id=0").mock(Response(200, json=[]))
        respx_mock.post("projects/").mock(Response(200, json={"project_id": "<id>"}))
        respx_mock.post("projects/<id>/items/").mock(Response(200))

        Project("<name>", 0).put("<key>", "<value>", note="<note>")

        assert respx_mock.calls.last.request.content == str.encode(
            dumps(
                {
                    "key": "<key>",
                    "created_at": mock_now.replace(tzinfo=None).isoformat(),
                    "updated_at": mock_now.replace(tzinfo=None).isoformat(),
                    "value_type": "JSONableItem",
                    "value": {
                        "note": "<note>",
                        "parameters": {"value": "<value>"},
                        "representation": {
                            "media_type": "application/json",
                            "value": "<value>",
                            "raised": False,
                            "traceback": None,
                            "schema": 1,
                        },
                    },
                },
                separators=(",", ":"),
            )
        )

    def test_get(self, monkeypatch, respx_mock):
        monkeypatch.setattr("skore.hub.project.project.AuthenticatedClient", FakeClient)
        respx_mock.get("projects/?tenant_id=0").mock(Response(200, json=[]))
        respx_mock.post("projects/").mock(Response(200, json={"project_id": "<id>"}))
        respx_mock.get("projects/<id>/items/<key>").mock(
            Response(
                200,
                json={
                    "created_at": datetime.max.isoformat(),
                    "value_type": "JSONableItem",
                    "value": {
                        "note": "<new-note>",
                        "parameters": {"value": "<new-value>"},
                    },
                },
            )
        )
        respx_mock.get("projects/<id>/items/<key>/history").mock(
            Response(
                200,
                json=[
                    {
                        "created_at": datetime.min.isoformat(),
                        "value_type": "JSONableItem",
                        "value": {
                            "note": "<old-note>",
                            "parameters": {"value": "<old-value>"},
                        },
                    },
                    {
                        "created_at": datetime.max.isoformat(),
                        "value_type": "JSONableItem",
                        "value": {
                            "note": "<new-note>",
                            "parameters": {"value": "<new-value>"},
                        },
                    },
                ],
            )
        )

        project = Project("<name>", 0)

        assert project.get("<key>") == "<new-value>"
        assert project.get("<key>", version=0) == "<old-value>"
        assert project.get("<key>", version="all") == ["<old-value>", "<new-value>"]
        assert project.get("<key>", metadata=True) == {
            "value": "<new-value>",
            "date": datetime.max.replace(tzinfo=timezone.utc),
            "note": "<new-note>",
        }
        assert project.get("<key>", version=0, metadata=True) == {
            "value": "<old-value>",
            "date": datetime.min.replace(tzinfo=timezone.utc),
            "note": "<old-note>",
        }
        assert project.get("<key>", version="all", metadata=True) == [
            {
                "value": "<old-value>",
                "date": datetime.min.replace(tzinfo=timezone.utc),
                "note": "<old-note>",
            },
            {
                "value": "<new-value>",
                "date": datetime.max.replace(tzinfo=timezone.utc),
                "note": "<new-note>",
            },
        ]

        with raises(ValueError, match="`version` should be"):
            project.get("<key>", version=-2)
