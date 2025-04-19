from json import dumps
from urllib.parse import urljoin

from httpx import Client, Response
from pandas import Categorical, DataFrame, RangeIndex, MultiIndex, Index
from pandas.testing import assert_frame_equal
from pytest import fixture, mark, raises
from skore_remote_project import Project
from skore_remote_project.project.metadata import Metadata


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
        monkeypatch.setattr(
            "skore_remote_project.project.metadata.AuthenticatedClient",
            FakeClient,
        )

    def test_tenant(self):
        assert Project("<tenant>", "<name>").tenant == "<tenant>"

    def test_name(self):
        assert Project("<tenant>", "<name>").name == "<name>"

    @mark.respx(assert_all_called=True)
    def test_run_id(self, monkeypatch, nowstr, respx_mock):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<id>"})
        )

        assert Project("<tenant>", "<name>").run_id == "<id>"

    @mark.respx(assert_all_called=True)
    def test_metadata(self, monkeypatch, nowstr, respx_mock):
        url = "projects/<tenant>/<name>/runs"
        respx_mock.post(url).mock(Response(200, json={"id": "<id>"}))

        url = "projects/<tenant>/<name>/experiments/estimator-reports"
        respx_mock.get(url).mock(
            Response(
                200,
                json=[
                    {
                        "id": 0,
                        "run_id": 0,
                        "ml_task": "<ml_task>",
                        "estimator_class_name": "<estimator_class_name>",
                        "dataset_fingerprint": "<dataset_fingerprint>",
                        "created_at": nowstr,
                        "metrics": [],
                    },
                ],
            )
        )

        project = Project("<tenant>", "<name>")
        metadata = project.metadata()

        assert isinstance(metadata, DataFrame)
        assert isinstance(metadata, Metadata)
        assert metadata.project == project

        assert_frame_equal(
            metadata,
            DataFrame(
                {
                    "run_id": [0],
                    "ml_task": ["<ml_task>"],
                    "learner": Categorical(["<estimator_class_name>"]),
                    "dataset": ["<dataset_fingerprint>"],
                    "date": [nowstr],
                },
                MultiIndex.from_arrays(
                    [
                        RangeIndex(1),
                        Index(["0"], name="id", dtype=str),
                    ]
                ),
            ),
        )

    def test_put_exception(self):
        with raises(TypeError, match="Key must be a string"):
            Project("<tenant>", "<name>").put(None, "<value>")

        with raises(TypeError, match="Note must be a string"):
            Project("<tenant>", "<name>").put("<key>", "<value>", note=0)

    @mark.respx(assert_all_called=True)
    def test_put(self, monkeypatch, respx_mock, now):
        respx_mock.post("projects/<tenant>/<name>/runs").mock(
            Response(200, json={"id": "<id>"})
        )
        respx_mock.post("projects/<tenant>/<name>/items/").mock(Response(200))

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
