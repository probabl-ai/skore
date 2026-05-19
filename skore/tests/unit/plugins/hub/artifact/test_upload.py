from typing import Literal

from httpx import Response
from pytest import fixture, mark

from skore._plugins.hub.artifact.artifact import Artifact
from skore._plugins.hub.artifact.plan import ArtifactPlan
from skore._plugins.hub.artifact.upload import (
    complete_uploads,
    prepare_artifacts,
    request_upload_urls,
    upload_chunks,
)


@fixture
def two_plans():
    return [
        ArtifactPlan(
            checksum="blake2b-aaa", size=3, content_type="text/plain", payload=b"foo"
        ),
        ArtifactPlan(
            checksum="blake2b-bbb",
            size=3,
            content_type="application/json",
            payload=b"bar",
        ),
    ]


class FakeBytesArtifact(Artifact):
    content_type: Literal["text/plain"] = "text/plain"
    payload: bytes = b"hello"

    def content_to_upload(self) -> bytes:
        return self.payload


class FakeNoneArtifact(Artifact):
    content_type: Literal["text/plain"] = "text/plain"

    def content_to_upload(self) -> None:
        return None


@mark.usefixtures(
    "monkeypatch_tmpdir",
    "monkeypatch_project_hub_client",
    "monkeypatch_project_routes",
)
class TestPrepareArtifacts:
    def test_preserves_artifact_to_plan_alignment(self, project):
        """Position-aligned with input. ``None`` for artifacts without content."""
        artifacts = [
            FakeBytesArtifact(project=project, payload=b"one"),
            FakeNoneArtifact(project=project),
            FakeBytesArtifact(project=project, payload=b"two"),
        ]
        plans = prepare_artifacts(artifacts)

        assert len(plans) == 3
        assert plans[0].payload == b"one"
        assert plans[1] is None
        assert plans[2].payload == b"two"

    def test_empty_list_returns_empty(self):
        assert prepare_artifacts([]) == []


@mark.usefixtures(
    "monkeypatch_artifact_hub_client",
    "monkeypatch_project_hub_client",
    "monkeypatch_project_routes",
)
class TestRequestUploadUrls:
    @mark.respx()
    def test_one_call_per_batch(self, respx_mock, project, two_plans):
        route = respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(
                201,
                json=[
                    {"checksum": "blake2b-aaa", "upload_url": "http://o/a"},
                    {"checksum": "blake2b-bbb", "upload_url": "http://o/b"},
                ],
            )
        )

        from skore._plugins.hub.artifact.upload import HUBClient

        with HUBClient() as client:
            urls = request_upload_urls(client, project, two_plans)

        # Exactly one network call to /artifacts.
        assert route.called
        assert route.call_count == 1

        # Body contained all checksums in one payload.
        sent = route.calls.last.request.read()
        assert b"blake2b-aaa" in sent
        assert b"blake2b-bbb" in sent

        assert set(urls.keys()) == {"blake2b-aaa", "blake2b-bbb"}
        assert urls["blake2b-aaa"][0]["upload_url"] == "http://o/a"

    @mark.respx()
    def test_already_uploaded_checksum_absent_from_response(
        self, respx_mock, project, two_plans
    ):
        respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(
                201,
                json=[{"checksum": "blake2b-aaa", "upload_url": "http://o/a"}],
                # "blake2b-bbb" omitted: backend says it's already there.
            )
        )

        from skore._plugins.hub.artifact.upload import HUBClient

        with HUBClient() as client:
            urls = request_upload_urls(client, project, two_plans)

        assert "blake2b-aaa" in urls
        assert "blake2b-bbb" not in urls


class TestUploadChunks:
    @mark.respx()
    def test_parallel_put_across_artifacts(self, respx_mock, two_plans):
        a_route = respx_mock.put("http://o/a").mock(
            return_value=Response(200, headers={"etag": '"<a-etag>"'})
        )
        b_route = respx_mock.put("http://o/b").mock(
            return_value=Response(200, headers={"etag": '"<b-etag>"'})
        )

        urls = {
            "blake2b-aaa": [{"upload_url": "http://o/a", "chunk_id": None}],
            "blake2b-bbb": [{"upload_url": "http://o/b", "chunk_id": None}],
        }

        from httpx import Client as ClientT

        with ClientT() as client:
            etags = upload_chunks(client, two_plans, urls)

        assert a_route.called and b_route.called
        assert etags == {
            "blake2b-aaa": {1: '"<a-etag>"'},
            "blake2b-bbb": {1: '"<b-etag>"'},
        }

    @mark.respx()
    def test_skips_plans_with_no_urls(self, respx_mock, two_plans):
        a_route = respx_mock.put("http://o/a").mock(
            return_value=Response(200, headers={"etag": '"<a-etag>"'})
        )
        # Only one plan has a URL: the other was already uploaded.
        urls = {"blake2b-aaa": [{"upload_url": "http://o/a", "chunk_id": None}]}

        from httpx import Client as ClientT

        with ClientT() as client:
            etags = upload_chunks(client, two_plans, urls)

        assert a_route.called
        assert etags == {"blake2b-aaa": {1: '"<a-etag>"'}}


@mark.usefixtures(
    "monkeypatch_artifact_hub_client",
    "monkeypatch_project_hub_client",
    "monkeypatch_project_routes",
)
class TestCompleteUploads:
    @mark.respx()
    def test_one_call_per_batch(self, respx_mock, project):
        route = respx_mock.post(
            "projects/workspace/name/artifacts/complete"
        ).mock(return_value=Response(200))

        etags = {
            "blake2b-aaa": {1: '"<a>"'},
            "blake2b-bbb": {1: '"<b>"'},
        }

        from skore._plugins.hub.artifact.upload import HUBClient

        with HUBClient() as client:
            complete_uploads(client, project, etags)

        assert route.call_count == 1
        sent = route.calls.last.request.read()
        assert b"blake2b-aaa" in sent and b"blake2b-bbb" in sent

    @mark.respx(assert_all_called=False)
    def test_no_call_when_etags_empty(self, respx_mock, project):
        route = respx_mock.post(
            "projects/workspace/name/artifacts/complete"
        ).mock(return_value=Response(200))

        from skore._plugins.hub.artifact.upload import HUBClient

        with HUBClient() as client:
            complete_uploads(client, project, {})

        assert not route.called
