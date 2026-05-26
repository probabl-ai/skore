from typing import Literal

from httpx import Client as HTTPXClient
from httpx import Response
from pytest import fixture, mark

from skore._plugins.hub.artifact.artifact import Artifact
from skore._plugins.hub.artifact.plan import ArtifactPlan
from skore._plugins.hub.artifact.upload import (
    complete_uploads,
    request_upload_urls,
    upload_artifacts,
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


@mark.usefixtures("monkeypatch_artifact_hub_client")
class TestRequestUploadUrls:
    @mark.respx()
    def test_one_call_per_batch(self, respx_mock, two_plans):
        route = respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(
                201,
                json=[
                    {"checksum": "blake2b-aaa", "upload_url": "http://o/a"},
                    {"checksum": "blake2b-bbb", "upload_url": "http://o/b"},
                ],
            )
        )

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as client:
            urls = request_upload_urls(
                hub_client=client,
                workspace="workspace",
                project_name="name",
                plans=two_plans,
            )

        assert route.call_count == 1

        # Body contained all checksums in one payload.
        sent = route.calls.last.request.read()
        assert b"blake2b-aaa" in sent
        assert b"blake2b-bbb" in sent

        # Return value is aligned with input plans.
        assert len(urls) == 2
        assert urls[0][0]["upload_url"] == "http://o/a"
        assert urls[1][0]["upload_url"] == "http://o/b"

    @mark.respx()
    def test_already_uploaded_checksum_is_empty(self, respx_mock, two_plans):
        respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(
                201,
                json=[{"checksum": "blake2b-aaa", "upload_url": "http://o/a"}],
                # "blake2b-bbb" omitted: backend says it's already there.
            )
        )

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as client:
            urls = request_upload_urls(
                hub_client=client,
                workspace="workspace",
                project_name="name",
                plans=two_plans,
            )

        assert urls[0][0]["upload_url"] == "http://o/a"
        assert urls[1] == []


class TestUploadChunks:
    @mark.respx()
    def test_parallel_put_across_artifacts(self, respx_mock, two_plans):
        a_route = respx_mock.put("http://o/a").mock(
            return_value=Response(200, headers={"etag": '"<a-etag>"'})
        )
        b_route = respx_mock.put("http://o/b").mock(
            return_value=Response(200, headers={"etag": '"<b-etag>"'})
        )

        urls_per_plan = [
            [{"upload_url": "http://o/a", "chunk_id": None}],
            [{"upload_url": "http://o/b", "chunk_id": None}],
        ]

        with HTTPXClient() as client:
            etags = upload_chunks(
                storage_client=client,
                plans=two_plans,
                urls_per_plan=urls_per_plan,
            )

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
        # Only one plan has URLs: the other was already uploaded.
        urls_per_plan = [
            [{"upload_url": "http://o/a", "chunk_id": None}],
            [],
        ]

        with HTTPXClient() as client:
            etags = upload_chunks(
                storage_client=client,
                plans=two_plans,
                urls_per_plan=urls_per_plan,
            )

        assert a_route.called
        assert etags == {"blake2b-aaa": {1: '"<a-etag>"'}}


@mark.usefixtures("monkeypatch_artifact_hub_client")
class TestCompleteUploads:
    @mark.respx()
    def test_one_call_per_batch(self, respx_mock):
        route = respx_mock.post("projects/workspace/name/artifacts/complete").mock(
            return_value=Response(200)
        )

        etags = {
            "blake2b-aaa": {1: '"<a>"'},
            "blake2b-bbb": {1: '"<b>"'},
        }

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as client:
            complete_uploads(
                hub_client=client,
                workspace="workspace",
                project_name="name",
                etags_per_checksum=etags,
            )

        assert route.call_count == 1
        sent = route.calls.last.request.read()
        assert b"blake2b-aaa" in sent and b"blake2b-bbb" in sent

    @mark.respx(assert_all_called=False)
    def test_no_call_when_etags_empty(self, respx_mock):
        route = respx_mock.post("projects/workspace/name/artifacts/complete").mock(
            return_value=Response(200)
        )

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as client:
            complete_uploads(
                hub_client=client,
                workspace="workspace",
                project_name="name",
                etags_per_checksum={},
            )

        assert not route.called


@mark.usefixtures(
    "monkeypatch_tmpdir",
    "monkeypatch_artifact_hub_client",
    "monkeypatch_project_hub_client",
    "monkeypatch_project_routes",
)
class TestUploadArtifacts:
    @mark.respx()
    def test_returns_plans_aligned_with_artifacts(self, respx_mock, project):
        respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(201, json=[])  # nothing to upload, all known
        )

        from skore._plugins.hub.artifact.upload import HubClient

        artifacts = [
            FakeBytesArtifact(project=project, payload=b"one"),
            FakeNoneArtifact(project=project),
            FakeBytesArtifact(project=project, payload=b"two"),
        ]

        with HubClient() as hub_client, HTTPXClient() as storage_client:
            plans = upload_artifacts(
                hub_client=hub_client,
                storage_client=storage_client,
                workspace="workspace",
                project_name="name",
                artifacts=artifacts,
            )

        assert len(plans) == 3
        assert plans[0] is not None and plans[0].checksum.startswith("blake2b-")
        assert plans[1] is None  # no content
        assert plans[2] is not None
        assert plans[0].checksum != plans[2].checksum

    @mark.respx(assert_all_called=False)
    def test_no_network_calls_when_artifacts_empty(self, respx_mock, project):
        artifacts_route = respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(201, json=[])
        )

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as hub_client, HTTPXClient() as storage_client:
            plans = upload_artifacts(
                hub_client=hub_client,
                storage_client=storage_client,
                workspace="workspace",
                project_name="name",
                artifacts=[FakeNoneArtifact(project=project)],
            )

        assert plans == [None]
        assert not artifacts_route.called
