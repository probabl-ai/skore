"""End-to-end tests of the batched upload pipeline at the orchestrator level."""

import json

from httpx import Client as HTTPXClient
from httpx import Response
from pytest import fixture, mark

from skore._plugins.hub.artifact.pickle import Pickle
from skore._plugins.hub.artifact.upload import upload_artifacts
from skore._plugins.hub.report import EstimatorReportPayload


@fixture(autouse=True)
def _skip_table_report(monkeypatch):
    """Skip skrub TableReport rendering (heavy + has its own conftest stub)."""
    monkeypatch.setattr(
        "skore._plugins.hub.artifact.media.data.TableReport.content_to_upload",
        lambda self: None,
    )


def _artifacts_for(payload, project, report):
    media_artifacts = [
        media_cls(project=project, report=report) for media_cls in payload.MEDIAS
    ]
    return [*media_artifacts, Pickle(project=project, report=report)]


class TestUploadArtifacts:
    @mark.respx(assert_all_called=False)
    def test_single_post_artifacts_and_single_complete(
        self, respx_mock, project, binary_classification
    ):
        # Backend reports every checksum as already-uploaded (idempotency short-circuit).
        artifacts_route = respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(201, json=[])
        )
        complete_route = respx_mock.post(
            "projects/workspace/name/artifacts/complete"
        ).mock(return_value=Response(200))

        payload = EstimatorReportPayload(
            project=project, key="k", report=binary_classification
        )
        artifacts = _artifacts_for(payload, project, binary_classification)

        # Late import: ``monkeypatch_artifact_hub_client`` swaps the symbol in
        # the upload module's namespace; binding at module load locks in the
        # real class.
        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as hub_client, HTTPXClient() as storage_client:
            upload_artifacts(
                hub_client=hub_client,
                storage_client=storage_client,
                workspace="workspace",
                project_name="name",
                artifacts=artifacts,
            )

        # Exactly one batched POST /artifacts.
        assert artifacts_route.call_count == 1
        # With an empty URL response, nothing was uploaded so /complete is
        # never called.
        assert complete_route.call_count == 0

        # The /artifacts payload contains every artifact with content.
        body = json.loads(artifacts_route.calls.last.request.read())
        assert isinstance(body, list)
        assert len(body) >= 2  # at least pickle + one media

    @mark.respx(assert_all_called=False)
    def test_skips_already_uploaded_artifacts(
        self, respx_mock, project, binary_classification
    ):
        """If the backend omits half the checksums from its response (signaling
        they already exist), the client skips their PUTs and the
        ``/artifacts/complete`` call only lists the newly-uploaded ones.
        """
        captured = []

        def first_half_uploaded(request):
            body = json.loads(request.read())
            captured.append(body)
            half = len(body) // 2
            return Response(
                201,
                json=[
                    {
                        "checksum": entry["checksum"],
                        "upload_url": f"http://chunk-{i}.com/",
                        "chunk_id": None,
                    }
                    for i, entry in enumerate(body[half:], start=half)
                ],
            )

        respx_mock.post("projects/workspace/name/artifacts").mock(
            side_effect=first_half_uploaded
        )
        respx_mock.put(url__regex=r"http://chunk-\d+\.com.*").mock(
            return_value=Response(200, headers={"etag": '"<etag>"'})
        )
        complete_route = respx_mock.post(
            "projects/workspace/name/artifacts/complete"
        ).mock(return_value=Response(200))

        payload = EstimatorReportPayload(
            project=project, key="k", report=binary_classification
        )
        artifacts = _artifacts_for(payload, project, binary_classification)

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as hub_client, HTTPXClient() as storage_client:
            upload_artifacts(
                hub_client=hub_client,
                storage_client=storage_client,
                workspace="workspace",
                project_name="name",
                artifacts=artifacts,
            )

        # POST /artifacts: one call with all checksums.
        assert len(captured) == 1
        total = len(captured[0])

        # POST /complete: one call, listing only the newly-uploaded second half.
        assert complete_route.call_count == 1
        completed = json.loads(complete_route.calls.last.request.read())
        assert len(completed) == total - (total // 2)

    @mark.respx(assert_all_called=False)
    def test_returns_plans_aligned_with_artifacts(
        self, respx_mock, project, binary_classification
    ):
        respx_mock.post("projects/workspace/name/artifacts").mock(
            return_value=Response(201, json=[])
        )

        payload = EstimatorReportPayload(
            project=project, key="k", report=binary_classification
        )
        artifacts = _artifacts_for(payload, project, binary_classification)

        from skore._plugins.hub.artifact.upload import HubClient

        with HubClient() as hub_client, HTTPXClient() as storage_client:
            plans = upload_artifacts(
                hub_client=hub_client,
                storage_client=storage_client,
                workspace="workspace",
                project_name="name",
                artifacts=artifacts,
            )

        assert len(plans) == len(artifacts)
        # The pickle artifact (last) always has content.
        assert plans[-1] is not None and plans[-1].checksum.startswith("blake2b-")
