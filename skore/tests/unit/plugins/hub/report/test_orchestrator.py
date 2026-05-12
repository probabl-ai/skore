from httpx import Response
from pytest import fixture, mark

from skore._plugins.hub.report import EstimatorReportPayload


@fixture(autouse=True)
def _skip_table_report(monkeypatch):
    """Skip skrub TableReport rendering (heavy + has its own conftest stub)."""
    monkeypatch.setattr(
        "skore._plugins.hub.artifact.media.data.TableReport.content_to_upload",
        lambda self: None,
    )


class TestUploadArtifacts:
    @mark.respx(assert_all_called=False)
    def test_single_post_artifacts_and_single_complete(
        self, respx_mock, project, binary_classification
    ):
        # Override the conftest's per-artifact POST /artifacts route with one
        # that returns no URLs (idempotency short-circuit). The conftest's
        # `setup_report` autouse fixture installs default upload routes; here
        # we register a more specific handler that takes precedence.
        artifacts_route = respx_mock.post(
            "projects/workspace/name/artifacts"
        ).mock(return_value=Response(201, json=[]))
        complete_route = respx_mock.post(
            "projects/workspace/name/artifacts/complete"
        ).mock(return_value=Response(200))

        payload = EstimatorReportPayload(
            project=project, key="k", report=binary_classification
        )

        from httpx import Client as HTTPXClient
        from skore._plugins.hub.artifact.upload import HUBClient

        with HUBClient() as hub_client, HTTPXClient() as storage_client:
            payload.upload_artifacts(hub_client, storage_client)

        # The whole point of the optimization: exactly one batched POST.
        assert artifacts_route.call_count == 1
        # With an empty URL response, nothing was uploaded so /complete is
        # never called.
        assert complete_route.call_count == 0

        # The /artifacts payload contains ALL artifacts (medias + pickle)
        # in a single list.
        import json

        sent = artifacts_route.calls.last.request.read()
        body = json.loads(sent)
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
        import json

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

        from httpx import Client as HTTPXClient
        from skore._plugins.hub.artifact.upload import HUBClient

        with HUBClient() as hub_client, HTTPXClient() as storage_client:
            payload.upload_artifacts(hub_client, storage_client)

        # POST /artifacts: one call with all checksums.
        assert len(captured) == 1
        total = len(captured[0])

        # POST /complete: one call, listing only the newly-uploaded second half.
        assert complete_route.call_count == 1
        completed = json.loads(complete_route.calls.last.request.read())
        assert len(completed) == total - (total // 2)
