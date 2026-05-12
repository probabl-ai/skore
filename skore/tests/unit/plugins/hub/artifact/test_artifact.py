from typing import Literal

from pytest import mark

from skore._plugins.hub.artifact.artifact import Artifact
from skore._plugins.hub.artifact.plan import ArtifactPlan
from skore._plugins.hub.artifact.upload import plan_upload


class FakeBytesArtifact(Artifact):
    content_type: Literal["text/plain"] = "text/plain"

    def content_to_upload(self) -> bytes:
        return b"hello"


class FakeNoneArtifact(Artifact):
    content_type: Literal["text/plain"] = "text/plain"

    def content_to_upload(self) -> None:
        return None


@mark.usefixtures(
    "monkeypatch_tmpdir",
    "monkeypatch_project_hub_client",
    "monkeypatch_project_routes",
)
class TestPlanUpload:
    def test_returns_plan_for_bytes_content(self, project):
        plan = plan_upload(FakeBytesArtifact(project=project))

        assert isinstance(plan, ArtifactPlan)
        assert plan.content_type == "text/plain"
        assert plan.size == 5
        assert plan.checksum.startswith("blake2b-")
        # Small content stays in memory.
        assert plan.payload == b"hello"

    def test_returns_none_when_content_is_none(self, project):
        assert plan_upload(FakeNoneArtifact(project=project)) is None

    def test_does_not_touch_network(self, project, respx_mock):
        # The ``project`` fixture itself triggers requests during ``Project``
        # construction; reset the mock so we only observe ``plan_upload``.
        respx_mock.reset()
        plan = plan_upload(FakeBytesArtifact(project=project))
        assert plan is not None
        assert len(respx_mock.calls) == 0
