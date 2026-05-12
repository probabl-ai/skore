from contextlib import contextmanager
from typing import Literal

from pytest import mark

from skore._plugins.hub.artifact.artifact import Artifact
from skore._plugins.hub.artifact.plan import ArtifactPlan


class FakeBytesArtifact(Artifact):
    content_type: Literal["text/plain"] = "text/plain"

    def content_to_upload(self) -> bytes:
        return b"hello"


class FakeNoneArtifact(Artifact):
    content_type: Literal["text/plain"] = "text/plain"

    def content_to_upload(self) -> None:
        return None


class FakeContextArtifact(Artifact):
    content_type: Literal["application/octet-stream"] = "application/octet-stream"

    @contextmanager
    def content_to_upload(self):
        yield b"context-managed"


@mark.usefixtures(
    "monkeypatch_tmpdir",
    "monkeypatch_project_hub_client",
    "monkeypatch_project_routes",
)
class TestArtifactLocalPlan:
    def test_returns_plan_for_bytes_content(self, project):
        artifact = FakeBytesArtifact(project=project)
        plan = artifact.local_plan()

        assert isinstance(plan, ArtifactPlan)
        assert plan.content_type == "text/plain"
        assert plan.size == 5
        assert plan.checksum.startswith("blake2b-")
        # Small content stays in memory.
        assert plan.payload == b"hello"

    def test_returns_none_when_content_is_none(self, project):
        artifact = FakeNoneArtifact(project=project)
        assert artifact.local_plan() is None

    def test_handles_context_manager_content(self, project):
        artifact = FakeContextArtifact(project=project)
        plan = artifact.local_plan()
        assert plan is not None
        assert plan.payload == b"context-managed"

    def test_does_not_touch_network(self, project, respx_mock):
        # The ``project`` fixture itself triggers requests during ``Project``
        # construction; reset the mock so we only observe ``local_plan()``.
        respx_mock.reset()
        artifact = FakeBytesArtifact(project=project)
        plan = artifact.local_plan()
        assert plan is not None
        # No requests should have been made by ``local_plan()``.
        assert len(respx_mock.calls) == 0
