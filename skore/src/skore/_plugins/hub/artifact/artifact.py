"""Interface definition of the payload used to associate an artifact with a project."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from hashlib import blake2b
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore._plugins.hub.artifact.plan import ArtifactPlan
from skore._plugins.hub.project.project import Project

Content = str | bytes | None


class Artifact(BaseModel, ABC):
    """
    Interface definition of the payload used to associate an artifact with a project.

    Attributes
    ----------
    project : Project
        The project to which the artifact's payload must be associated.
    content_type : str
        The content-type of the artifact content.

    Notes
    -----
    It triggers the upload of the content of the artifact, in a lazy way. It is uploaded
    as a file to the ``hub`` artifacts storage.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    project: Project = Field(repr=False, exclude=True)
    content_type: str = Field(init=False)

    @abstractmethod
    def content_to_upload(self) -> Content | AbstractContextManager[Content]:
        """
        Content of the artifact to upload.

        Example
        -------
        You can implement this ``abstractmethod`` to return directly the content:

            def content_to_upload(self) -> str:
                return "<str>"

        or to yield the content, as a ``contextmanager`` would:

            from contextlib import contextmanager

            @contextmanager
            def content_to_upload(self) -> Generator[str, None, None]:
                yield "<str>"
        """

    # Below this threshold we keep content in memory; above, we spool to disk.
    SMALL_CONTENT_THRESHOLD: ClassVar[int] = 10 * 1024 * 1024  # 10 MB

    def local_plan(self) -> ArtifactPlan | None:
        """Compute content + checksum locally. No network calls.

        Returns
        -------
        ArtifactPlan | None
            A plan describing the prepared artifact, or ``None`` if the
            artifact has no content to upload for this report (e.g., a
            confusion-matrix media on a regression report).
        """
        contextmanager = self.content_to_upload()

        if not isinstance(contextmanager, AbstractContextManager):
            contextmanager = nullcontext(contextmanager)

        with contextmanager as content:
            if content is None:
                return None

            if isinstance(content, str):
                content = content.encode("utf-8")

            if len(content) <= self.SMALL_CONTENT_THRESHOLD:
                checksum = blake2b(content).hexdigest()
                return ArtifactPlan(
                    checksum=f"blake2b-{checksum}",
                    size=len(content),
                    content_type=self.content_type,
                    payload=content,
                )

            # Big content: spool to a tempfile.
            with NamedTemporaryFile(mode="wb", delete=False) as f:
                f.write(content)
                path = Path(f.name)

            digest = blake2b()
            with path.open("rb") as f:
                while chunk := f.read(1 << 20):
                    digest.update(chunk)

            return ArtifactPlan(
                checksum=f"blake2b-{digest.hexdigest()}",
                size=path.stat().st_size,
                content_type=self.content_type,
                payload=path,
            )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checksum(self) -> str | None:
        """Checksum assigned during ``ReportPayload.upload_artifacts``.

        Reads from ``self._plan`` (attached by the orchestrator). Returns
        ``None`` if the artifact has no content (e.g., a confusion-matrix
        media on a regression report) or if no orchestrator has run yet.
        """
        plan = getattr(self, "_plan", None)
        return plan.checksum if plan is not None else None
