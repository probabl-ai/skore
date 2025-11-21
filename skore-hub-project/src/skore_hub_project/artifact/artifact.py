"""Interface definition of the payload used to associate an artifact with a project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.serializer import Serializer, TxtSerializer
from skore_hub_project.artifact.upload import upload as upload_content
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

Content = EstimatorReport | CrossValidationReport | str | bytes | None


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: Project = Field(repr=False, exclude=True)
    serializer_cls: ClassVar[type[Serializer]] = TxtSerializer
    content_type: str = Field(init=False)

    @property
    @abstractmethod
    def content_to_upload(self) -> bytes | None:
        """Content of the artifact to upload."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def checksum(self) -> str | None:
        """Checksum used to identify the content of the artifact."""

    def upload(
        self,
        *,
        pool: ThreadPoolExecutor | None = None,
        checksums_being_uploaded: set[str] | None = None,
    ) -> None:
        """Upload the artifact and set its checksum."""
        contextmanager = self.content_to_upload()
        checksums_being_uploaded = (
            set() if checksums_being_uploaded is None else checksums_being_uploaded
        )

        if not isinstance(contextmanager, AbstractContextManager):
            contextmanager = nullcontext(contextmanager)

        with contextmanager as content:
            if content is None:
                self.checksum = None
                return

            with self.serializer_cls(content) as serializer:
                if serializer.checksum not in checksums_being_uploaded:
                    checksums_being_uploaded.add(serializer.checksum)
                    upload_content(
                        project=self.project,
                        serializer=serializer,
                        content=content,
                        content_type=self.content_type,
                        pool=(
                            ThreadPoolExecutor(max_workers=6) if pool is None else pool
                        ),
                    )

                self.checksum = serializer.checksum
