"""Interface definition of the payload used to associate an artifact with a project."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.upload import upload

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
    content_type: str

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

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def checksum(self) -> str | None:
        """Checksum used to identify the content of the artifact."""
        contextmanager = self.content_to_upload()

        if not isinstance(contextmanager, AbstractContextManager):
            contextmanager = nullcontext(contextmanager)

        with contextmanager as content:
            if content is not None:
                return upload(
                    project=self.project,
                    content=content,
                    content_type=self.content_type,
                )

        return None
