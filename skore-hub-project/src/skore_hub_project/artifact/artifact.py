"""Payload definition used to upload an artifact to ``hub``."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.upload import upload

Content = str | bytes | None


class Artifact(BaseModel, ABC):
    """
    Payload used to send the artifact of a report to ``hub``.

    Attributes
    ----------
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    project: Project = Field(repr=False, exclude=True)
    content_type: str

    @abstractmethod
    def content_to_upload(self) -> Content | AbstractContextManager[Content]:
        """
        Example
        -------

            def content_to_upload(self) -> str:
                return "<str>"

        or

            from contextlib import contextmanager

            @contextmanager
            def content_to_upload(self) -> Generator[str, None, None]:
                yield "<str>"
        """

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def checksum(self) -> str | None:
        """Checksum, useful for retrieving the artifact from artifact storage."""
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
