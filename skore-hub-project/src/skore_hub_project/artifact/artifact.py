"""Interface definition of the payload used to associate an artifact with a project."""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.artifact.upload import upload
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
    serializer_cls: ClassVar[type[Serializer]] = Serializer
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def checksum(self) -> str | None:
        """Checksum used to identify the content of the artifact."""
        try:
            return self.__checksum
        except AttributeError:
            message = (
                "You cannot access the checksum of an artifact "
                "without explicitly uploading it. "
                "Please use `artifact.upload()` before."
            )

            raise RuntimeError(message) from None

    @checksum.setter
    def checksum(self, checksum: str | None):
        self.__checksum = checksum

    def upload(self):
        """Upload artifact."""
        contextmanager = self.content_to_upload()

        if not isinstance(contextmanager, AbstractContextManager):
            contextmanager = nullcontext(contextmanager)

        with contextmanager as content:
            if content is not None:
                self.checksum = upload(
                    project=self.project,
                    serializer_cls=self.serializer_cls,
                    content=content,
                    content_type=self.content_type,
                )
            else:
                self.checksum = None
