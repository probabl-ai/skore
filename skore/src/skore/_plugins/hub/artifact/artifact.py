"""Interface definition of the payload used to associate an artifact with a project."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

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
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    project: Project = Field(repr=False, exclude=True)
    content_type: str = Field(init=False)

    @abstractmethod
    def content_to_upload(self) -> Content:
        """Compute the content to upload, or return ``None`` if not applicable."""
