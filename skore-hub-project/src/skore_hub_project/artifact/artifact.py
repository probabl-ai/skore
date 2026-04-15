"""Interface definition of the payload used to associate an artifact with a project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from skore_hub_project.artifact.upload import upload, uploaded as file_uploaded
from skore_hub_project.client.client import HUBClient
from skore_hub_project.project.project import Project


class Artifact(BaseModel, ABC):
    """
    Interface definition of the payload used to associate an artifact with a project.

    Attributes
    ----------
    project : Project
        The project to which the artifact's payload must be associated.
    content_type : str
        The content-type of the artifact content.
    computed : bool
        True when the artifact content is computed, False otherwise.
    uploaded : bool
        True when the artifact is uploaded, False otherwise.

    Notes
    -----
    It triggers the upload of the content of the artifact, in a lazy way. It is uploaded
    as a file to the ``hub`` artifacts storage.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: Project = Field(repr=False, exclude=True)
    content_type: str = Field(init=False)
    computed: bool = Field(init=False, repr=False, exclude=True, default=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def checksum(self) -> str | None:
        """The checksum used to identify the content of the artifact."""

    def __del__(self) -> None:  # noqa: D105
        self.filepath.unlink(True)

    @cached_property
    def filepath(self) -> Path:
        """The temporary filepath used to store the content of the artifact."""
        with NamedTemporaryFile(mode="w+b", delete=False) as file:
            return Path(file.name)

    @abstractmethod
    def compute(self) -> None:
        """
        Compute and write the content of the artifact in ``artifact.filepath``.

        Notes
        -----
        It is triggered when ``artifact.upload`` is called, in a lazy way.
        """

    @property
    def uploaded(self) -> bool:
        assert self.checksum is not None, "`checksum` must not be None"

        if not hasattr(self, "__uploaded"):
            self.__uploaded = file_uploaded(self.project, self.checksum)

        return self.__uploaded

    def upload(self) -> None:
        """
        Upload the artifact.

        Notes
        -----
        Artifact that was already uploaded in its whole will be ignored.
        It triggers the compute of the content of the artifact, in a lazy way.
        """
        assert self.checksum is not None, "`checksum` must not be None"

        if not self.uploaded:
            assert self.computed, "`artifact` must be computed"
            assert not self.uploaded, "`artifact` must not be uploaded"

            upload(
                project=self.project,
                checksum=self.checksum,
                filepath=self.filepath,
                content_type=self.content_type,
            )

        self.__uploaded = True

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:  # noqa: D102
        assert self.checksum is not None, "`checksum` must not be None"
        assert self.computed, "`artifact` must be computed"
        assert self.uploaded, "`artifact` must be uploaded"

        return super().model_dump(**kwargs)
