"""Interface definition of the payload used to associate an artifact with a project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project.artifact.upload import upload, uploaded
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
        """Compute and write the content of the artifact in ``artifact.filepath``."""

    @property
    def uploaded(self) -> bool:
        assert self.checksum is not None, "`checksum` must not be None"

        if not hasattr(self, "_Artifact__uploaded"):
            self.__uploaded = uploaded(self.project, self.checksum)

        return self.__uploaded

    def upload(self) -> None:
        """Upload the artifact."""
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
