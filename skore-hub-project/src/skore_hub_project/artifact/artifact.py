"""Interface definition of the payload used to associate an artifact with a project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.upload import upload as upload_content
from skore_hub_project.client.client import HUBClient


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
    uploaded: bool = Field(init=False, repr=False, exclude=True, default=False)

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def checksum(self) -> str | None:
        """The checksum used to identify the content of the artifact."""

    def __del__(self) -> None:  # noqa: D105
        self.filepath.unlink(True)

    @cached_property
    def filepath(self) -> Path:
        """The filepath used to store the content."""
        with NamedTemporaryFile(mode="w+b", delete=False) as file:
            return Path(file.name)

    @abstractmethod
    def compute(self) -> None: ...

    def upload(
        self,
        *,
        pool: ThreadPoolExecutor,
        checksums_being_uploaded: set[str],
    ) -> None:
        """
        Upload the artifact.

        Notes
        -----
        Artifact that was already uploaded in its whole will be ignored.
        """
        self.uploaded = True

        try:
            if (self.checksum is None) or (self.checksum in checksums_being_uploaded):
                return None

            checksums_being_uploaded.add(self.checksum)

            with HUBClient() as hub_client:
                # Ask for the artifact.
                #
                # An non-empty response means that an artifact with the same checksum
                # already exists. The content doesn't have to be re-uploaded.
                if (
                    response := hub_client.get(
                        url=f"projects/{self.project.quoted_tenant}/{self.project.quoted_name}/artifacts",
                        params={
                            "artifact_checksum": self.checksum,
                            "status": "uploaded",
                        },
                    )
                ) and (response.json()):
                    return None

            self.compute()

            upload_content(
                project=self.project,
                checksum=self.checksum,
                filepath=self.filepath,
                content_type=self.content_type,
                pool=pool,
            )
        finally:
            self.filepath.unlink(missing_ok=True)

    def model_dump(self, *args, **kwargs):
        if not self.uploaded:
            raise RuntimeError(
                "You cannot access the dictionary representation of the model of an "
                "artifact without explicitly uploading it. "
                "Please use `artifact.upload()` before."
            )

        return super().model_dump(*args, **kwargs)
