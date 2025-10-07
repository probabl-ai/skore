"""Payload definition used to upload an artifact to ``hub``."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import AbstractContextManager, nullcontext
from functools import cached_property
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.artifact.upload import upload


class Artifact(BaseModel, ABC):
    """
    Payload used to send the artifact of a report to ``hub``.

    Attributes
    ----------
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    serializer: ClassVar[type[Serializer]]
    project: Project = Field(repr=False, exclude=True)

    @abstractmethod
    def object_to_upload(self) -> Any | Generator[Any, None, None]:
        """
        Example
        -------

            def object_to_upload(self) -> str:
                return "<str>"

        or

            from contextlib import contextmanager

            @contextmanager
            def object_to_upload(self) -> Generator[str, None, None]:
                yield "<str>"
        """

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def checksum(self) -> str | None:
        """Checksum, useful for retrieving the artifact from artifact storage."""
        contextmanager = self.object_to_upload()

        if not isinstance(contextmanager, AbstractContextManager):
            contextmanager = nullcontext(contextmanager)

        with contextmanager as object:
            if object is not None:
                return upload(
                    project=self.project,
                    object=object,
                    serializer_cls=self.serializer,
                )

        return None
