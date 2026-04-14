"""Class definition of the payload used to associate a media with the report."""

from __future__ import annotations

from abc import ABC
from functools import cached_property
from hashlib import blake2b
from sys import version_info
from typing import Generic, TypeVar

from pydantic import Field, computed_field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))

if version_info >= (3, 11):
    from hashlib import file_digest
else:

    def file_digest(fileobj, digest, /, *, _bufsize=2**18):  # type: ignore
        """Backported from https://github.com/python/cpython/blob/3.14/Lib/hashlib.py#L195."""
        digestobj = digest()

        if hasattr(fileobj, "getbuffer"):
            digestobj.update(fileobj.getbuffer())
            return digestobj

        if not (
            hasattr(fileobj, "readinto")
            and hasattr(fileobj, "readable")
            and fileobj.readable()
        ):
            raise ValueError(
                f"'{fileobj!r}' is not a file-like object in binary reading mode."
            )

        buf = bytearray(_bufsize)  # Reusable buffer to reduce allocations.
        view = memoryview(buf)
        while True:
            size = fileobj.readinto(buf)
            if size is None:
                raise BlockingIOError("I/O operation would block.")
            if size == 0:
                break  # EOF
            digestobj.update(view[:size])

        return digestobj


class Media(Artifact, ABC, Generic[Report]):
    """
    Payload used to associate a media with the report.

    Attributes
    ----------
    project : Project
        The project to which the media's payload must be associated.
    content_type : str
        The content-type of the media content.
    computed : bool
        True when the media content is computed, False otherwise.
    uploaded : bool
        True when the media is uploaded, False otherwise.
    report : EstimatorReport | CrossValidationReport
        The report on which compute the media.
    name : str
        The name of the media.
    data_source : str | None
        The source of the data used to generate the media.
    """

    report: Report = Field(repr=False, exclude=True)
    name: str = Field(init=False)
    data_source: str | None = Field(init=False)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def checksum(self) -> str | None:
        """The checksum of the serialized media using `BLAKE2b`."""
        if not self.computed:
            self.compute()

        if not self.filepath.stat().st_size:
            return None

        with open(self.filepath, "rb") as file:
            digest = file_digest(file, blake2b)

        return f"blake2b-{digest.hexdigest()}"
