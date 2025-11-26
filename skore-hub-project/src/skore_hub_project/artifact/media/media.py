"""Class definition of the payload used to associate a media with the report."""

from abc import ABC
from functools import cached_property
from typing import Generic, TypeVar

from blake3 import blake3 as Blake3
from pydantic import Field, computed_field

from skore_hub_project import bytes_to_b64_str
from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


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
        """
        Checksum used to identify the content of the media.

        Notes
        -----
        Depending on the size of the serialized content, the checksum can be computed on
        one or more threads:

            Note that this can be slower for inputs shorter than ~1 MB

        https://github.com/oconnor663/blake3-py
        """
        if not self.computed:
            self.compute()

        if not self.filepath.stat().st_size:
            return None

        # Compute checksum with the appropriate number of threads
        threads = 1 if (self.filepath.stat().st_size < 1e6) else Blake3.AUTO
        hasher = Blake3(max_threads=threads)
        checksum = hasher.update_mmap(self.filepath).digest()

        return f"blake3-{bytes_to_b64_str(checksum)}"
