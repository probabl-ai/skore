"""Definition of the payload used to associate the pickled report with the report."""

from functools import cached_property
from typing import Literal

from joblib import dump
from pydantic import Field, computed_field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

Report = EstimatorReport | CrossValidationReport


class Pickle(Artifact):
    """
    Payload used to associate the pickled report with the report, using ``joblib``.

    Attributes
    ----------
    project : Project
        The project to which the pickle's payload must be associated.
    content_type : str
        The content-type of the pickle content.
    computed : bool
        True when the pickle content is computed, False otherwise.
    uploaded : bool
        True when the pickle is uploaded, False otherwise.
    report : EstimatorReport | CrossValidationReport
        The report to pickle.

    Notes
    -----
    The artifact is computed in a lazy way. It pickles the report on disk to reduce RAM
    footprint, before uploading it to the artifacts storage.
    """

    content_type: Literal["application/octet-stream"] = "application/octet-stream"
    report: Report = Field(repr=False, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def checksum(self) -> str:
        """The checksum of the pickled report."""
        return f"skore-{self.report.__class__.__name__}-{self.report.id}"

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        self.computed = True
        dump(self.report, self.filepath)
