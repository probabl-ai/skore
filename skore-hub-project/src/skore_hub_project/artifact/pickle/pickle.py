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
    It uploads the pickled report to the artifacts storage in a lazy way.

    The report is uploaded without its cache, to avoid salting the checksum.
    The report is primarily pickled on disk to reduce RAM footprint.
    """

    content_type: Literal["application/octet-stream"] = "application/octet-stream"
    report: Report = Field(repr=False, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def checksum(self) -> str:
        """The checksum of the pickled report."""
        return f"skore-{self.report.__class__.__name__}-{self.report._hash}"

    def compute(self) -> None:  # noqa: D102
        if self.computed:
            return

        self.computed = True

        reports = [self.report] + getattr(self.report, "estimator_reports_", [])
        caches = [report_to_clear._cache for report_to_clear in reports]

        self.report.clear_cache()

        try:
            dump(self.report, self.filepath)
        finally:
            for report, cache in zip(reports, caches, strict=True):
                report._cache = cache
