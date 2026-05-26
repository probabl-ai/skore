"""Definition of the payload used to associate the pickled report with the report."""

from io import BytesIO
from typing import Literal

from joblib import dump
from pydantic import Field

from skore import CrossValidationReport, EstimatorReport
from skore._plugins.hub.artifact.artifact import Artifact

Report = EstimatorReport | CrossValidationReport


class Pickle(Artifact):
    """
    Payload used to associate the pickled report with the report, using ``joblib``.

    Attributes
    ----------
    project : Project
        The project to which the artifact's payload must be associated.
    content_type : str
        The content-type of the artifact content.
    report : EstimatorReport | CrossValidationReport
        The report to be pickled.

    Notes
    -----
    It uploads the pickled report to the artifacts storage in a lazy way.

    The report is uploaded without its cache, to avoid salting the checksum.
    """

    report: Report = Field(repr=False, exclude=True)
    content_type: Literal["application/octet-stream"] = "application/octet-stream"

    def content_to_upload(self) -> bytes:
        """Compute the pickled report (without cache) to upload."""
        reports = [self.report] + getattr(self.report, "estimator_reports_", [])
        reports_with_cache = [
            (report, report._cache) for report in reports if hasattr(report, "_cache")
        ]
        self.report.clear_cache()

        try:
            with BytesIO() as stream:
                dump(self.report, stream)
                return stream.getvalue()
        finally:
            for report, cache in reports_with_cache:
                report._cache = cache
