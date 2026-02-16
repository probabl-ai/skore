"""Definition of the payload used to associate the pickled report with the report."""

from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from typing import Literal

from joblib import dump
from pydantic import Field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

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
        The report to pickled.

    Notes
    -----
    It uploads the pickled report to the artifacts storage in a lazy way.

    The report is uploaded without its cache, to avoid salting the checksum.
    The report is primarily pickled on disk to reduce RAM footprint.
    """

    report: Report = Field(repr=False, exclude=True)
    content_type: Literal["application/octet-stream"] = "application/octet-stream"

    @contextmanager
    def content_to_upload(self) -> Generator[bytes, None, None]:
        """
        Content of the pickled report.

        Notes
        -----
        The report is pickled without its cache, to avoid salting the checksum.
        """
        reports = [self.report]
        if hasattr(self.report, "reports_"):
            reports.extend(self.report.reports_)
        elif hasattr(self.report, "estimator_reports_"):
            # TODO: remove this when the minimum version of skore is 0.13
            # it is only necessary for backward compatibility with skore < 0.13
            reports.extend(self.report.estimator_reports_)
        caches = [report_to_clear._cache for report_to_clear in reports]

        self.report.clear_cache()

        try:
            with BytesIO() as stream:
                dump(self.report, stream)

                pickle_bytes = stream.getvalue()

            yield pickle_bytes
        finally:
            for report, cache in zip(reports, caches, strict=True):
                report._cache = cache
