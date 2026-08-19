"""Definition of the payload used to associate the pickled report with the report."""

from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from typing import Literal

from joblib import dump
from pydantic import Field

from skore import CrossValidationReport, EstimatorReport
from skore.plugins.hub.artifact.artifact import Artifact

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

    The report is uploaded without its prediction/display cache and without its
    checks-results cache, to avoid salting the checksum.
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
        The report is pickled without its prediction/display cache and without its
        checks-results cache, to avoid salting the checksum.
        """
        reports = [self.report] + getattr(self.report, "reports_", [])
        reports_with_cache = [
            (report, report._cache) for report in reports if hasattr(report, "_cache")
        ]
        reports_with_check_results_cache = [
            (report, report._check_results_cache)
            for report in reports
            if hasattr(report, "_check_results_cache")
        ]
        self.report._clear_cache()
        for report, _ in reports_with_check_results_cache:
            del report._check_results_cache

        try:
            with BytesIO() as stream:
                dump(self.report, stream)

                pickle_bytes = stream.getvalue()

            yield pickle_bytes
        finally:
            for report, cache in reports_with_cache:
                report._cache = cache
            for report, check_results_cache in reports_with_check_results_cache:
                report._check_results_cache = check_results_cache
