"""Payload definition used to upload a report to ``hub`` using ``joblib``."""

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
    Payload used to upload a report to ``hub`` using ``joblib``.

    Notes
    -----
    It uploads the report pickle to the artifacts storage in a lazy way.

    The report is uploaded without its cache, to avoid salting the checksum.
    The report is primarily pickled on disk to reduce RAM footprint.
    """

    report: Report = Field(repr=False, exclude=True)
    content_type: Literal["application/octet-stream"] = "application/octet-stream"

    @contextmanager
    def content_to_upload(self) -> Generator[bytes, None, None]:
        """
        Notes
        -----
        The report is pickled without its cache, to avoid salting the hash.
        """
        reports = [self.report] + getattr(self.report, "estimator_reports_", [])
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
