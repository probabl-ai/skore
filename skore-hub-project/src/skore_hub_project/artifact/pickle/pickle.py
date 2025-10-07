"""Payload definition used to upload a report to ``hub`` using ``joblib``."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import ClassVar

from pydantic import Field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.artifact.serializer import JoblibSerializer
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

    serializer: ClassVar[type] = JoblibSerializer
    report: Report = Field(repr=False, exclude=True)

    @contextmanager
    def object_to_upload(self) -> Generator[Report, None, None]:
        """
        Notes
        -----
        The report is pickled without its cache, to avoid salting the hash.
        """
        reports = [self.report] + getattr(self.report, "estimator_reports_", [])
        caches = [report_to_clear._cache for report_to_clear in reports]

        self.report.clear_cache()

        try:
            yield self.report
        finally:
            for report, cache in zip(reports, caches, strict=True):
                report._cache = cache
