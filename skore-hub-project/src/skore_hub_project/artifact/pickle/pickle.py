"""Definition of the payload used to associate the pickled report with the report."""

from collections.abc import Generator
from contextlib import contextmanager
from functools import cached_property
from io import BytesIO
from typing import ClassVar, Literal

from joblib import dump
from pydantic import Field

from skore_hub_project.artifact.artifact import Artifact
from skore_hub_project.artifact.serializer import Serializer
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

Report = EstimatorReport | CrossValidationReport


class ReportSerializer(Serializer):
    """Serialize a report using joblib, directly on disk to reduce RAM footprint."""

    def __init__(self, report: Report):
        self.report = report

    def __call__(self) -> None:
        """Serialize a report using joblib, directly on disk to reduce RAM footprint."""
        if hasattr(self, "__called__"):
            return

        self.__called__ = True

        with BytesIO() as stream:
            dump(self.report, stream)
            self.filepath.write_bytes(stream.getvalue())

    @cached_property
    def checksum(self) -> str:
        """The checksum of the serialized report."""
        return f"skore-{self.report.__class__.__name__}-{self.report._hash}"


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

    serializer_cls: ClassVar[type[ReportSerializer]] = ReportSerializer
    content_type: Literal["application/octet-stream"] = "application/octet-stream"
    report: Report = Field(repr=False, exclude=True)

    @contextmanager
    def content_to_upload(self) -> Generator[Report, None, None]:
        """
        Content of the pickled report.

        Notes
        -----
        The report is pickled without its cache, to avoid salting the checksum.
        """
        reports = [self.report] + getattr(self.report, "estimator_reports_", [])
        caches = [report_to_clear._cache for report_to_clear in reports]

        self.report.clear_cache()

        try:
            yield self.report
        finally:
            for report, cache in zip(reports, caches, strict=True):
                report._cache = cache
