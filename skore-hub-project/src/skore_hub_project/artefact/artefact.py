from __future__ import annotations

from functools import cached_property
from abc import ABC
from typing import Any, ClassVar, Literal

from skore import EstimatorReport, CrossValidationReport
from pydantic import BaseModel, Field, computed_field


from .upload import upload


Project = Any


class Artefact(ABC, BaseModel):
    project: Project = Field(repr=False, exclude=True)
    object: Any = Field(repr=False, exclude=True)


class EstimatorReportArtefact(Artefact):
    """
    Notes
    -----
    It uploads the report to artefacts storage in a lazy way.

    The report is uploaded without its cache, to avoid salting the checksum.
    The report is primarily pickled on disk to reduce RAM footprint.
    """

    object: EstimatorReport = Field(repr=False, exclude=True, alias="report")

    @computed_field
    @cached_property
    def checksum(self) -> str:
        """
        Artefact checksum, useful for retrieving the artefact from artefact storage.
        """

        cache = report._cache
        report._cache = {}

        try:
            return upload(self.project, self.report, "estimator-report")
        finally:
            report._cache = cache


class CrossValidationReportArtefact(Artefact):
    """
    Notes
    -----
    It uploads the report to artefacts storage in a lazy way.

    The report is uploaded without its cache, to avoid salting the checksum.
    The report is primarily pickled on disk to reduce RAM footprint.
    """

    object: CrossValidationReport = Field(repr=False, exclude=True, alias="report")

    @computed_field
    @cached_property
    def checksum(self) -> str:
        """
        Artefact checksum, useful for retrieving the artefact from artefact storage.
        """

        reports = [self.report] + self.report.estimator_reports_
        caches = []

        for report in reports:
            caches.append(report._cache)
            report._cache = {}

        try:
            return upload(self.project, self.report, "cross-validation-report")
        finally:
            for i, report in enumerate(reports):
                report._cache = cache[i]
