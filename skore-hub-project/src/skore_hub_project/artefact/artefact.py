from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project import Project
from skore_hub_project.artefact.upload import upload


class Artefact(ABC, BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

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
        cache = self.object._cache
        self.object._cache = {}

        try:
            return upload(self.project, self.object, "estimator-report")
        finally:
            self.object._cache = cache


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
        reports = [self.object] + self.object.estimator_reports_
        caches = []

        for report in reports:
            caches.append(report._cache)
            report._cache = {}

        try:
            return upload(self.project, self.object, "cross-validation-report")
        finally:
            for i, report in enumerate(reports):
                report._cache = caches[i]
