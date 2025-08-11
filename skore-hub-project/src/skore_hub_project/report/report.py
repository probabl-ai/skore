from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, computed_field
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project import Project
from skore_hub_project.artefact.artefact import Artefact
from skore_hub_project.media.media import Media
from skore_hub_project.metric.metric import Metric


class ReportPayload(ABC, BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    METRICS: ClassVar[tuple[Metric, ...]]
    MEDIAS: ClassVar[tuple[Media, ...]]

    project: Project = Field(repr=False, exclude=True)
    report: EstimatorReport | CrossValidationReport = Field(repr=False, exclude=True)
    upload: bool = Field(default=True, repr=False, exclude=True)
    key: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def run_id(self) -> int:
        """Return the current run identifier of the project."""
        return self.project.run_id

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimator_class_name(self) -> str:
        """Return the name of the report's estimator."""
        return self.report.estimator_name_

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def dataset_fingerprint(self) -> str:
        """Return the hash of the targets in the test-set."""
        import joblib

        return joblib.hash(
            self.report.y_test if hasattr(self.report, "y_test") else self.report.y
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ml_task(self) -> str:
        """Return the type of ML task covered by the report."""
        return self.report._ml_task  # change to `self.report.ml_task` after rebase main

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def parameters(self) -> Artefact | dict[()]: ...

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def metrics(self) -> list[Metric]:
        payloads = [
            payload
            for metric in self.METRICS
            if (payload := metric(report=self.report)).value is not None
        ]

        return payloads

    @computed_field  # type: ignore[prop-decorator]
    @property
    def related_items(self) -> list[Media]:
        payloads = [
            payload
            for media in self.MEDIAS
            if (payload := media(report=self.report)).representation is not None
        ]

        return payloads
