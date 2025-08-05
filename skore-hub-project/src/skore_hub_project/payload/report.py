from typing import Any

from .base import BasePayload
from .. import Project

Artefact = Any
Metric = Any
Report = Any


# Create protocols for CrossValidationReport and EstimatorReport


class ReportPayload(BasePayload):
    project: Project = Field(repr=False, exclude=True)
    report: Report = Field(repr=False, exclude=True)
    upload: bool = Field(default=True, repr=False, exclude=True)
    key: str
    run_id: int

    @computed_field
    @cached_property
    def estimator_class_name(self) -> str:
        """Return the name of the report's estimator."""
        return self.report.estimator_name_

    @computed_field
    @cached_property
    def dataset_fingerprint(self) -> str:
        """Return the hash of the targets in the test-set."""
        import joblib

        return joblib.hash(self.report.y_test)

    @computed_field
    @cached_property
    def ml_task(self) -> str:
        """Return the type of ML task covered by the report."""
        return self.report._ml_task

    @computed_field
    @cached_property
    def parameters(self) -> list[Artefact] | None:
        if self.upload:
            return ["<parameters>"]
        return None

    @computed_field
    @cached_property
    @abstractmethod
    def metrics(self) -> list[Metric] | None:
        return None

    @computed_field
    @cached_property
    @abstractmethod
    def medias(self) -> list[Metric] | None:
        return None
