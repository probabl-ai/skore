from abc import ABC
from functools import cached_property
from typing import Any, ClassVar

from pydantic import Field, computed_field

from skore_hub_project import Payload
from skore_hub_project.media.media import Media
from skore_hub_project.metric.metric import Metric


Artefact = Any
Report = Any
Project = Any

# Create protocols for CrossValidationReport and EstimatorReport


class ReportPayload(ABC, Payload):
    METRICS: ClassVar[tuple[Metric]]
    MEDIAS: ClassVar[tuple[Media]]
    project: Project = Field(repr=False, exclude=True)
    report: Report = Field(repr=False, exclude=True)
    upload: bool = Field(default=True, repr=False, exclude=True)
    key: str
    run_id: str

    @computed_field
    @property
    def estimator_class_name(self) -> str:
        """Return the name of the report's estimator."""
        return self.report.estimator_name_

    @computed_field
    @cached_property
    def dataset_fingerprint(self) -> str:
        """Return the hash of the targets in the test-set."""
        import joblib

        return joblib.hash(
            self.report.y_test if hasattr(self.report, "y_test") else self.report.y
        )

    @computed_field
    @property
    def ml_task(self) -> str:
        """Return the type of ML task covered by the report."""
        return self.report._ml_task  # change to `self.report.ml_task` after rebase main

    @computed_field
    @cached_property
    def parameters(self) -> list[Artefact] | None:
        if self.upload:
            return ["<parameters>"]
        return None

    @computed_field
    @cached_property
    def metrics(self) -> list[Metric] | None:
        return [metric(report=self.report) for metric in self.METRICS]  # filter None

    @computed_field
    @property
    def related_items(self) -> list[Media] | None:
        return [media(report=self.report) for media in self.MEDIAS]  # filter None
