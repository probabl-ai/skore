"""Class definition of the payload used to send a report to ``hub``."""

from abc import ABC
from functools import cached_property
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.pickle import Pickle
from skore_hub_project.metric.metric import Metric
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


class ReportPayload(BaseModel, Generic[Report], ABC):
    """
    Payload used to send a report to ``hub``.

    Attributes
    ----------
    METRICS : ClassVar[tuple[Metric, ...]]
        The metric classes that have to be computed from the report.
    MEDIAS : ClassVar[tuple[Media, ...]]
        The media classes that have to be computed from the report.
    project : Project
        The project to which the report payload should be sent.
    report : EstimatorReport | CrossValidationReport
        The report on which to calculate the payload to be sent.
    key : str
        The key to associate to the report.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    METRICS: ClassVar[tuple[type[Metric], ...]]
    MEDIAS: ClassVar[tuple[type[Media], ...]]

    project: Project = Field(repr=False, exclude=True)
    report: Report = Field(repr=False, exclude=True)
    key: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimator_class_name(self) -> str:
        """The name of the report's estimator."""
        return self.report.estimator_name_

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def dataset_fingerprint(self) -> str:
        """The hash of the targets in the test-set."""
        import joblib

        return joblib.hash(
            self.report.y_test
            if isinstance(self.report, EstimatorReport)
            else self.report.y
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ml_task(self) -> str:
        """The type of ML task covered by the report."""
        return self.report.ml_task

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def metrics(self) -> list[Metric]:
        """
        The list of scalar metrics that have been computed from the report.

        Notes
        -----
        Unavailable metrics have been filtered out.

        All metrics whose value is not a scalar are currently ignored:
        - ignore ``list[float]`` for multi-output ML task,
        - ignore ``dict[str: float]`` for multi-classes ML task.

        The position field is used to drive the ``hub``'s parallel coordinates plot:
        - int [0, inf[, to be displayed at the position,
        - None, not to be displayed.
        """
        payloads = []

        for metric_cls in self.METRICS:
            payload = metric_cls(report=self.report)

            if payload.value is not None:
                payloads.append(payload)

        return payloads

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def medias(self) -> list[Media]:
        """
        The list of medias that have been computed from the report.

        Medias are `pandas.Dataframe`, SVG images, Python dictionaries or HTML string.

        Notes
        -----
        Unavailable medias have been filtered out.
        """
        payloads = []

        for media_cls in self.MEDIAS:
            payload = media_cls(project=self.project, report=self.report)

            if payload.checksum is not None:
                payloads.append(payload)

        return payloads

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def pickle(self) -> Pickle:
        """
        The checksum of the instance.

        The checksum of the instance that was assigned before being uploaded to the
        artifact storage. It is based on its ``joblib`` serialization and mainly used to
        retrieve it from the artifacts storage.
        """
        return Pickle(project=self.project, report=self.report)
