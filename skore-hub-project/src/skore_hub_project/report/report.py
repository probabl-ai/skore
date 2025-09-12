"""Class definition of the payload used to send a report to ``hub``."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import ClassVar, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field

from skore_hub_project import Project
from skore_hub_project.artefact.artefact import Artefact
from skore_hub_project.media.media import Media
from skore_hub_project.metric.metric import Metric
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport


class ReportPayload(ABC, BaseModel):
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
    upload : bool, optional
        Upload the report to the artefacts storage, default True.
    key : str
        The key to associate to the report.
    """

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
        """The current run identifier of the project."""
        return self.project.run_id

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
            self.report.y_test if hasattr(self.report, "y_test") else self.report.y
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ml_task(self) -> str:
        """The type of ML task covered by the report."""
        return self.report.ml_task

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def parameters(self) -> Artefact | dict[()]:
        """
        The checksum of the instance.

        The checksum of the instance that was assigned after being uploaded to the
        artefact storage. It is based on its ``joblib`` serialization and mainly used to
        retrieve it from the artefacts storage.

        .. deprecated
          The ``parameters`` property will be removed in favor of a new ``checksum``
          property in a near future.
        """

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
        payloads = [
            payload
            for metric in self.METRICS
            if (payload := metric(report=self.report)).value is not None
        ]

        return payloads

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def related_items(self) -> list[Media]:
        """
        The list of medias that have been computed from the report.

        Medias are `pandas.Dataframe`, SVG images, Python dictionaries or HTML string.

        Notes
        -----
        Unavailable medias have been filtered out.
        """
        payloads = [
            payload
            for media in cast(list[Callable], self.MEDIAS)
            if (payload := media(report=self.report)).representation is not None
        ]

        return payloads
