"""Class definition of the payload used to send a report to ``hub``."""

from __future__ import annotations

from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property, partial
from typing import ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from skore_hub_project import switch_mpl_backend
from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.pickle import Pickle
from skore_hub_project.metric.metric import Metric
from skore_hub_project.project.project import Project
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

SkinnedProgress = partial(
    Progress,
    TextColumn("[bold cyan]{task.description}..."),
    BarColumn(
        complete_style="dark_orange",
        finished_style="dark_orange",
        pulse_style="orange1",
    ),
    TextColumn("[orange1]{task.percentage:>3.0f}%"),
    TimeElapsedColumn(),
    transient=True,
)

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


class ReportPayload(BaseModel, ABC, Generic[Report]):
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

    METRICS: ClassVar[tuple[type[Metric[Report]], ...]]
    MEDIAS: ClassVar[tuple[type[Media[Report]], ...]]

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

        return cast(
            str,
            joblib.hash(
                self.report.y_test
                if isinstance(self.report, EstimatorReport)
                else self.report.y
            ),
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ml_task(self) -> str:
        """The type of ML task covered by the report."""
        return self.report.ml_task

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def metrics(self) -> list[Metric[Report]]:
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
        self.report.cache_predictions()

        metrics = [metric_cls(report=self.report) for metric_cls in self.METRICS]

        with (
            switch_mpl_backend(),
            SkinnedProgress() as progress,
            ThreadPoolExecutor() as pool,
        ):
            tasks = [
                pool.submit(lambda metric: metric.compute(), metric)
                for metric in metrics
            ]

            for task in progress.track(
                as_completed(tasks),
                description=f"Computing {self.report.__class__.__name__} metrics",
                total=len(tasks),
            ):
                task.result()

        return [metric for metric in metrics if metric.value is not None]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def medias(self) -> list[Media[Report]]:
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
