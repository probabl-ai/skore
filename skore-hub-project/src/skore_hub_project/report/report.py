"""Class definition of the payload used to send a report to ``hub``."""

from abc import ABC
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property, partial
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from skore_hub_project import Project
from skore_hub_project.artifact.media.media import Media
from skore_hub_project.artifact.pickle import Pickle
from skore_hub_project.metric.metric import Metric
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport
from skore_hub_project import switch_mpl_backend

SkinnedProgress = partial(
    Progress,
    TextColumn("[bold cyan blink]{task.description}..."),
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
            metric = metric_cls(report=self.report)
            metric.compute()

            if metric.value is not None:
                payloads.append(metric)

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
        medias = [
            media_cls(project=self.project, report=self.report)
            for media_cls in self.MEDIAS
        ]

        with (
            switch_mpl_backend(),
            SkinnedProgress() as progress,
            ThreadPoolExecutor() as compute_pool,
            ThreadPoolExecutor(max_workers=6) as upload_pool,
        ):
            tasks = [
                compute_pool.submit((lambda m: m.upload(pool=upload_pool)), media)
                for media in medias
            ]

            try:
                # expliquer pourquoi ça fonctionne
                # même en cas de media avec les memes checksum + rajouter un test
                deque(
                    progress.track(
                        as_completed(tasks),
                        description=f"Uploading {self.report.__class__.__name__} media",
                        total=len(tasks),
                    )
                )
            except BaseException:
                # Cancel all remaining tasks, especially on `KeyboardInterrupt`.
                for task in tasks:
                    task.cancel()

                raise

        return [media for media in medias if media.checksum is not None]

    @computed_field  # type: ignore[prop-decorator]
    @cached_property
    def pickle(self) -> Pickle:
        """
        The checksum of the instance.

        The checksum of the instance that was assigned before being uploaded to the
        artifact storage. It is based on its ``joblib`` serialization and mainly used to
        retrieve it from the artifacts storage.
        """
        with ThreadPoolExecutor(max_workers=6) as upload_pool:
            pickle = Pickle(project=self.project, report=self.report)
            pickle.upload(pool=upload_pool)

        return pickle
