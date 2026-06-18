"""Class definition of the payload used to send a report to ``hub``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import ClassVar, Generic, TypeVar

import joblib
from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from skore import THREADABLE, CrossValidationReport, EstimatorReport, console
from skore._plugins.hub.artifact.media.media import Media
from skore._plugins.hub.artifact.pickle import Pickle
from skore._plugins.hub.metric import Metric
from skore._plugins.hub.project.project import Project

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
    console=console,
    transient=True,
    auto_refresh=THREADABLE,
)

Report = TypeVar("Report", bound=(EstimatorReport | CrossValidationReport))


class ReportPayload(BaseModel, ABC, Generic[Report]):
    """
    Payload used to send a report to ``hub``.

    Attributes
    ----------
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
        fingerprint: str = joblib.hash(
            self.report.y_test
            if isinstance(self.report, EstimatorReport)
            else self.report.y
        )
        return fingerprint

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ml_task(self) -> str:
        """The type of ML task covered by the report."""
        return self.report.ml_task

    @computed_field  # type: ignore[prop-decorator]
    @property
    @abstractmethod
    def metrics(self) -> list[Metric[Report]]:
        """The metrics computed from the report."""

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

        with SkinnedProgress() as progress:
            for media_cls in progress.track(
                self.MEDIAS,
                description=(
                    f"Computing/uploading {self.report.__class__.__name__} "
                    f"#{self.report._hash} media"
                ),
                total=len(self.MEDIAS),
            ):
                payload = media_cls(project=self.project, report=self.report)

                # NOTE: Accessing `payload.checksum` lazily uploads the artifact.
                if payload.checksum is not None:
                    payloads.append(payload)

                progress.refresh()

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
