"""Class definition of the payload used to send a report to ``hub``."""

from __future__ import annotations

from abc import ABC
from functools import cached_property, partial
from operator import methodcaller
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar, cast

from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, Field, computed_field
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from skore import THREADABLE, CrossValidationReport, EstimatorReport, console
from skore._plugins.hub.artifact.media.media import Media
from skore._plugins.hub.artifact.pickle import Pickle
from skore._plugins.hub.metric.metric import Metric
from skore._plugins.hub.project.project import Project

if TYPE_CHECKING:
    import httpx

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
        tasks = list(map(delayed(methodcaller("compute")), metrics))

        with SkinnedProgress() as progress:
            for _ in progress.track(
                Parallel(backend="threading")(tasks),
                description=(
                    f"Computing {self.report.__class__.__name__} "
                    f"#{self.report._hash} metrics"
                ),
                total=len(tasks),
            ):
                progress.refresh()

        return [metric for metric in metrics if metric.value is not None]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def medias(self) -> list[Media[Report]]:
        """Pre-uploaded media artifacts.

        Populated by ``upload_artifacts``. Reading this field has no side
        effects: no network, no compute. Returns an empty list if the
        orchestrator hasn't run yet (e.g., direct ``model_dump`` in a test).
        """
        return getattr(self, "_media_instances", [])

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pickle(self) -> Pickle:
        """Pre-uploaded pickle artifact.

        Populated by ``upload_artifacts``. If the orchestrator hasn't run,
        falls back to a fresh ``Pickle`` instance whose ``checksum`` is
        ``None`` (matching the pre-upload state).
        """
        instance: Pickle | None = getattr(self, "_pickle_instance", None)
        if instance is None:
            return Pickle(project=self.project, report=self.report)
        return instance

    def upload_artifacts(
        self,
        hub_client: httpx.Client,
        storage_client: httpx.Client,
    ) -> None:
        """Compute and upload every artifact (medias + pickle) for this report.

        Pipeline:
        1. Realize ``metrics`` (existing parallel implementation).
        2. Compute media artifact content + checksum in parallel.
        3. Compute the pickle artifact on the *main thread* (it temporarily
           mutates ``report._cache`` non-atomically; cannot share the fan-out).
        4. One ``POST /artifacts``, parallel chunk PUTs, one ``POST
           /artifacts/complete``.

        After this returns, each artifact instance carries a ``_plan``
        attribute that the ``medias`` / ``pickle`` computed fields and
        ``Artifact.checksum`` read from during ``model_dump()``.
        """
        from skore._plugins.hub.artifact.pickle import Pickle
        from skore._plugins.hub.artifact.upload import (
            complete_uploads,
            prepare_artifacts,
            request_upload_urls,
            upload_chunks,
        )

        # 1. Metrics.
        _ = self.metrics

        # 2. Media plans, in parallel.
        from skore._plugins.hub.artifact.artifact import Artifact

        media_artifacts: list[Artifact] = [
            media_cls(project=self.project, report=self.report)
            for media_cls in self.MEDIAS
        ]
        media_plans = prepare_artifacts(media_artifacts)

        # 3. Pickle plan, on the main thread.
        pickle_artifact = Pickle(project=self.project, report=self.report)
        pickle_plan = pickle_artifact.local_plan()

        artifacts = media_artifacts + [pickle_artifact]
        plans = media_plans + [pickle_plan]

        # Attach each plan to its artifact for downstream ``checksum`` reads.
        for artifact, plan in zip(artifacts, plans, strict=True):
            object.__setattr__(artifact, "_plan", plan)

        # Keep instance lists for the (later) side-effect-free ``medias`` /
        # ``pickle`` fields to return.
        object.__setattr__(
            self,
            "_media_instances",
            [
                a
                for a, p in zip(media_artifacts, media_plans, strict=True)
                if p is not None
            ],
        )
        object.__setattr__(self, "_pickle_instance", pickle_artifact)

        # 4. Network: batched URL request, parallel PUTs, batched complete.
        non_null_plans = [p for p in plans if p is not None]
        urls = request_upload_urls(hub_client, self.project, non_null_plans)
        etags = upload_chunks(storage_client, non_null_plans, urls)
        complete_uploads(hub_client, self.project, etags)
