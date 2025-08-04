"""Class definition of the ``skore`` hub project."""

from __future__ import annotations

from functools import cached_property
from operator import itemgetter
from tempfile import TemporaryFile
from types import SimpleNamespace
from typing import TYPE_CHECKING
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic import computed_field
from typing import Any
from abc import abstractmethod

import joblib

from ..client.client import Client, HTTPStatusError, HUBClient
from ..item import skore_estimator_report_item
from . import artefact

if TYPE_CHECKING:
    from typing import TypedDict

    from skore import EstimatorReport

    class Metadata(TypedDict):  # noqa: D101
        id: str
        run_id: str
        key: str
        date: str
        learner: str
        dataset: str
        ml_task: str
        rmse: float | None
        log_loss: float | None
        roc_auc: float | None
        fit_time: float
        predict_time: float


Project = Any
EstimatorReport = Any
Artefact = Any
Metric = Any


class Payload(BaseModel):
    class Config:
        frozen = True

    def todict(self):
        return model.model_dump(exclude_none=True)


class ReportPayload(Payload):
    project: Project = Field(repr=False, exclude=True)
    report: EstimatorReport = Field(repr=False, exclude=True)
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


class EstimatorReportPayload(ReportPayload):
    @computed_field
    @cached_property
    def metrics(self) -> list[Metric] | None:
        return None

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None


class CrossValidationReportPayload(ReportPayload):
    @computed_field
    @cached_property
    def metrics(self) -> list[Metric] | None:
        return None

    @computed_field
    @cached_property
    def medias(self) -> list[Metric] | None:
        return None


model = EstimatorReportPayload(project="<project>", report="<report>")
model.todict()


class CrossValidationReportPayload(EstimatorReportPayload):
    splitting_strategy_name: str
    # for each split and for each sample in the dataset
    # - 0 if the sample is in the train-set,
    # - 1 if the sample is in the test-set.
    splits: list[list[Literal[0, 1]]]
    # index of the group for each sample in the dataset
    groups: list[int] | None = None
    # all class names of the dataset
    class_names: list[str] | None = None
    # class name index for each sample in the dataset
    classes: list[int] | None = None
    #
    estimators: list[EstimatorReportPayload]


from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _CVIterableWrapper


# splitting_strategy_name: str
is_sklearn_splitter = isinstance(report.splitter, BaseCrossValidator)
is_iterable_splitter = isinstance(report.splitter, _CVIterableWrapper)
is_standard_strategy = is_sklearn_splitter and (not is_iterable_splitter)
strategy = (is_standard_strategy and report.splitter.__class__.__name__) or "custom"
# splits: list[list[Literal[0, 1]]]
splits = [[0] * len(report.X)] * len(report.split_indices)

for i, (_, test_indices) in enumerate(report.split_indices):
    for test_indice in test_indices:
        splits[i][test_indice] = 1

# groups: list[int] | None = None
groups = None
# class_names: list[str] | None = None
# classes: list[int] | None = None
if report.ml_task.includes("classification"):
    class_to_class_indice = defaultdict(lambda: len(class_to_class_indice))
    sample_to_class_indice = list(map(class_to_class_indice, report.y))
    classes = list(map(str, class_to_class_indice))
else:
    sample_to_class_indice = None
    classes = None

class_names = classes
classes = sample_to_class_indice

assert len(classes) == len(report.X)
assert max(classes) == len(class_names)
# estimators: list[EstimatorReportPayload]
estimators = list(map(EstimatorReportPayload, report.estimator_reports_))


class Project:
    """
    API to manage a collection of key-report pairs persisted in a hub storage.

    It communicates with the Probabl's ``skore hub`` storage, based on the pickle
    representation. Its constructor initializes a hub project by creating a new
    project or by loading an existing one from a defined tenant.

    The class main methods are :func:`~skore_hub_project.Project.put`,
    :func:`~skore_hub_project.reports.metadata` and
    :func:`~skore_hub_project.reports.get`, respectively to insert a key-report pair
    into the Project, to obtain the reports metadata and to get a specific report.

    Parameters
    ----------
    tenant : str
        The tenant of the project.

        A tenant is a ``skore hub`` concept that must be configured on the
        ``skore hub`` interface. It represents an isolated entity managing users,
        projects, and resources. It can be a company, organization, or team that
        operates independently within the system.
    name : str
        The name of the project.

    Attributes
    ----------
    tenant : str
        The tenant of the project.
    name : str
        The name of the project.
    run_id : str
        The current run identifier of the project.
    """

    def __init__(self, tenant: str, name: str):
        """
        Initialize a hub project.

        Initialize a hub project by creating a new project or by loading an existing
        one from a defined tenant.

        Parameters
        ----------
        tenant : Path
            The tenant of the project.

            A tenant is a ``skore hub`` concept that must be configured on the
            ``skore hub`` interface. It represents an isolated entity managing users,
            projects, and resources. It can be a company, organization, or team that
            operates independently within the system.
        name : str
            The name of the project.
        """
        self.__tenant = tenant
        self.__name = name

    @property
    def tenant(self) -> str:
        """The tenant of the project."""
        return self.__tenant

    @property
    def name(self) -> str:
        """The name of the project."""
        return self.__name

    @cached_property
    def run_id(self) -> str:
        """The current run identifier of the project."""
        with HUBClient() as client:
            request = client.post(f"projects/{self.tenant}/{self.name}/runs")
            run = request.json()

            return run["id"]

    def put(self, key: str, report: EstimatorReport):
        """
        Put a key-report pair to the hub project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the hub project.
        report : skore.EstimatorReport
            The report to associate with ``key`` in the hub project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        from skore import EstimatorReport, CrossValidationReport

        if not isinstance(report, EstimatorReport | CrossValidationReport):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` or `skore.CrossValidationReport`"
                f"(found '{type(report)}')"
            )

        # Upload report to artefacts storage.
        #
        # The report is pickled without its cache, to avoid salting the checksum.
        # The report is pickled on disk to reduce RAM footprint.
        cache = report._cache
        report._cache = {}

        try:
            checksum = artefact.upload(self, report, "report-pickle")
        finally:
            report._cache = cache

        # Send metadata for `EstimatorReport`.
        with HUBClient() as client:
            client.post(
                url=f"projects/{self.tenant}/{self.name}/items",
                json=dict(
                    (
                        ("key", key),
                        ("run_id", self.run_id),
                        ("estimator_class_name", report.estimator_name_),
                        ("dataset_fingerprint", joblib.hash(report.y_test)),
                        ("ml_task", report._ml_task),
                        ("metrics", metrics(report)),
                        ("related_items", artefacts(report)),
                        ("parameters", {"checksum": checksum}),
                    )
                ),
            )

        # Send metadata for `CrossValidationReport`.
        with HUBClient() as client:
            client.post(
                url=f"projects/{self.tenant}/{self.name}/items",
                json=dict(
                    (
                        ("key", key),
                        ("run_id", self.run_id),
                        ("estimator_class_name", report.estimator_name_),
                        ("dataset_fingerprint", joblib.hash(report.y_test)),
                        ("ml_task", report._ml_task),
                        ("metrics", metrics(report)),
                        ("related_items", artefacts(report)),
                        ("parameters", {"checksum": checksum}),
                    )
                ),
            )

    @property
    def reports(self):
        """Accessor for interaction with the persisted reports."""

        def get(id: str) -> EstimatorReport:
            """Get a persisted report by its id."""
            # Retrieve report metadata.
            with HUBClient() as client:
                response = client.get(
                    url=f"projects/{self.tenant}/{self.name}/experiments/estimator-reports/{id}"
                )

            metadata = response.json()
            checksum = metadata["raw"]["checksum"]

            # Ask for read url.
            with HUBClient() as client:
                response = client.get(
                    url=f"projects/{self.tenant}/{self.name}/artefacts/read",
                    params={"artefact_checksum": [checksum]},
                )

            url = response.json()[0]["url"]

            # Download pickled report before unpickling it.
            #
            # It uses streaming responses that do not load the entire response body into
            # memory at once.
            with (
                TemporaryFile(mode="w+b") as tmpfile,
                Client() as client,
                client.stream(method="GET", url=url, timeout=30) as response,
            ):
                for data in response.iter_bytes():
                    tmpfile.write(data)

                tmpfile.seek(0)

                return joblib.load(tmpfile)

        def metadata() -> list[Metadata]:
            """Obtain metadata for all persisted reports regardless of their run."""

            def dto(summary):
                metrics = {
                    metric["name"]: metric["value"]
                    for metric in summary["metrics"]
                    if metric["data_source"] in (None, "test")
                }

                return {
                    "id": summary["id"],
                    "run_id": summary["run_id"],
                    "key": summary["key"],
                    "date": summary["created_at"],
                    "learner": summary["estimator_class_name"],
                    "dataset": summary["dataset_fingerprint"],
                    "ml_task": summary["ml_task"],
                    "rmse": metrics.get("rmse"),
                    "log_loss": metrics.get("log_loss"),
                    "roc_auc": metrics.get("roc_auc"),
                    "fit_time": metrics.get("fit_time"),
                    "predict_time": metrics.get("predict_time"),
                }

            with HUBClient() as client:
                response = client.get(
                    f"projects/{self.tenant}/{self.name}/experiments/estimator-reports"
                )

            return sorted(map(dto, response.json()), key=itemgetter("date"))

        # Ensure project is created by calling `self.run_id`
        return self.run_id and SimpleNamespace(get=get, metadata=metadata)

    def __repr__(self) -> str:  # noqa: D105
        return f"Project(mode='hub', name='{self.name}', tenant='{self.tenant}')"

    @staticmethod
    def delete(tenant: str, name: str):
        """
        Delete a hub project.

        Parameters
        ----------
        tenant : Path
            The tenant of the project.

            A tenant is a ``skore hub`` concept that must be configured on the
            ``skore hub`` interface. It represents an isolated entity managing users,
            projects, and resources. It can be a company, organization, or team that
            operates independently within the system.
        name : str
            The name of the project.
        """
        with HUBClient() as client:
            try:
                client.delete(f"projects/{tenant}/{name}")
            except HTTPStatusError as e:
                if e.response.status_code == 403:
                    raise PermissionError(
                        f"Failed to delete the project; "
                        f"please contact the '{tenant}' owner"
                    ) from e
                raise
