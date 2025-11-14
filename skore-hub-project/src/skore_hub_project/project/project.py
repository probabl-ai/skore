"""Class definition of the ``skore`` hub project."""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable
from functools import cached_property, wraps
from operator import itemgetter
from tempfile import TemporaryFile
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)
from urllib.parse import quote

import joblib
import orjson
from httpx import HTTPStatusError
from rich.progress import Progress, TextColumn, TimeElapsedColumn

from skore_hub_project.client.client import Client, HUBClient
from skore_hub_project.protocol import CrossValidationReport, EstimatorReport

P = ParamSpec("P")
R = TypeVar("R")


if TYPE_CHECKING:

    class Metadata(TypedDict):  # noqa: D101
        id: str
        key: str
        date: str
        learner: str
        ml_task: str
        report_type: str
        dataset: str
        rmse: float | None
        log_loss: float | None
        roc_auc: float | None
        fit_time: float | None
        predict_time: float | None
        rmse_mean: float | None
        log_loss_mean: float | None
        roc_auc_mean: float | None
        fit_time_mean: float | None
        predict_time_mean: float | None


def ensure_project_is_created(method: Callable[P, R]) -> Callable[P, R]:
    """Ensure project is created before executing any other operation."""

    @runtime_checkable
    class Project(Protocol):
        created: bool
        quoted_tenant: str
        quoted_name: str

    @wraps(method)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        project = args[0]

        assert isinstance(project, Project), "You can only wrap `Project` methods"

        if not project.created:
            with HUBClient() as hub_client:
                hub_client.post(
                    f"projects/{project.quoted_tenant}/{project.quoted_name}"
                )

            project.created = True

        return method(*args, **kwargs)

    return wrapper


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
    """

    __REPORT_URN_PATTERN = re.compile(
        r"skore:report:(?P<type>(estimator|cross-validation)):(?P<id>.+)"
    )

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
        self.created = False

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
    def quoted_tenant(self) -> str:
        """The quoted tenant of the project."""
        return quote(self.__tenant, safe="")

    @cached_property
    def quoted_name(self) -> str:
        """The quoted name of the project."""
        return quote(self.__name, safe="")

    @ensure_project_is_created
    def put(self, key: str, report: EstimatorReport | CrossValidationReport) -> None:
        """
        Put a key-report pair to the hub project.

        If the key already exists, its last report is modified to point to this new
        report, while keeping track of the report history.

        Parameters
        ----------
        key : str
            The key to associate with ``report`` in the hub project.
        report : skore.EstimatorReport | skore.CrossValidationReport
            The report to associate with ``key`` in the hub project.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        from ..report import CrossValidationReportPayload, EstimatorReportPayload

        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        payload: EstimatorReportPayload | CrossValidationReportPayload

        if isinstance(report, EstimatorReport):
            payload = EstimatorReportPayload(project=self, key=key, report=report)
            endpoint = "estimator-reports"
        elif isinstance(report, CrossValidationReport):
            payload = CrossValidationReportPayload(project=self, key=key, report=report)
            endpoint = "cross-validation-reports"
        else:
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` or "
                f"`skore.CrossValidationReport` (found '{type(report)}')"
            )

        progress = Progress(
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            transient=True,
        )

        with progress:
            task = progress.add_task(
                description=f"[bold red1]Putting [bright_white on red1 blink]{key}"
            )

            payload_dict = payload.model_dump()
            payload_json_bytes = orjson.dumps(
                payload_dict, option=orjson.OPT_NON_STR_KEYS
            )

            with HUBClient() as hub_client:
                hub_client.post(
                    url=f"projects/{self.quoted_tenant}/{self.quoted_name}/{endpoint}",
                    content=payload_json_bytes,
                    headers={
                        "Content-Length": str(len(payload_json_bytes)),
                        "Content-Type": "application/json",
                    },
                )

            progress.update(task, completed=1)

    @ensure_project_is_created
    def get(self, urn: str) -> EstimatorReport | CrossValidationReport:
        """Get a persisted report by its URN."""
        if m := re.match(Project.__REPORT_URN_PATTERN, urn):
            tenant = self.quoted_tenant
            name = self.quoted_name
            type = m["type"]
            id = m["id"]
            url = f"projects/{tenant}/{name}/{type}-reports/{id}"
        else:
            raise ValueError(
                f"URN '{urn}' format does not match '{Project.__REPORT_URN_PATTERN}'"
            )

        # Retrieve presigned URL
        with HUBClient() as hub_client:
            response = hub_client.get(url=url)
            metadata = response.json()
            presigned_url = metadata["pickle"]["presigned_url"]

        report: EstimatorReport | CrossValidationReport

        # Download pickled report before unpickling it.
        #
        # It uses streaming responses that do not load the entire response body into
        # memory at once.
        with (
            TemporaryFile(mode="w+b") as tmpfile,
            Client() as client,
            client.stream(method="GET", url=presigned_url, timeout=30) as response,
        ):
            for data in response.iter_bytes():
                tmpfile.write(data)

            tmpfile.seek(0)

            report = joblib.load(tmpfile)

        return report

    @ensure_project_is_created
    def summarize(self) -> list[Metadata]:
        """Obtain metadata/metrics for all persisted reports in insertion order."""

        def dto(response: Any) -> Metadata:
            report_type, summary = response
            metrics = {
                metric["name"]: metric["value"]
                for metric in summary["metrics"]
                if metric["data_source"] in (None, "test")
            }

            return {
                "id": summary["urn"],
                "key": summary["key"],
                "date": summary["created_at"],
                "learner": summary["estimator_class_name"],
                "ml_task": summary["ml_task"],
                "report_type": report_type,
                "dataset": summary["dataset_fingerprint"],
                "rmse": metrics.get("rmse"),
                "log_loss": metrics.get("log_loss"),
                "roc_auc": metrics.get("roc_auc"),
                "fit_time": metrics.get("fit_time"),
                "predict_time": metrics.get("predict_time"),
                "rmse_mean": metrics.get("rmse_mean"),
                "log_loss_mean": metrics.get("log_loss_mean"),
                "roc_auc_mean": metrics.get("roc_auc_mean"),
                "fit_time_mean": metrics.get("fit_time_mean"),
                "predict_time_mean": metrics.get("predict_time_mean"),
            }

        with HUBClient() as hub_client:
            responses = itertools.chain(
                zip(
                    itertools.repeat("estimator"),
                    hub_client.get(
                        f"projects/{self.quoted_tenant}/{self.quoted_name}/estimator-reports/"
                    ).json(),
                ),
                zip(
                    itertools.repeat("cross-validation"),
                    hub_client.get(
                        f"projects/{self.quoted_tenant}/{self.quoted_name}/cross-validation-reports/"
                    ).json(),
                ),
            )

        return sorted(map(dto, responses), key=itemgetter("date"))

    def __repr__(self) -> str:  # noqa: D105
        return f"Project(mode='hub', name='{self.name}', tenant='{self.tenant}')"

    @staticmethod
    def delete(tenant: str, name: str) -> None:
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
        with HUBClient() as hub_client:
            try:
                hub_client.delete(
                    f"projects/{quote(tenant, safe='')}/{quote(name, safe='')}"
                )
            except HTTPStatusError as e:
                if e.response.status_code == 403:
                    raise PermissionError(
                        f"Failed to delete the project '{name}'; "
                        f"please contact the '{tenant}' owner"
                    ) from e
                raise
