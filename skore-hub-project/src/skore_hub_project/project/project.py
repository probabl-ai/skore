"""Class definition of the ``skore`` hub project."""

from __future__ import annotations

from functools import cached_property
from operator import itemgetter
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .. import item as item_module
from ..client.client import AuthenticatedClient, HTTPStatusError
from ..item.item import lazy_is_instance

if TYPE_CHECKING:
    from typing import TypedDict, Union

    from skore.sklearn import EstimatorReport

    class Metadata(TypedDict):  # noqa: D101
        id: str
        run_id: str
        key: str
        date: str
        learner: str
        dataset: str
        ml_task: str
        rmse: Union[float, None]
        log_loss: Union[float, None]
        roc_auc: Union[float, None]
        fit_time: float
        predict_time: float


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
        with AuthenticatedClient(raises=True) as client:
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

        if not lazy_is_instance(
            report, "skore.sklearn._estimator.report.EstimatorReport"
        ):
            raise TypeError(
                f"Report must be a `skore.EstimatorReport` (found '{type(report)}')"
            )

        item = item_module.object_to_item(report)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.tenant}/{self.name}/items",
                json={
                    **item.__metadata__,
                    **item.__representation__,
                    **item.__parameters__,
                    "key": key,
                    "run_id": self.run_id,
                },
            )

    @property
    def reports(self):
        """Accessor for interaction with the persisted reports."""

        def get(id: str) -> EstimatorReport:
            """Get a persisted report by its id."""

            def dto(report):
                item_class_name = report["raw"]["class"]
                item_class = getattr(item_module, item_class_name)
                item_parameters = report["raw"]["parameters"]
                item = item_class(**item_parameters)
                return item.__raw__

            with AuthenticatedClient(raises=True) as client:
                response = client.get(
                    f"projects/{self.tenant}/{self.name}/experiments/estimator-reports/{id}"
                )

            return dto(response.json())

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

            with AuthenticatedClient(raises=True) as client:
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
        with AuthenticatedClient(raises=True) as client:
            try:
                client.delete(f"projects/{tenant}/{name}")
            except HTTPStatusError as e:
                if e.response.status_code == 403:
                    raise PermissionError(
                        f"Failed to delete the project; "
                        f"please contact the '{tenant}' owner"
                    ) from e
                raise
