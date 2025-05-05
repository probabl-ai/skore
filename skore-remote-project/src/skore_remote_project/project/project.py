"""Class definition of the ``skore`` remote project."""

from __future__ import annotations

from operator import itemgetter
from functools import cached_property
from typing import TYPE_CHECKING

from .. import item as item_module
from ..client.client import AuthenticatedClient


if TYPE_CHECKING:
    from typing import Any, Optional, TypedDict, Union
    from skore import EstimatorReport

    class EstimatorReportMetadata(TypedDict):  # noqa: D101
        id: str
        run_id: str
        key: str
        date: str
        note: Union[str, None]
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
    API to manage a collection of key-value pairs persisted in a remote storage.

    It communicates with the Probabl's ``skore hub`` storage.
    Its constructor initializes a remote project by creating a new project or by
    loading an existing one from a defined tenant.

    The class main method is :func:`~skore_remote_project.Project.put` to insert a
    key-value pair into the Project.

    You can add any type of objects. In some cases, especially on classes you defined,
    the persistency is based on the pickle representation.

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
        Initialize a remote project.

        Initialize a remote project by creating a new project or by loading an existing
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

    def put(self, key: str, value: Any, *, note: Optional[str] = None):
        """
        Put a key-value pair to the remote project.

        If the key already exists, its last value is modified to point to this new
        value, while keeping track of the value history.

        Parameters
        ----------
        key : str
            The key to associate with ``value`` in the remote project.
        value : Any
            The value to associate with ``key`` in the remote project.
        note : str, optional
            A note to attach with the key-value pair, default None.

        Raises
        ------
        TypeError
            If the combination of parameters are not valid.
        """
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not isinstance(note, (type(None), str)):
            raise TypeError(f"Note must be a string (found '{type(note)}')")

        item = item_module.object_to_item(value)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.tenant}/{self.name}/items/",
                json={
                    **item.__metadata__,
                    **item.__representation__,
                    **item.__parameters__,
                    "key": key,
                    "run_id": self.run_id,
                    "note": note,
                },
            )

    @property
    def experiments(self):
        class Namespace:
            @staticmethod
            def __call__(id: int) -> EstimatorReport:
                def dto(report):
                    item_class_name = report["raw"]["class"]
                    item_class = getattr(item_module, item_class_name)
                    item_parameters = report["raw"]["parameters"]
                    item = item_class(**item_parameters)
                    return item.__raw__

                with AuthenticatedClient(raises=True) as client:
                    response = client.get(
                        "/".join(
                            (
                                "projects",
                                self.project.tenant,
                                self.project.name,
                                "experiments",
                                "estimator-reports",
                                id,
                            )
                        )
                    )

                return dto(response.json())

            @staticmethod
            def metadata() -> EstimatorReportMetadata:
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
                        "note": summary["note"],
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
                        "/".join(
                            (
                                "projects",
                                self.tenant,
                                self.name,
                                "experiments",
                                "estimator-reports",
                            )
                        )
                    )

                return sorted(map(dto, response.json()), key=itemgetter("date"))

        return self.run_id and Namespace()
